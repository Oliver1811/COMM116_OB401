"""
agent2.py — Router-based vision agent with dynamic tool specialisation.

Extends the pipeline from agent.py with:

  1. TaskFamily classification — five task families detected by regex from the
     question text (no extra model call needed for simple formulaic questions).
  2. ROUTE stage — lightweight classification + metadata extraction run once
     after OBSERVE, before the main loop.  Can optionally call the LLM via
     _build_router_prompt/_parse_router_output to handle ambiguous questions.
  3. Task-specific code prompt builders — one per family, each injecting
     concrete HSV colour ranges, validated algorithmic templates, and
     family-specific constraints so the code generator has minimal degrees of
     freedom and therefore less room for error.
  4. Stricter code validation:
     - `def` is COMPLETELY BANNED in ALL generated code (flat procedural only).
     - Placeholder / demo array detection in both generation-time and
       execution-time checks.
  5. Extended _agent_result_is_valid for all five task families.

Public API (drop-in replacement for agent.py):
    solve(image_path, question) -> str
    hybrid_solve(image_path, question) -> dict
    get_last_run_stats() -> dict
    is_structured_task(question) -> bool
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image

from model_loader import generate_response
from sandbox import execute
from utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Hard limits (same as agent.py)
# ---------------------------------------------------------------------------

MAX_STEPS: int = 5
SANDBOX_TIMEOUT: float = 4.0
MAX_IMAGE_SIZE: int = 512
MAX_HISTORY: int = 3
MAX_TOOL_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Stage machine + TaskFamily
# ---------------------------------------------------------------------------

class Stage(Enum):
    OBSERVE  = "OBSERVE"
    ROUTE    = "ROUTE"      # NEW: lightweight classification, no image re-encoded
    PLAN     = "PLAN"
    GENERATE = "GENERATE"
    EXECUTE  = "EXECUTE"
    REFLECT  = "REFLECT"
    DONE     = "DONE"


class TaskFamily(Enum):
    COUNT_SHAPES       = "count_shapes"
    MOST_COMMON_COLOUR = "most_common_colour"
    BAR_CHART_MAX      = "bar_chart_max"
    LINE_ANGLE         = "line_angle"
    REGION_FRACTION    = "region_fraction"
    FALLBACK           = "fallback"


@dataclass
class AgentState:
    """Tracks the full mutable state of one solve() call."""

    stage: Stage = Stage.OBSERVE
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    full_history: list[dict[str, Any]] = field(default_factory=list)
    stage_trace: list[str] = field(default_factory=list)

    # Image observation — computed ONCE before the loop, reused every step.
    observation: str | None = None

    # Router output — computed in ROUTE stage, passed to code builders.
    task_family: TaskFamily = TaskFamily.FALLBACK
    route_meta: dict = field(default_factory=dict)

    # Per-step working fields (reset each iteration)
    last_tool_name: str | None = None
    last_plan: str | None = None
    last_code: str | None = None

    # Track whether baseline fallback was used
    used_baseline_fallback: bool = False
    last_tool_result: str | None = None
    last_stdout: str | None = None
    last_error: str | None = None
    last_generation_attempts: int = 1

    # Reflection feedback from the *previous* step.
    last_reflection: str | None = None
    last_error_type: str | None = None

    # Result
    final_answer: str | None = None

    def advance(self, to: Stage) -> None:
        logger.debug("  [state] %s -> %s", self.stage.value, to.value)
        self.stage = to
        self.stage_trace.append(to.value)

    def push_history(self) -> None:
        record = {
            "step": self.step,
            "tool": self.last_tool_name,
            "plan": self.last_plan,
            "code": self.last_code,
            "tool_result": self.last_tool_result,
            "output": self.last_stdout,
            "error": self.last_error,
            "generation_attempts": self.last_generation_attempts,
            "reflection": self.last_reflection,
        }
        self.full_history.append(record)
        self.history.append(record)
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    def reset_step_fields(self) -> None:
        self.last_tool_name = None
        self.last_plan = None
        self.last_code = None
        self.last_tool_result = None
        self.last_stdout = None
        self.last_error = None
        self.last_generation_attempts = 1


_last_run_stats: dict[str, Any] = {
    "steps": 0,
    "final_answer": None,
    "stage_trace": [],
    "tool_steps": [],
    "tool_retry_count": 0,
    "used_baseline_fallback": False,
}

# ---------------------------------------------------------------------------
# Image pre-processing (identical to agent.py)
# ---------------------------------------------------------------------------

def _prepare_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_SIZE:
        return image_path
    scale = MAX_IMAGE_SIZE / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = str(Path("outputs") / ("resized_" + Path(image_path).name))
    Path("outputs").mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_COUNT_RE = re.compile(r"\bhow many\b|\bcount the\b|\bnumber of\b", re.IGNORECASE)
_MOST_COMMON_COLOUR_RE = re.compile(
    r"\bwhich colou?r occurs most often\b|\bmost common colou?r\b",
    re.IGNORECASE,
)
_BAR_CHART_MAX_RE = re.compile(
    r"\bvalue of the (tallest|shortest|highest|lowest) bar\b"
    r"|\b(tallest|shortest|highest|lowest) bar\b"
    r"|\bbar chart\b|\bbar graph\b",
    re.IGNORECASE,
)
_LINE_ANGLE_RE = re.compile(
    r"\bacute angle\b|\bangle between\b|\bangle of the\b|\bdegrees\b",
    re.IGNORECASE,
)
_REGION_FRACTION_RE = re.compile(
    r"\bpercentage of the image\b|\bwhat percentage\b|\bfraction of\b"
    r"|\bpercent.{0,20}covered\b|\bcovered.{0,20}percent\b",
    re.IGNORECASE,
)
# Legacy / compatibility regexes used by is_structured_task
_CHART_RE = re.compile(
    r"\b(tallest bar|shortest bar|highest bar|lowest bar|bar height|bar value"
    r"|value of the|how tall|how high|bar chart|bar graph)\b",
    re.IGNORECASE,
)
_COLOUR_RE = re.compile(
    r"\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|grey|gray|cyan|magenta)\b",
    re.IGNORECASE,
)
_SHAPE_RE = re.compile(
    r"\b(square|circle|triangle|rectangle|oval|diamond|star)s?\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Dataset-specific HSV colour ranges (validated against dev set images)
# Colour space: cv2.COLOR_RGB2HSV  (arr is RGB from PIL/numpy)
# H range: 0-180, S range: 0-255, V range: 0-255
# ---------------------------------------------------------------------------

# Each entry is a list of (lower, upper) tuples for cv2.inRange.
# Red wraps around hue=0/180 and requires two masks OR-ed together.
_HSV_RANGES: dict[str, list[tuple[list[int], list[int]]]] = {
    "red":     [([0,  80, 80], [12, 255, 255]),
                ([168, 80, 80], [180, 255, 255])],   # wrap-around red
    "green":   [([25, 100, 50], [45, 255, 255])],    # H=34 (lime-green in this palette)
    "blue":    [([90, 100, 100], [115, 255, 255])],  # H≈102 (cornflower blue)
    "purple":  [([110,  50, 100], [135, 255, 255])], # H≈118 (lavender-slate)
    "yellow":  [([20, 100, 100], [35, 255, 255])],
    "orange":  [([8,  100, 100], [20, 255, 255])],
    "cyan":    [([85, 100, 100], [100, 255, 255])],
    "magenta": [([155, 80, 100], [175, 255, 255])],
    "pink":    [([0,   30, 150], [12, 255, 255]),
                ([165,  30, 150], [180, 255, 255])],
    "black":   [([0,    0,   0], [180, 50, 80])],
    "white":   [([0,    0, 200], [180, 30, 255])],
    "grey":    [([0,    0,  80], [180, 40, 200])],
    "gray":    [([0,    0,  80], [180, 40, 200])],
    "brown":   [([10,  80,  50], [20, 255, 150])],
}


def _hsv_range_lines(colour: str) -> str:
    """Return Python lines that build a mask for `colour` as a code snippet."""
    ranges = _HSV_RANGES.get(colour.lower(), [])
    if not ranges:
        return f"# No HSV range for '{colour}' — using saturation threshold\nmask = (hsv[:,:,1] > 60).astype(np.uint8) * 255"
    lines = ["hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)"]
    if len(ranges) == 1:
        lo, hi = ranges[0]
        lines.append(f"mask = cv2.inRange(hsv, np.array({lo}), np.array({hi}))")
    else:
        parts = []
        for i, (lo, hi) in enumerate(ranges):
            var = f"m{i+1}"
            lines.append(f"{var} = cv2.inRange(hsv, np.array({lo}), np.array({hi}))")
            parts.append(var)
        lines.append(f"mask = cv2.bitwise_or({', '.join(parts)})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------

def _classify_task_family(question: str) -> TaskFamily:
    """Classify the question into one of the five task families using regex."""
    if _MOST_COMMON_COLOUR_RE.search(question):
        return TaskFamily.MOST_COMMON_COLOUR
    if _COUNT_RE.search(question) and not _REGION_FRACTION_RE.search(question):
        return TaskFamily.COUNT_SHAPES
    if _BAR_CHART_MAX_RE.search(question):
        return TaskFamily.BAR_CHART_MAX
    if _LINE_ANGLE_RE.search(question):
        return TaskFamily.LINE_ANGLE
    if _REGION_FRACTION_RE.search(question):
        return TaskFamily.REGION_FRACTION
    return TaskFamily.FALLBACK


def _extract_route_meta(question: str, task_family: TaskFamily) -> dict:
    """Extract structured metadata from the question using regex.

    Returns a dict with:
        task_family : str    — canonical family name
        colour      : str | None — target colour (lowercase)
        shape       : str | None — target shape (lowercase)
        answer_type : str    — expected output format
    """
    colour_m = _COLOUR_RE.search(question)
    shape_m  = _SHAPE_RE.search(question)
    colour   = colour_m.group(1).lower() if colour_m else None
    shape    = shape_m.group(1).lower()  if shape_m  else None

    answer_type_map = {
        TaskFamily.COUNT_SHAPES:       "integer",
        TaskFamily.MOST_COMMON_COLOUR: "colour_label",
        TaskFamily.BAR_CHART_MAX:      "integer",
        TaskFamily.LINE_ANGLE:         "integer_degrees",
        TaskFamily.REGION_FRACTION:    "integer_percent",
        TaskFamily.FALLBACK:           "free_text",
    }

    return {
        "task_family": task_family.value,
        "colour":      colour,
        "shape":       shape,
        "answer_type": answer_type_map.get(task_family, "free_text"),
    }


def _build_router_prompt(
    question: str,
    observation: str,
    task_family: TaskFamily,
) -> str:
    """Build an LLM prompt to refine routing metadata from observation context.

    For simple formulaic questions this is not called (regex is sufficient).
    Exposed as a utility for future use with ambiguous questions.
    """
    return (
        f"You are a task analyser. Extract structured metadata from the question.\n\n"
        f"Question: {question}\n\n"
        f"Image observation: {observation[:400]}\n\n"
        f"Preliminary task family: {task_family.value}\n\n"
        f"Output EXACTLY these fields (one per line):\n"
        f"  TASK_FAMILY: {task_family.value}\n"
        f"  TARGET_COLOUR: <colour name or 'none'>\n"
        f"  TARGET_SHAPE: <shape name or 'none'>\n"
        f"  TARGET_QUERY: <brief query descriptor>\n"
        f"  ANSWER_TYPE: <integer | colour_label | integer_degrees | integer_percent | free_text>\n"
    )


def _parse_router_output(text: str) -> dict:
    """Parse the structured router LLM output back into a route_meta dict."""
    meta: dict[str, Any] = {}
    for key in ("TASK_FAMILY", "TARGET_COLOUR", "TARGET_SHAPE", "TARGET_QUERY", "ANSWER_TYPE"):
        m = re.search(rf"{key}\s*:\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            meta[key.lower()] = val if val not in ("none", "n/a", "-") else None
    return meta

# ---------------------------------------------------------------------------
# Helper functions (identical to agent.py)
# ---------------------------------------------------------------------------

def _is_most_common_colour_question(question: str) -> bool:
    return bool(_MOST_COMMON_COLOUR_RE.search(question))


def _is_counting_question(question: str) -> bool:
    return bool(_COUNT_RE.search(question))


def _history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["Previous observations:"]
    for rec in history:
        lines.append(f"  [Step {rec['step']}]")
        if rec.get("tool"):
            lines.append(f"    TOOL: {rec['tool']}")
        if rec.get("reflection"):
            lines.append(f"    REFLECTION (why this step ran): {rec['reflection']}")
        if rec.get("plan"):
            lines.append(f"    PLAN: {rec['plan']}")
        if rec.get("tool_result") is not None:
            lines.append(f"    TOOL_RESULT: {rec['tool_result']}")
        out = (rec.get("output") or "")[:400]
        if out:
            lines.append(f"    OUTPUT: {out}")
        if rec.get("error"):
            lines.append(f"    ERROR: {rec['error']}")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Prompt builders — observation, plan
# ---------------------------------------------------------------------------

def _build_pure_observation_prompt() -> str:
    return (
        "Describe this image factually and neutrally.\n"
        "State only visible content. Do not infer, estimate, or solve any task.\n"
        "Include:\n"
        "  - Visible objects, shapes, colours, and approximate positions\n"
        "  - Spatial relationships between objects\n"
        "  - Any visible text, numbers, axes, legends, or tick marks\n"
        "  - If this is a chart, name the chart type and describe the visible bars, lines, or segments\n"
        "Rules:\n"
        "  - Do NOT answer any question\n"
        "  - Do NOT guess hidden or ambiguous details\n"
        "  - Do NOT give a final count for any category unless the exact count is plainly visible as text in the image\n"
        "  - Prefer short factual statements over interpretation"
    )


def _build_plan_prompt(
    question: str,
    observation: str,
    state: "AgentState",
    task_hints: str,
) -> str:
    is_structured = state.task_family != TaskFamily.FALLBACK

    if state.last_reflection and state.history:
        return _build_repair_plan_prompt(
            question, observation, state, task_hints, is_structured
        )

    carry_forward = ""
    if state.history:
        last = state.history[-1]
        if last.get("output") and last["output"] not in ("(no output)", "(no code generated)"):
            carry_forward = (
                f"\nThe previous step produced: {last['output']}\n"
                f"Use this result in your plan.\n"
            )

    if is_structured:
        final_answer_clause = (
            f"You MUST write a Python plan — do NOT write FINAL_ANSWER here.\n"
            f"The answer must be computed by code, not estimated from the description.\n\n"
        )
    else:
        final_answer_clause = (
            f"If the observation already contains a clear, unambiguous answer, write:\n"
            f"  FINAL_ANSWER: <value>\n\n"
            f"Otherwise write a numbered plan (3-5 steps) for Python code:\n"
        )

    family_hint = ""
    if state.task_family != TaskFamily.FALLBACK:
        family_hint = f"Task family: {state.task_family.value}\n"
        if state.route_meta.get("colour"):
            family_hint += f"Target colour: {state.route_meta['colour']}\n"
        if state.route_meta.get("shape"):
            family_hint += f"Target shape: {state.route_meta['shape']}\n"
        family_hint += "\n"

    return (
        f"You are a planning agent. Decide how to answer a visual question using Python.\n\n"
        f"Image observation:\n{observation}\n\n"
        f"Question: {question}\n\n"
        f"{family_hint}"
        f"{task_hints}"
        f"{_history_block(state.history)}"
        f"{carry_forward}"
        f"{final_answer_clause}"
        f"Write a numbered plan with 3 to 5 steps.\n\n"
        f"Required output format:\n"
        f"  TOOL_NAME: <short_function_name>\n"
        f"  PLAN:\n"
        f"  1. <plain English step>\n"
        f"  2. <plain English step>\n"
        f"  3. <plain English step>\n"
        f"  4. Print the final answer\n\n"
        f"PLAN RULES:\n"
        f"  - Plain English only\n"
        f"  - No Python syntax\n"
        f"  - No variable names\n"
        f"  - No assignments\n"
        f"  - No function names or library calls\n"
        f"  - Describe operations in plain English only\n"
        f"  - The final step must say to print the answer\n\n"
        f"Output only TOOL_NAME and PLAN (no code)."
    )


def _build_repair_plan_prompt(
    question: str,
    observation: str,
    state: "AgentState",
    task_hints: str,
    is_structured: bool,
) -> str:
    last = state.history[-1]
    prev_plan  = (last.get("plan")  or "(unknown)").strip()[:400]
    prev_code  = (last.get("code")  or "(no code generated)").strip()[:500]
    prev_error = (last.get("error") or "").strip()[:250]
    reflection_text = state.last_reflection or ""
    steps_left = MAX_STEPS - state.step

    # Detect repetitive failure patterns
    repetition_warning = ""
    if len(state.history) >= 2:
        # Check for repeated error patterns
        current_error_sig = prev_error.lower()[:100] if prev_error else ""
        repeated_errors = []
        repeated_codes = []
        
        for i, past in enumerate(state.history[:-1]):
            past_error = (past.get("error") or "").strip()[:100].lower()
            past_code = (past.get("code") or "").strip()[:200]
            
            # Detect same error type
            if current_error_sig and past_error:
                if (current_error_sig in past_error or past_error in current_error_sig):
                    repeated_errors.append(i + 1)
            
            # Detect similar code patterns (especially syntax errors)
            if "syntax error" in current_error_sig.lower() and prev_code and past_code:
                # Check for very similar code (first 150 chars)
                if prev_code[:150] == past_code[:150]:
                    repeated_codes.append(i + 1)
        
        if repeated_errors or repeated_codes:
            repetition_warning = (
                f"\n{'=' * 60}\n"
                f"⚠️  CRITICAL: REPETITIVE FAILURE DETECTED\n"
                f"{'=' * 60}\n"
            )
            if repeated_errors:
                repetition_warning += (
                    f"You have encountered this SAME error type in step(s): {repeated_errors}\n"
                )
            if repeated_codes:
                repetition_warning += (
                    f"You have generated IDENTICAL code in step(s): {repeated_codes}\n"
                )
            repetition_warning += (
                f"\n🚨 YOU ARE STUCK IN A LOOP. The approach you keep trying DOES NOT WORK.\n\n"
                f"MANDATORY NEW STRATEGY:\n"
                f"  • Use a COMPLETELY DIFFERENT algorithmic approach\n"
                f"  • If you were using cv2.findContours, try pixel scanning instead\n"
                f"  • If you were using thresholding, try color-based filtering\n"
                f"  • If you were using morphology, try direct array slicing\n"
                f"  • Think: what would a human manually count/measure in this image?\n"
                f"  • Simplify your approach dramatically\n\n"
                f"{'=' * 60}\n\n"
            )

    no_fa_clause = (
        "You MUST write a new Python plan. Do NOT write FINAL_ANSWER here.\n\n"
        if is_structured else
        "Do NOT write FINAL_ANSWER unless the answer is already proven by prior successful execution.\n\n"
    )

    family_hint = ""
    if state.task_family != TaskFamily.FALLBACK:
        family_hint = f"Task family: {state.task_family.value}\n"
        if state.route_meta.get("colour"):
            family_hint += f"Target colour: {state.route_meta['colour']}\n"
        if state.route_meta.get("shape"):
            family_hint += f"Target shape: {state.route_meta['shape']}\n"
        family_hint += "\n"

    return (
        f"REPAIR PLAN — step {state.step}/{MAX_STEPS} "
        f"with {steps_left} remaining after this step.\n\n"
        f"A previous attempt failed. Write a different plan that avoids the same mistake.\n\n"
        f"{repetition_warning}"
        f"Question: {question}\n\n"
        f"Image observation:\n{observation}\n\n"
        f"{family_hint}"
        f"Previous failed plan:\n{prev_plan}\n\n"
        f"Previous failed code excerpt:\n{prev_code}\n\n"
        f"Failure reason:\n{reflection_text or prev_error or '(unknown)'}\n\n"
        f"{task_hints}"
        f"{no_fa_clause}"
        f"Write a NEW numbered plan with 3 to 5 steps.\n\n"
        f"Required output format:\n"
        f"  TOOL_NAME: <short_function_name>\n"
        f"  PLAN:\n"
        f"  1. <plain English step>\n"
        f"  2. <plain English step>\n"
        f"  3. <plain English step>\n"
        f"  4. Print the final answer\n\n"
        f"REPAIR RULES:\n"
        f"  - Plain English only\n"
        f"  - No Python syntax\n"
        f"  - No variable names\n"
        f"  - No library names\n"
        f"  - No assignments\n"
        f"  - Create a different approach than before\n"
        f"  - The plan must lead to executable code, not a guessed answer\n"
        f"  - The last step must say to print the answer\n\n"
        f"Focus on the failure reason above and produce the smallest viable corrected plan.\n"
        f"Output only TOOL_NAME and PLAN."
    )

# ---------------------------------------------------------------------------
# Task-specific code prompt builders
# Each builder injects verified algorithmic scaffolding and concrete HSV ranges
# so the generator has minimal room to introduce errors.
# ---------------------------------------------------------------------------

_CODEGEN_HEADER = """\
You must write executable Python code for a real image task.

RUNTIME ENVIRONMENT:
  arr — pre-loaded NumPy RGB image (already in memory, ready to use).
  img — pre-loaded PIL image (already in memory, ready to use).
  cv2, np, math, Image are already imported.

REQUIRED CONSTRAINTS:
  - Write flat procedural code (no function definitions).
  - No imports (libraries already available).
  - Use arr directly for OpenCV operations.
  - Create new variables for derived images: hsv, gray, mask, edges, etc.
  - Comments must be on their own line or at the end of complete statements.
  - Final line MUST be: print(<answer>)

CORRECT CODE PATTERNS:
  # Starting with pre-loaded arr:
  hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
  gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
  
  # Creating derived images:
  mask = (hsv[:,:,1] > 60).astype(np.uint8) * 255
  edges = cv2.Canny(gray, 30, 100)
  
  # Comments on their own line:
  # Process the mask
  cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  
  # Or at end of statement:
  h, w = arr.shape[:2]  # image dimensions
"""


def _build_count_shapes_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    colour = route_meta.get("colour") or ""
    shape  = route_meta.get("shape")  or ""

    hsv_block = _hsv_range_lines(colour) if colour else (
        "hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)\n"
        "# No specific colour — detect all saturated regions\n"
        "mask = (hsv[:,:,1] > 60).astype(np.uint8) * 255"
    )

    shape_filter = ""
    if shape == "triangle":
        shape_filter = (
            "# Keep only triangle-like contours: 3 vertices after approximation\n"
            "valid = []\n"
            "for c in cnts:\n"
            "    if cv2.contourArea(c) > 80:\n"
            "        peri = cv2.arcLength(c, True)\n"
            "        approx = cv2.approxPolyDP(c, 0.04 * peri, True)\n"
            "        if len(approx) == 3:\n"
            "            valid.append(c)"
        )
    elif shape == "square":
        shape_filter = (
            "# Keep only square-like contours: aspect ratio 0.7 – 1.4\n"
            "valid = [c for c in cnts if cv2.contourArea(c) > 80 and\n"
            "         0.7 <= cv2.boundingRect(c)[2] / max(cv2.boundingRect(c)[3], 1) <= 1.4]"
        )
    elif shape == "circle":
        shape_filter = (
            "# Keep only circle-like contours: circularity > 0.65\n"
            "valid = [c for c in cnts if cv2.contourArea(c) > 80 and\n"
            "         4*np.pi*cv2.contourArea(c) / max(cv2.arcLength(c, True)**2, 1e-6) > 0.65]"
        )
    elif shape == "rectangle":
        shape_filter = (
            "# Keep rectangular contours (area vs bounding box > 0.6)\n"
            "valid = [c for c in cnts if cv2.contourArea(c) > 80 and\n"
            "         cv2.contourArea(c) / max(cv2.boundingRect(c)[2]*cv2.boundingRect(c)[3], 1) > 0.6]"
        )
    else:
        shape_filter = "valid = [c for c in cnts if cv2.contourArea(c) > 80]"

    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    # Skip heavy morphology for triangles (it distorts sharp corners)
    if shape == "triangle":
        morph_block = "# Find contours directly (morphology distorts sharp corners)"
    else:
        morph_block = (
            "# Morphological cleanup to fill gaps and remove noise\n"
            "k = np.ones((5, 5), np.uint8)\n"
            "mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)\n"
            "mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)"
        )

    return (
        f"{_CODEGEN_HEADER}\n"
        f"TASK: Count {colour or 'all'} {shape or 'shape'} objects in the image.\n"
        f"Expected output: a single integer.\n\n"
        f"Question: {question}\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n"
        f"ALGORITHM TEMPLATE (adapt to the actual image — do NOT copy placeholder arrays):\n"
        f"```python\n"
        f"{hsv_block}\n"
        f"{morph_block}\n"
        f"cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
        f"{shape_filter}\n"
        f"print(len(valid))\n"
        f"```\n"
        f"Write valid Python syntax. Comments on their own line or at end of statements.\n"
        f"Use arr directly (already loaded). Flat procedural code.\n"
    )


def _build_most_common_colour_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    shape = route_meta.get("shape") or "shape"

    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    return (
        f"{_CODEGEN_HEADER}\n"
        f"TASK: Among the {shape} objects in the image, which colour (red/green/blue/purple)\n"
        f"occurs most often? Output exactly one colour word.\n\n"
        f"Question: {question}\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n"
        f"ALGORITHM TEMPLATE:\n"
        f"Colour palette for this dataset (cv2.COLOR_RGB2HSV hue values):\n"
        f"  red:    H in [0,12] or [168,180], S >= 80\n"
        f"  green:  H in [25,45], S >= 100\n"
        f"  blue:   H in [90,115], S >= 100\n"
        f"  purple: H in [110,135], S >= 50\n\n"
        f"```python\n"
        f"hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)\n"
        f"# Detect all {shape} objects by saturation\n"
        f"sat = (hsv[:,:,1] > 50).astype(np.uint8) * 255\n"
        f"k = np.ones((3, 3), np.uint8)\n"
        f"sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, k)\n"
        f"cnts, _ = cv2.findContours(sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n"
        f"valid = [c for c in cnts if cv2.contourArea(c) > 200]\n"
        f"counts = {{'red': 0, 'green': 0, 'blue': 0, 'purple': 0}}\n"
        f"for c in valid:\n"
        f"    M = cv2.moments(c)\n"
        f"    if M['m00'] == 0:\n"
        f"        continue\n"
        f"    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])\n"
        f"    cy = max(0, min(cy, arr.shape[0]-1))\n"
        f"    cx = max(0, min(cx, arr.shape[1]-1))\n"
        f"    h_val = int(hsv[cy, cx, 0])\n"
        f"    s_val = int(hsv[cy, cx, 1])\n"
        f"    if s_val < 50:\n"
        f"        continue\n"
        f"    if 110 <= h_val <= 135:\n"
        f"        counts['purple'] += 1\n"
        f"    elif 90 <= h_val <= 115:\n"
        f"        counts['blue'] += 1\n"
        f"    elif 25 <= h_val <= 45:\n"
        f"        counts['green'] += 1\n"
        f"    else:\n"
        f"        counts['red'] += 1\n"
        f"print(max(counts, key=counts.get))\n"
        f"```\n"
        f"Write valid Python syntax. Comments on their own line or at end of statements.\n"
        f"Use arr directly (already loaded). Flat procedural code.\n"
    )


def _build_bar_chart_max_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    return (
        f"{_CODEGEN_HEADER}\n"
        f"TASK: Find the integer value of the tallest bar in the bar chart.\n"
        f"Expected output: a single integer.\n\n"
        f"Question: {question}\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n"
        f"ALGORITHM TEMPLATE (validated for this dataset's bar chart images):\n"
        f"Key insight: each data unit corresponds to a fixed number of pixels.\n"
        f"Find the pixel height of each bar, then use best-fit integer scaling to\n"
        f"recover the data value.\n\n"
        f"```python\n"
        f"hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)\n"
        f"# Detect coloured bars by saturation (excludes gray axes/ticks)\n"
        f"colored = (hsv[:, :, 1] > 50).astype(np.uint8)\n"
        f"# Find baseline: last row containing any bar pixel\n"
        f"rows_with_bar = np.where(colored.any(axis=1))[0]\n"
        f"baseline = int(rows_with_bar.max())\n"
        f"# Measure bar heights by scanning columns\n"
        f"bar_heights = []\n"
        f"in_bar = False\n"
        f"col_tops = []\n"
        f"for c in range(arr.shape[1]):\n"
        f"    col = colored[:baseline, c]\n"
        f"    if col.any():\n"
        f"        col_tops.append(baseline - int(np.argmax(col)))\n"
        f"        in_bar = True\n"
        f"    else:\n"
        f"        if in_bar and col_tops:\n"
        f"            bar_heights.append(max(col_tops))\n"
        f"            col_tops = []\n"
        f"            in_bar = False\n"
        f"if in_bar and col_tops:\n"
        f"    bar_heights.append(max(col_tops))\n"
        f"# Best-fit integer scale: for each candidate y_max, compare rounding error\n"
        f"H_max = max(bar_heights)\n"
        f"best_y, best_err = 1, float('inf')\n"
        f"for y in range(1, 16):\n"
        f"    unit = H_max / y\n"
        f"    err = sum(abs(h / unit - round(h / unit)) for h in bar_heights)\n"
        f"    if err < best_err:\n"
        f"        best_err = err\n"
        f"        best_y = y\n"
        f"print(best_y)\n"
        f"```\n"
        f"Write valid Python syntax. Comments on their own line or at end of statements.\n"
        f"Use arr directly (already loaded). Flat procedural code.\n"
    )


def _build_line_angle_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    colour = route_meta.get("colour") or "black"
    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    return (
        f"{_CODEGEN_HEADER}\n"
        f"TASK: Find the acute angle (integer degrees) between the two {colour} lines.\n"
        f"Expected output: a single integer (degrees).\n\n"
        f"Question: {question}\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n"
        f"ALGORITHM TEMPLATE (validated on this dataset):\n"
        f"```python\n"
        f"gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)\n"
        f"edges = cv2.Canny(gray, 30, 100)\n"
        f"lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)\n"
        f"angles = []\n"
        f"for ln in lines:\n"
        f"    x1, y1, x2, y2 = ln[0]\n"
        f"    if x2 != x1 or y2 != y1:\n"
        f"        angles.append(float(np.degrees(np.arctan2(y2-y1, x2-x1))) % 180)\n"
        f"angles = sorted(angles)\n"
        f"# Split into two groups by finding the largest angular gap\n"
        f"diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]\n"
        f"split = int(np.argmax(diffs)) + 1\n"
        f"a1 = float(np.median(angles[:split]))\n"
        f"a2 = float(np.median(angles[split:]))\n"
        f"diff = abs(a2 - a1)\n"
        f"acute = min(diff, 180.0 - diff)\n"
        f"print(round(acute))\n"
        f"```\n"
        f"Write valid Python syntax. Comments on their own line or at end of statements.\n"
        f"Use arr directly (already loaded). Flat procedural code.\n"
    )


def _build_region_fraction_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    colour = route_meta.get("colour") or ""
    hsv_block = _hsv_range_lines(colour) if colour else (
        "hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)\n"
        "mask = (hsv[:,:,1] > 60).astype(np.uint8) * 255"
    )

    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    return (
        f"{_CODEGEN_HEADER}\n"
        f"TASK: Find what percentage of the image (0-100) is covered by the {colour or 'target'} region.\n"
        f"Expected output: a single integer.\n\n"
        f"Question: {question}\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n"
        f"ALGORITHM TEMPLATE:\n"
        f"```python\n"
        f"{hsv_block}\n"
        f"# Dilate 1px to capture anti-aliased edges\n"
        f"k = np.ones((3, 3), np.uint8)\n"
        f"mask = cv2.dilate(mask, k, iterations=1)\n"
        f"total = arr.shape[0] * arr.shape[1]\n"
        f"pct = round(int(mask.sum()) / 255 * 100 / total)\n"
        f"print(pct)\n"
        f"```\n"
        f"Write valid Python syntax. Comments on their own line or at end of statements.\n"
        f"Use arr directly (already loaded). Flat procedural code.\n"
    )


def _build_fallback_code_prompt(
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    """Fallback for non-structured or unclassified tasks (mirrors agent.py prompt)."""
    correction_block = f"\nPREVIOUS FAILURE TO FIX:\n{correction_hint}\n" if correction_hint else ""

    return (
        f"You must write executable Python code for the real task below.\n\n"
        f"IMPORTANT: arr already contains the correct image input for this task.\n"
        f"Use arr directly - it is already loaded and ready to process.\n\n"
        f"Question: {question}\n\n"
        f"Plan:\n{plan}\n"
        f"{correction_block}\n\n"
        f"RUNTIME ENVIRONMENT:\n"
        f"  - arr: NumPy RGB image array (already loaded)\n"
        f"  - img: PIL image (already loaded)\n"
        f"  - Available: cv2, np, math, Image\n\n"
        f"REQUIRED:\n"
        f"  - Write flat procedural code (no functions)\n"
        f"  - No imports (already available)\n"
        f"  - Use arr as read-only input\n"
        f"  - Store results in new variables\n"
        f"  - End with: print(<result>)\n\n"
        f"CORRECT CODE EXAMPLES:\n"
        f"  # Start by processing arr:\n"
        f"  hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)\n"
        f"  gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)\n"
        f"  \n"
        f"  # Create masks and derived images:\n"
        f"  mask = (hsv[:,:,1] > 60).astype(np.uint8) * 255\n"
        f"  edges = cv2.Canny(gray, 30, 100)\n"
        f"  \n"
        f"  # Extract results:\n"
        f"  count = len(contours)\n"
        f"  print(count)\n\n"
        f"Output only a ```python ... ``` code block.\n"
        f"```python\n"
    )


def _get_code_prompt(
    task_family: TaskFamily,
    question: str,
    plan: str,
    route_meta: dict,
    correction_hint: str = "",
) -> str:
    """Dispatch to the appropriate task-specific code prompt builder."""
    builders = {
        TaskFamily.COUNT_SHAPES:       _build_count_shapes_code_prompt,
        TaskFamily.MOST_COMMON_COLOUR: _build_most_common_colour_code_prompt,
        TaskFamily.BAR_CHART_MAX:      _build_bar_chart_max_code_prompt,
        TaskFamily.LINE_ANGLE:         _build_line_angle_code_prompt,
        TaskFamily.REGION_FRACTION:    _build_region_fraction_code_prompt,
        TaskFamily.FALLBACK:           _build_fallback_code_prompt,
    }
    builder = builders.get(task_family, _build_fallback_code_prompt)
    return builder(question, plan, route_meta, correction_hint)

# ---------------------------------------------------------------------------
# Task hints (brief plain-English guidance for the PLAN stage)
# ---------------------------------------------------------------------------

def _build_task_hints(question: str, task_family: TaskFamily, route_meta: dict) -> str:
    colour = route_meta.get("colour") or ""
    shape  = route_meta.get("shape")  or "shape"

    if task_family == TaskFamily.COUNT_SHAPES:
        return (
            f"\nCOUNTING TASK: Count {colour or 'target-colour'} {shape} objects.\n"
            f"  1. Use the pre-loaded image only.\n"
            f"  2. Isolate {colour} regions by colour masking.\n"
            f"  3. Detect distinct contours in the masked region.\n"
            f"  4. Filter small noise (contours with tiny area).\n"
            f"  5. Print only the final integer count.\n"
            f"Rules: treat arr as immutable; do not guess from observation.\n\n"
        )

    if task_family == TaskFamily.MOST_COMMON_COLOUR:
        return (
            f"\nCATEGORICAL COLOUR TASK: Find the most common colour among {shape} objects.\n"
            f"  1. Detect all {shape} objects using contour detection.\n"
            f"  2. For each object, sample its centre pixel in HSV.\n"
            f"  3. Classify each object as red, green, blue, or purple.\n"
            f"  4. Count and print the most frequent label.\n"
            f"Rules: treat arr as immutable; do not hardcode the answer.\n\n"
        )

    if task_family == TaskFamily.BAR_CHART_MAX:
        return (
            f"\nBAR CHART TASK: Find the integer value of the tallest bar.\n"
            f"  1. Detect coloured bar regions by saturation.\n"
            f"  2. Measure bar heights in pixels (from baseline to bar top).\n"
            f"  3. Use best-fit integer scaling to recover the data value.\n"
            f"  4. Print the integer.\n"
            f"Rules: treat arr as immutable; do not read axis labels as text.\n\n"
        )

    if task_family == TaskFamily.LINE_ANGLE:
        return (
            f"\nLINE ANGLE TASK: Find the acute angle (integer degrees) between two lines.\n"
            f"  1. Detect line segments using Hough transform.\n"
            f"  2. Cluster the detected angles into two groups.\n"
            f"  3. Compute the acute angle between the two line directions.\n"
            f"  4. Print the integer closest to the true angle.\n"
            f"Rules: treat arr as immutable.\n\n"
        )

    if task_family == TaskFamily.REGION_FRACTION:
        return (
            f"\nREGION FRACTION TASK: Find what percentage of the image is covered by {colour or 'target'} region.\n"
            f"  1. Isolate {colour} region using colour masking.\n"
            f"  2. Count foreground pixels.\n"
            f"  3. Divide by total pixels, multiply by 100, round to integer.\n"
            f"  4. Print the integer percentage.\n"
            f"Rules: treat arr as immutable.\n\n"
        )

    return ""  # FALLBACK — no specific hints

# ---------------------------------------------------------------------------
# Reflection prompt (identical to agent.py, minor extension for new families)
# ---------------------------------------------------------------------------

_REFLECT_ERROR_TYPES = (
    "syntax_error",
    "sandbox_violation",
    "stateless_assumption",
    "incomplete_code",
    "wrong_method",
    "no_output",
)


def _build_reflection_prompt(state: "AgentState", question: str) -> str:
    all_obs = []
    for rec in state.history:
        code  = (rec.get("code") or "(no code)")[:600]
        out   = (rec.get("tool_result") or rec.get("output") or "")[:200]
        error = (rec.get("error") or "")[:300]
        plan  = (rec.get("plan") or "")[:200]
        tool  = rec.get("tool") or f"step_{rec['step']}_tool"
        block = (
            f"[Step {rec['step']}] TOOL: {tool}\n"
            f"PLAN: {plan}\n"
            f"CODE:\n{code}\n"
            f"OUTPUT: {out or '(none)'}\n"
            f"ERROR:  {error or '(none)'}"
        )
        all_obs.append(block)

    obs_block = "\n\n".join(all_obs) if all_obs else "(none)"

    colour_match = _COLOUR_RE.search(question)
    shape_match  = _SHAPE_RE.search(question)
    task_requirements = "\nTASK REQUIREMENTS:\n"
    if colour_match:
        task_requirements += f"- Must detect colour: {colour_match.group(1)} using HSV masking\n"
    if shape_match:
        task_requirements += f"- Must detect shape: {shape_match.group(1)} using contour filtering\n"
    if not colour_match and not shape_match:
        task_requirements += "- General visual reasoning task\n"

    last        = state.history[-1] if state.history else {}
    last_error  = last.get("error") or ""
    last_output = last.get("output") or last.get("tool_result") or ""
    code_failed = bool(last_error) or not last_output or last_output in ("(no output)", "(no code generated)")

    error_type_legend = (
        "  syntax_error         — Python syntax is invalid\n"
        "  sandbox_violation    — code tried to imread/open/plt.show/redefine arr/use def\n"
        "  stateless_assumption — code assumed state (variables, files) that don't exist\n"
        "  incomplete_code      — code is missing steps (no print, empty body, etc.)\n"
        "  wrong_method         — method ran but approach is incorrect for this task\n"
        "  no_output            — code ran but printed nothing\n"
    )

    if code_failed:
        return (
            f"You are a code diagnostic agent. The latest code step failed or produced no output.\n"
            f"Question: {question}\n\n"
            f"Step history (newest last):\n{obs_block}\n\n"
            f"ERROR TYPE LEGEND:\n{error_type_legend}\n"
            f"Respond with EXACTLY these three lines and nothing else:\n"
            f"  ERROR_TYPE: <one token from the legend above>\n"
            f"  FIX: <one concrete sentence describing the exact code correction needed>\n"
            f"  NEXT_ACTION: <one sentence telling the planner what to write differently>\n\n"
            f"Rules:\n"
            f"  - Do NOT write FINAL_ANSWER — the code has not produced a valid result yet.\n"
            f"  - FIX and NEXT_ACTION must be specific, not vague.\n"
            f"  - Do not add any other text before or after the three lines."
        )

    colour_rules = ""
    if _is_most_common_colour_question(question):
        colour_rules = (
            f"  - Must have detected shapes first (contours), then classified colour per shape.\n"
            f"  - Must NOT have hardcoded a colour name.\n"
            f"  - Final answer must be one of: red, green, blue, purple.\n"
        )

    return (
        f"You are evaluating whether the previous steps correctly answered the question.\n"
        f"Question: {question}\n\n"
        f"All steps so far:\n{obs_block}\n"
        f"{task_requirements}\n"
        f"VALIDATION RULES:\n"
        f"  - If colour was required but HSV masking was NOT used → wrong_method.\n"
        f"  - If shape was required but contour filtering was NOT used → wrong_method.\n"
        f"  - If result > 50 and task is counting → likely pixel counting → wrong_method.\n"
        f"  - If result is a small integer (0–20) from contour counting → accept.\n"
        f"  - If result is 0 and method is valid → accept (0 is a valid count).\n"
        f"{colour_rules}\n"
        f"ERROR TYPE LEGEND:\n{error_type_legend}\n"
        f"If the result is CORRECT, respond with EXACTLY:\n"
        f"  FINAL_ANSWER: <value>\n"
        f"  FIX: n/a\n"
        f"  NEXT_ACTION: n/a\n\n"
        f"If the result is WRONG, respond with EXACTLY:\n"
        f"  ERROR_TYPE: <token>\n"
        f"  FIX: <one concrete sentence>\n"
        f"  NEXT_ACTION: <one sentence for the planner>\n\n"
        f"Rules:\n"
        f"  - Do NOT mix FINAL_ANSWER with ERROR_TYPE.\n"
        f"  - FINAL_ANSWER must be the bare answer value only.\n"
        f"  - Do not add any other text."
    )

# ---------------------------------------------------------------------------
# Response parsers (identical to agent.py)
# ---------------------------------------------------------------------------

def _parse_generation(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tool_name": None,
        "plan": None,
        "code": None,
        "final_answer": None,
    }

    fa = re.search(r"FINAL_ANSWER\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if fa:
        val = fa.group(1).strip()
        if not re.match(
            r"^(no answer|unable to|cannot be|not enough|i cannot|n/a|none|unknown|no answer provided)",
            val, re.IGNORECASE,
        ):
            result["final_answer"] = val
            return result

    tool = re.search(r"TOOL(?:_NAME)?\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
    if tool:
        result["tool_name"] = tool.group(1).strip().lower()

    maybe_json = re.search(r"\{.*\}", text, re.DOTALL)
    if maybe_json:
        try:
            parsed = json.loads(maybe_json.group(0))
            if isinstance(parsed, dict):
                result["tool_name"] = result["tool_name"] or parsed.get("tool_name")
                result["plan"] = parsed.get("plan") or result["plan"]
                result["code"] = parsed.get("code") or result["code"]
        except json.JSONDecodeError:
            pass

    plan = re.search(
        r"PLAN\s*:\s*(.+?)(?=\nCODE:|\n```|\nFINAL_ANSWER:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if plan:
        result["plan"] = plan.group(1).strip()

    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        result["code"] = fence.group(1).strip()
        return result

    label = re.search(r"CODE\s*:\s*\n((?:[ \t]+\S.+\n?|[ \t]*\n)*)", text, re.IGNORECASE)
    if label:
        code = textwrap.dedent(label.group(1)).strip()
        if code:
            result["code"] = code
            return result

    markers = ("import ", "from ", "print(", "def ", "img =", "arr =", "np.")
    if any(m in text for m in markers):
        start = min((text.find(m) for m in markers if m in text), default=0)
        result["code"] = text[start:].strip()

    return result


_REFLECTION_NON_ANSWERS = re.compile(
    r"^(yes|no|none|unknown|n/a|i don|the image|based on|from the|given that)",
    re.IGNORECASE,
)


def _parse_reflection(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "final_answer":    None,
        "continue_reason": None,
        "error_type":      None,
        "fix":             None,
        "next_action":     None,
    }

    fa_m  = re.search(r"FINAL_ANSWER\s*:\s*([^\n]+)",  text, re.IGNORECASE)
    et_m  = re.search(r"ERROR_TYPE\s*:\s*([^\n]+)",    text, re.IGNORECASE)
    fix_m = re.search(r"FIX\s*:\s*([^\n]+)",           text, re.IGNORECASE)
    na_m  = re.search(r"NEXT_ACTION\s*:\s*([^\n]+)",   text, re.IGNORECASE)

    fix_val = fix_m.group(1).strip() if fix_m  else None
    na_val  = na_m.group(1).strip()  if na_m   else None
    if fix_val and fix_val.lower() in ("n/a", "na", "none", "-"):
        fix_val = None
    if na_val and na_val.lower() in ("n/a", "na", "none", "-"):
        na_val = None

    result["fix"]         = fix_val
    result["next_action"] = na_val

    if fa_m:
        value   = fa_m.group(1).strip().strip(".")
        cleaned = value.upper()
        if (
            value
            and cleaned not in ("CONTINUE", "N/A", "NA")
            and len(value.split()) <= 5
            and not _REFLECTION_NON_ANSWERS.match(value)
        ):
            result["final_answer"] = value
            return result

    if et_m:
        et = et_m.group(1).strip().lower()
        result["error_type"] = et if et in _REFLECT_ERROR_TYPES else et
        parts = [p for p in (fix_val, na_val) if p]
        if parts:
            result["continue_reason"] = " ".join(parts)
        else:
            result["continue_reason"] = f"[{et}] Reflection produced no FIX/NEXT_ACTION detail."
        return result

    cont = re.search(r"CONTINUE\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if cont:
        result["continue_reason"] = cont.group(1).strip()
        return result

    result["continue_reason"] = "Reflection returned unparseable output; retry with a simpler approach."
    return result

# ---------------------------------------------------------------------------
# Code validation
# ---------------------------------------------------------------------------

def _validate_code(code: str) -> str | None:
    """Return None if code is valid Python, or an error string."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def _validate_code_semantics(code: str) -> str | None:
    """Return None if code passes semantic checks, or a planner-facing error string.

    agent2-specific: `def` is COMPLETELY BANNED in all generated code.
    The generator must write flat procedural code only.
    """
    # ----- 1. def banned entirely -----
    if re.search(r"^\s*def\s+", code, re.MULTILINE):
        return (
            "Function definitions ('def') are not allowed. "
            "Write flat procedural code at the top level only. "
            "Do not wrap logic in a function — write it directly."
        )

    # ----- 2. Must reference pre-loaded image variables -----
    if "arr" not in code and "img" not in code:
        return (
            "Code does not reference 'arr' or 'img'. "
            "Use the pre-loaded variables — do not load the image yourself."
        )

    # ----- 3. Must produce output -----
    if "print(" not in code:
        return "Code never calls print(). The final line must be print(<answer>)."

    # ----- 4. Must not reload the image -----
    if "cv2.imread(" in code:
        return "Code calls cv2.imread() — remove it and use 'arr' directly."
    if "Image.open(" in code:
        return "Code calls Image.open() — remove it and use 'img' directly."
    if "PIL.Image.open(" in code:
        return "Code calls PIL.Image.open() — remove it and use 'img' directly."

    # ----- 5. Must not reassign arr (AST check) -----
    try:
        _tree = ast.parse(code)
        for _node in ast.walk(_tree):
            _targets: list = []
            if isinstance(_node, ast.Assign):
                _targets = _node.targets
            elif isinstance(_node, (ast.AugAssign, ast.AnnAssign)):
                _targets = [_node.target]
            elif isinstance(_node, ast.NamedExpr):
                _targets = [_node.target]
            for _target in _targets:
                for _sub in ast.walk(_target):
                    if isinstance(_sub, ast.Name) and _sub.id == "arr":
                        return (
                            "'arr' is a read-only pre-loaded variable and must not be reassigned. "
                            "Use a different variable name (e.g. 'hsv', 'gray', 'mask', 'tmp')."
                        )
    except SyntaxError:
        pass

    # ----- 6. No display calls -----
    if "plt.show(" in code:
        return "Code calls plt.show() — remove it; the sandbox has no display."

    # ----- 7. Placeholder / demo array detection -----
    placeholder_patterns = [
        r"np\.array\(\s*\[.*?\]\s*\)",          # inline literal arrays
        r"np\.zeros\(\s*\(\s*\d+.*?\)\s*\)",    # np.zeros with literal shape
        r"np\.ones\(\s*\(\s*\d+.*?\)\s*\)",     # np.ones with literal shape
        r"np\.random\.",                         # random arrays
    ]
    for pat in placeholder_patterns:
        if re.search(pat, code, re.DOTALL):
            # Only flag if the array is assigned to a name that looks image-like
            if re.search(
                r"(?:img|arr|image|test|dummy|sample|fake|mock)\s*=\s*np\.",
                code, re.IGNORECASE,
            ):
                return (
                    "Code creates a placeholder/demo array instead of using the real image. "
                    "Use 'arr' directly — it already contains the loaded image."
                )

    # ----- 8. Only imports / comments — no real computation -----
    meaningful = [
        ln.strip() for ln in code.splitlines()
        if ln.strip()
        and not ln.strip().startswith("#")
        and not ln.strip().startswith("import ")
        and not ln.strip().startswith("from ")
    ]
    if not meaningful:
        return "Code contains only imports and/or comments — no computation was written."

    return None


def _check_sandbox_violations(code: str, attempt: int = 1) -> str | None:
    """Check for sandbox contract violations (generation-time, triggers regeneration).
    
    Args:
        attempt: Retry attempt number (1-indexed). Higher attempts get stronger hints.
    """
    # Strip comments to avoid false positives when model writes "# do not use cv2.imread"
    code_lines = []
    for line in code.splitlines():
        # Remove inline comments
        if '#' in line:
            code_part = line[:line.index('#')]
        else:
            code_part = line
        if code_part.strip():
            code_lines.append(code_part)
    code_no_comments = '\n'.join(code_lines)
    
    if "cv2.imread(" in code_no_comments:
        if attempt >= 2:
            return (
                "FORBIDDEN: Do NOT write cv2.imread(). arr already contains the image. "
                "Start with: hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)"
            )
        return "Use 'arr' directly instead of loading an image with cv2.imread()."
    
    if re.search(r"(?<!PIL\.)Image\.open\(|PIL\.Image\.open\(", code_no_comments):
        if attempt >= 2:
            return (
                "FORBIDDEN: Do NOT write Image.open(). arr already contains the image. "
                "Start with: hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)"
            )
        return "Use 'img' directly instead of loading an image with Image.open()."
    
    if "plt.imread(" in code_no_comments:
        if attempt >= 2:
            return (
                "FORBIDDEN: Do NOT write plt.imread(). arr already contains the image. "
                "Start with: hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)"
            )
        return "Use 'arr' directly instead of loading an image with plt.imread()."

    # Ban def at generation time
    if re.search(r"^\s*def\s+", code, re.MULTILINE):
        if attempt >= 2:
            return (
                "CRITICAL: Function definitions are BANNED. "
                "Do NOT use 'def'. Write sequential lines of code without any function wrappers. "
                "Replace def blocks with direct inline code."
            )
        return (
            "Remove all function definitions ('def'). "
            "Write flat procedural code only — no def, no nested functions."
        )

    # Detect placeholder/demo arrays assigned to image-like names
    if re.search(
        r"(?:img|arr|image|test|dummy|sample|fake|mock)\s*=\s*np\."
        r"(?:array|zeros|ones|random)",
        code, re.IGNORECASE,
    ):
        if attempt >= 2:
            return (
                "CRITICAL: Do NOT create fake arrays like np.zeros or np.array([[...]]). "
                "The variable 'arr' ALREADY contains the real image — use it directly."
            )
        return (
            "Do not create placeholder or demo arrays. "
            "Use the pre-loaded 'arr' variable which already contains the real image data."
        )

    # Arr reassignment (AST)
    try:
        _tree = ast.parse(code)
        for _node in ast.walk(_tree):
            _targets: list = []
            if isinstance(_node, ast.Assign):
                _targets = _node.targets
            elif isinstance(_node, (ast.AugAssign, ast.AnnAssign)):
                _targets = [_node.target]
            elif isinstance(_node, ast.NamedExpr):
                _targets = [_node.target]
            for _target in _targets:
                for _sub in ast.walk(_target):
                    if isinstance(_sub, ast.Name) and _sub.id == "arr":
                        return (
                            "'arr' is read-only. Do not assign to it. "
                            "Store all derived images in new variables (hsv, gray, mask, tmp)."
                        )
    except SyntaxError:
        pass

    return None

# ---------------------------------------------------------------------------
# Utility helpers (identical to agent.py)
# ---------------------------------------------------------------------------

def _strip_fa_prefix(text: str) -> str:
    m = re.match(r"(?:FINAL_ANSWER|RESULT)\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text


def _extract_tool_result(stdout: str) -> str | None:
    tagged = re.findall(r"RESULT\s*:\s*([^\n]+)", stdout, re.IGNORECASE)
    if tagged:
        return tagged[-1].strip()
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if len(lines) == 1:
        return lines[0]
    if lines:
        return lines[-1]
    return None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_last_run_stats() -> dict[str, Any]:
    return _last_run_stats.copy()


def solve(image_path: str, question: str, baseline_fallback: str | None = None) -> str:
    """Answer *question* about *image_path* using the router-based pipeline:

      OBSERVE (once, question-agnostic)
        └─► ROUTE (classify task family, extract metadata — no model call)
              └─► loop ─► PLAN ─► GENERATE ─► EXECUTE ─► REFLECT

    Args:
        baseline_fallback: if provided, used as the answer when the agent loop
            exhausts without a valid result — ensures the agent never scores
            worse than baseline.
    """
    global _last_run_stats

    state = AgentState()
    prepared_path = _prepare_image(image_path)

    # ── OBSERVE ───────────────────────────────────────────────────────────────
    state.advance(Stage.OBSERVE)
    obs_messages = [
        {
            "role": "system",
            "content": (
                "You are a visual analyst. "
                "Describe images accurately, thoroughly, and without bias."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image_path": prepared_path},
                {"type": "text", "text": _build_pure_observation_prompt()},
            ],
        },
    ]
    try:
        state.observation = generate_response(obs_messages)
    except Exception as exc:
        logger.error("Observation call failed: %s", exc)
        state.observation = "(observation unavailable)"
    logger.info("  [OBSERVE] %s", state.observation[:300])

    # ── ROUTE (fast regex — no model call) ───────────────────────────────────
    state.advance(Stage.ROUTE)
    state.task_family = _classify_task_family(question)
    state.route_meta  = _extract_route_meta(question, state.task_family)
    logger.info(
        "  [ROUTE] family=%s  colour=%s  shape=%s",
        state.task_family.value,
        state.route_meta.get("colour"),
        state.route_meta.get("shape"),
    )

    # Pre-compute task hints for PLAN stage
    task_hints = _build_task_hints(question, state.task_family, state.route_meta)

    # ── Main loop: PLAN → GENERATE → EXECUTE → REFLECT ────────────────────────
    for step in range(1, MAX_STEPS + 1):
        state.step = step
        state.reset_step_fields()
        logger.info("====== Step %d/%d ======", step, MAX_STEPS)

        # ── PLAN ──────────────────────────────────────────────────────────────
        state.advance(Stage.PLAN)
        plan_prompt = _build_plan_prompt(question, state.observation, state, task_hints)
        plan_messages = [
            {
                "role": "system",
                "content": (
                    "You are a planning agent. "
                    "Given a visual observation and a question, write a numbered plan "
                    "for Python code to compute the answer. "
                    "Do NOT write code — only the plan steps."
                ),
            },
            {"role": "user", "content": plan_prompt},
        ]
        try:
            plan_response = generate_response(plan_messages)
        except Exception as exc:
            logger.error("Plan call failed at step %d: %s", step, exc)
            plan_response = ""

        plan_parsed = _parse_generation(plan_response)
        logger.info("  [PLAN] %s", (plan_parsed.get("plan") or plan_response)[:200])

        if plan_parsed.get("final_answer"):
            state.final_answer = _strip_fa_prefix(plan_parsed["final_answer"])
            state.advance(Stage.DONE)
            logger.info("  [PLAN] FINAL_ANSWER: %s", state.final_answer)
            break

        state.last_tool_name = plan_parsed.get("tool_name") or f"tool_step_{step}"
        state.last_plan = plan_parsed.get("plan") or plan_response.strip()

        # ── GENERATE ──────────────────────────────────────────────────────────
        state.advance(Stage.GENERATE)
        code_prompt = _get_code_prompt(
            state.task_family,
            question,
            state.last_plan,
            state.route_meta,
            correction_hint=state.last_reflection or "",
        )
        _gen_sys = {
            "role": "system",
            "content": (
                "You are a Python code writer. "
                "Always respond with a ```python ... ``` code block. "
                "Never answer without code."
            ),
        }
        gen_messages = [_gen_sys, {"role": "user", "content": code_prompt}]

        gen_parsed: dict[str, Any] = {
            "tool_name": None, "plan": None, "code": None, "final_answer": None,
        }
        attempts_used = 1
        gen_response = ""
        last_violation: str | None = None

        for attempt in range(1, MAX_TOOL_RETRIES + 1):
            attempts_used = attempt
            try:
                gen_response = generate_response(gen_messages)
            except Exception as exc:
                logger.error("Generation call failed at step %d: %s", step, exc)
                break

            gen_parsed = _parse_generation(gen_response)
            if not gen_parsed.get("code"):
                retry_prompt = code_prompt + "\n\nIMPORTANT: Your response MUST contain a ```python``` code block."
                gen_messages = [_gen_sys, {"role": "user", "content": retry_prompt}]
                logger.info("  [GENERATE] retry %d/%d (no code)", attempt, MAX_TOOL_RETRIES)
                continue

            violation = _check_sandbox_violations(gen_parsed["code"], attempt=attempt)
            if violation:
                last_violation = violation
                logger.info("  [GENERATE] sandbox violation attempt %d: %s", attempt, violation)
                correction_prompt = _get_code_prompt(
                    state.task_family, question, state.last_plan,
                    state.route_meta, correction_hint=violation,
                )
                gen_messages = [_gen_sys, {"role": "user", "content": correction_prompt}]
                gen_parsed["code"] = None
                continue

            break  # clean code

        state.last_generation_attempts = attempts_used
        state.last_code = gen_parsed.get("code")
        logger.info("  [TOOL] %s  [CODE]\n%s", state.last_tool_name, state.last_code or "(none)")

        if gen_parsed.get("final_answer"):
            state.final_answer = _strip_fa_prefix(gen_parsed["final_answer"])
            state.advance(Stage.DONE)
            logger.info("  [GENERATE] early FINAL_ANSWER: %s", state.final_answer)
            break

        if not state.last_code:
            # Call reflection LLM to diagnose why code generation failed
            state.last_stdout = "(no code generated)"
            state.push_history()  # Commit history before reflection call
            
            ref_prompt = _build_reflection_prompt(state, question)
            ref_messages = [{"role": "user", "content": ref_prompt}]
            try:
                ref_response = generate_response(ref_messages)
                ref_parsed = _parse_reflection(ref_response)
                logger.info(
                    "  [REFLECT-NO-CODE] error_type=%s  fix=%.120s",
                    ref_parsed.get("error_type") or "—",
                    ref_parsed.get("fix") or ref_parsed.get("continue_reason") or "—",
                )
                fix = ref_parsed.get("fix") or ""
                na = ref_parsed.get("next_action") or ""
                fallback = ref_parsed.get("continue_reason") or ""
                if fix or na:
                    state.last_reflection = " ".join(p for p in (fix, na) if p)
                elif fallback:
                    state.last_reflection = fallback
                else:
                    state.last_reflection = "Write a simpler plan: at most 4 lines of Python ending with print(answer)."
                state.last_error_type = ref_parsed.get("error_type")
            except Exception as exc:
                logger.error("Reflection call failed for no-code case at step %d: %s", step, exc)
                # Fallback to hardcoded messages
                if last_violation:
                    state.last_reflection = (
                        f"All code attempts were rejected: {last_violation} "
                        f"Write flat procedural code, no def, no imports, no arr reassignment."
                    )
                    state.last_error_type = "sandbox_violation"
                else:
                    state.last_reflection = "Write a simpler plan: at most 4 lines of Python ending with print(answer)."
            continue

        # ── EXECUTE ───────────────────────────────────────────────────────────
        state.advance(Stage.EXECUTE)
        syntax_err = _validate_code(state.last_code)
        if syntax_err:
            state.last_stdout = None
            state.last_error  = syntax_err
            logger.info("  [EXECUTE] syntax error: %s", syntax_err)
            state.last_reflection = (
                f"The generated code had a syntax error: {syntax_err}. "
                f"Revise the plan so the code step avoids this mistake."
            )
            state.advance(Stage.REFLECT)
            state.push_history()
            continue

        semantic_err = _validate_code_semantics(state.last_code)
        if semantic_err:
            state.last_stdout = None
            state.last_error  = semantic_err
            logger.info("  [EXECUTE] semantic error: %s", semantic_err)
            state.last_reflection = (
                f"Code rejected: {semantic_err} "
                f"Write flat procedural code using 'arr' directly. No def, no imports."
            )
            state.advance(Stage.REFLECT)
            state.push_history()
            continue

        exec_result = execute(
            state.last_code,
            context={"image_path": prepared_path},
            timeout=SANDBOX_TIMEOUT,
        )

        if exec_result["success"]:
            stdout = exec_result["stdout"].strip() or "(no output)"
            tool_result = _extract_tool_result(stdout)
            state.last_tool_result = tool_result

            fa_in_stdout = re.search(r"FINAL_ANSWER\s*:\s*([^\n]+)", stdout, re.IGNORECASE)
            if fa_in_stdout:
                state.final_answer = fa_in_stdout.group(1).strip()
                state.last_stdout  = stdout
                state.last_error   = None
                state.push_history()
                state.advance(Stage.DONE)
                logger.info("  [EXECUTE] FINAL_ANSWER in stdout: %s", state.final_answer)
                break

            # Direct accept: counting tasks (clean integer)
            if (
                state.task_family in (TaskFamily.COUNT_SHAPES, TaskFamily.BAR_CHART_MAX,
                                      TaskFamily.LINE_ANGLE, TaskFamily.REGION_FRACTION)
                and tool_result
                and re.fullmatch(r"-?\d+", tool_result)
            ):
                state.final_answer = tool_result
                state.last_stdout  = stdout
                state.last_error   = None
                state.push_history()
                state.advance(Stage.DONE)
                logger.info("  [EXECUTE] direct FINAL_ANSWER (integer): %s", state.final_answer)
                break

            # Direct accept: colour label tasks
            allowed_colours = {"red", "green", "blue", "purple"}
            if (
                state.task_family == TaskFamily.MOST_COMMON_COLOUR
                and tool_result
                and tool_result.lower() in allowed_colours
            ):
                state.final_answer = tool_result.lower()
                state.last_stdout  = stdout
                state.last_error   = None
                state.push_history()
                state.advance(Stage.DONE)
                logger.info("  [EXECUTE] direct FINAL_ANSWER (colour): %s", state.final_answer)
                break

            # Legacy: counting questions that did not match task_family route
            if _is_counting_question(question) and tool_result and re.fullmatch(r"-?\d+", tool_result):
                state.final_answer = tool_result
                state.last_stdout  = stdout
                state.last_error   = None
                state.push_history()
                state.advance(Stage.DONE)
                logger.info("  [EXECUTE] direct FINAL_ANSWER (count legacy): %s", state.final_answer)
                break

            state.last_stdout = stdout
            state.last_error  = None

        else:
            state.last_tool_result = None
            state.last_stdout      = None
            state.last_error       = exec_result["error"]

        logger.info(
            "  [EXECUTE] %s",
            (state.last_stdout or f"ERROR: {state.last_error}")[:200],
        )

        # ── REFLECT ───────────────────────────────────────────────────────────
        state.advance(Stage.REFLECT)
        state.push_history()

        # Always call reflection LLM (even when no code was generated) to detect patterns
        ref_prompt   = _build_reflection_prompt(state, question)
        ref_messages = [{"role": "user", "content": ref_prompt}]
        try:
            ref_response = generate_response(ref_messages)
        except Exception as exc:
            logger.error("Reflection call failed at step %d: %s", step, exc)
            # Fallback to hardcoded message when LLM completely fails
            no_code_at_all = state.last_code is None or not str(state.last_code).strip()
            if no_code_at_all:
                state.last_reflection = (
                    "The code generator returned no code. "
                    "Write a simpler plan: at most 4 lines of Python ending with print(answer)."
                )
            continue

        ref_parsed = _parse_reflection(ref_response)
        logger.debug("  reflect: %.200s", ref_response)
        logger.info(
            "  [REFLECT] error_type=%s  fix=%.120s",
            ref_parsed.get("error_type") or "—",
            ref_parsed.get("fix") or ref_parsed.get("continue_reason") or "—",
        )

        if ref_parsed["final_answer"] and ref_parsed["final_answer"].strip().upper() != "CONTINUE":
            state.final_answer = _strip_fa_prefix(ref_parsed["final_answer"])
            state.advance(Stage.DONE)
            logger.info("  [REFLECT] FINAL_ANSWER: %s", state.final_answer)
            break
        else:
            fix      = ref_parsed.get("fix") or ""
            na       = ref_parsed.get("next_action") or ""
            fallback = ref_parsed.get("continue_reason") or ""
            if fix or na:
                reason = " ".join(p for p in (fix, na) if p)
            elif fallback:
                reason = fallback
            else:
                reason = "Reflection gave no actionable fix; try a completely different approach."
            state.last_reflection = reason
            state.last_error_type = ref_parsed.get("error_type")
            logger.info("  [REFLECT] CONTINUE → planner: %s", reason)

    # ── Fallback: loop exhausted ───────────────────────────────────────────────
    if state.final_answer is None:
        numeric_outputs = []
        for rec in state.history:
            out = (rec.get("tool_result") or rec.get("output") or "").strip()
            if re.fullmatch(r"-?\d+", out):
                numeric_outputs.append(out)
        if numeric_outputs and len(set(numeric_outputs)) == 1:
            state.final_answer = numeric_outputs[-1]
        elif baseline_fallback is not None:
            state.final_answer = baseline_fallback
            state.used_baseline_fallback = True
            logger.info("  [FALLBACK] loop exhausted — using baseline answer: %s", baseline_fallback)
        else:
            state.final_answer = "Unable to determine the answer."

    _last_run_stats = {
        "steps": state.step,
        "final_answer": state.final_answer,
        "observation": state.observation,
        "stage_trace": state.stage_trace,
        "tool_steps": state.full_history,
        "tool_retry_count": sum(
            max(0, int(rec.get("generation_attempts", 1)) - 1)
            for rec in state.full_history
        ),
        "task_family": state.task_family.value,
        "route_meta": state.route_meta,
        "used_baseline_fallback": state.used_baseline_fallback,
    }
    logger.info(
        "Done in %d step(s) | family=%s | trace: %s | answer: %s",
        state.step,
        state.task_family.value,
        " -> ".join(state.stage_trace),
        state.final_answer,
    )
    return state.final_answer


# ---------------------------------------------------------------------------
# Hybrid public API
# ---------------------------------------------------------------------------

def is_structured_task(question: str) -> bool:
    """Return True if the question benefits from tool-based (code) solving."""
    family = _classify_task_family(question)
    return family != TaskFamily.FALLBACK


def _agent_result_is_valid(stats: dict[str, Any], question: str) -> bool:
    """Return True only when the agent produced a reliable, tool-executed answer."""
    final = (stats.get("final_answer") or "").strip()
    if not final or final == "Unable to determine the answer.":
        return False

    tool_steps = stats.get("tool_steps", [])
    successful = [
        s for s in tool_steps
        if s.get("output")
        and s["output"] not in ("(no output)", "(no code generated)")
        and not s.get("error")
    ]
    if not successful:
        return False

    family_str = stats.get("task_family", "")
    try:
        family = TaskFamily(family_str)
    except ValueError:
        family = _classify_task_family(question)

    if family in (TaskFamily.COUNT_SHAPES, TaskFamily.BAR_CHART_MAX,
                  TaskFamily.LINE_ANGLE, TaskFamily.REGION_FRACTION):
        return bool(re.fullmatch(r"\d+", final))

    if family == TaskFamily.MOST_COMMON_COLOUR:
        return final.lower() in {"red", "green", "blue", "purple"}

    # Legacy checks for questions not classified by new families
    if _is_counting_question(question) or _CHART_RE.search(question):
        return bool(re.fullmatch(r"\d+", final))
    if _is_most_common_colour_question(question):
        return final.lower() in {"red", "green", "blue", "purple"}

    # Fallback structured task: trust the tool output
    return True


def hybrid_solve(image_path: str, question: str) -> dict[str, Any]:
    """Hybrid solver: baseline-first with selective agent override.

    Strategy:
    1. Always obtain a baseline answer (cheap, reliable for open-ended questions).
    2. For structured tasks also run the agent loop.
    3. Use the agent answer only if it produced a valid, tool-executed result.
       Otherwise fall back to baseline.
    """
    from baseline import baseline_solve as _baseline_solve  # lazy import

    baseline_answer = _baseline_solve(image_path, question)
    logger.info("[hybrid] baseline → %s", baseline_answer[:80])

    if not is_structured_task(question):
        logger.info("[hybrid] non-structured task — using baseline")
        return {
            "prediction": baseline_answer,
            "source": "baseline",
            "reason": "non-structured task; agent skipped",
            "baseline": baseline_answer,
            "agent_stats": {},
        }

    agent_answer = solve(image_path, question)
    stats = get_last_run_stats()
    logger.info("[hybrid] agent    → %s  (%d steps)", agent_answer[:80], stats.get("steps", 0))

    if _agent_result_is_valid(stats, question):
        reason = (
            f"tool-executed answer ({stats.get('steps', 0)} step(s), "
            f"{stats.get('tool_retry_count', 0)} retry(s))"
        )
        logger.info("[hybrid] ACCEPT agent result: %s", agent_answer[:80])
        return {
            "prediction": agent_answer,
            "source": "agent",
            "reason": reason,
            "baseline": baseline_answer,
            "agent_stats": stats,
        }

    logger.info("[hybrid] REJECT agent result — falling back to baseline")
    return {
        "prediction": baseline_answer,
        "source": "baseline",
        "reason": "agent ran but produced no valid tool-executed answer",
        "baseline": baseline_answer,
        "agent_stats": stats,
    }
