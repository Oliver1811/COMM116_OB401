"""
run_eval.py — Evaluation harness for the vision agent and baseline.

Usage
-----
  python run_eval.py --data dev.jsonl --out outputs/dev_run
  python run_eval.py --data dev.jsonl --out outputs/dev_run --max-samples 20
  python run_eval.py --data dev.jsonl --out outputs/dev_run --sample-index 5
  python run_eval.py --data dev.jsonl --out outputs/dev_run --agent-only -v

Outputs (written to <out>/)
----------------------------
  predictions.jsonl          {"id": "...", "prediction": "..."}
  baseline_predictions.jsonl same format for baseline
  agent_detailed.jsonl       full per-sample record with runtime, GT, etc.
  baseline_detailed.jsonl    same for baseline
  metrics.json               accuracy, runtime, failure-rate, avg steps
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np

import importlib

from baseline import baseline_solve
from utils import Timer, load_jsonl, resolve_image_path, save_jsonl, setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Sample field accessors (flexible key names)
# ---------------------------------------------------------------------------

def _get_field(sample: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return default


def _get_id(sample: dict, fallback_idx: int) -> str:
    return str(_get_field(sample, "id", "question_id", "sample_id", "idx")
               or fallback_idx)


def _get_question(sample: dict) -> str:
    q = _get_field(sample, "question", "query", "text", "q")
    if q is None:
        raise KeyError(f"No question key in sample: {list(sample.keys())}")
    return str(q)


def _get_answer(sample: dict) -> str | None:
    val = _get_field(
        sample, "answer", "label", "gt_answer",
        "multiple_choice_answer", "target"
    )
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0]
    return str(val).strip()


def _get_image_path(sample: dict, jsonl_dir: Path) -> str:
    raw = _get_field(sample, "image", "image_path", "img", "file_name")
    if raw is None:
        raise KeyError(f"No image key in sample: {list(sample.keys())}")
    return resolve_image_path(str(raw), jsonl_dir)


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation and whitespace for comparison."""
    return text.strip().lower().rstrip(".,!?;:")


def _is_correct(prediction: str, ground_truth: str) -> bool:
    """
    Flexible exact-match: handles integers, floats, short phrases.
    Both directions substring check avoids penalising "the answer is 5" → "5".
    """
    pred = _normalise(prediction)
    gt = _normalise(ground_truth)

    if pred == gt:
        return True

    # Numeric comparison (e.g. "5" vs "5.0")
    try:
        if math.isclose(float(pred), float(gt), rel_tol=1e-6):
            return True
    except ValueError:
        pass

    # Substring check
    return gt in pred or pred in gt


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

def _save_results(
    agent_records: list[dict],
    baseline_records: list[dict],
    out_dir: Path,
) -> None:
    if agent_records:
        pred_file = out_dir / "predictions.jsonl"
        detail_file = out_dir / "agent_detailed.jsonl"
        # Spec requires: {"id": "...", "prediction": "..."} with prediction as string
        save_jsonl(
            [{"id": str(r["id"]), "prediction": str(r["prediction"])} for r in agent_records],
            str(pred_file),
        )
        save_jsonl(agent_records, str(detail_file))
        logger.debug("Saved %d agent records to %s", len(agent_records), out_dir)

    if baseline_records:
        pred_file = out_dir / "baseline_predictions.jsonl"
        detail_file = out_dir / "baseline_detailed.jsonl"
        save_jsonl(
            [{"id": str(r["id"]), "prediction": str(r["prediction"])} for r in baseline_records],
            str(pred_file),
        )
        save_jsonl(baseline_records, str(detail_file))
        logger.debug("Saved %d baseline records to %s", len(baseline_records), out_dir)


def _write_agent_trace(
    *,
    out_dir: Path,
    sample_id: str,
    image_path: str,
    question: str,
    prediction: str,
    runtime_s: float,
    stats: dict[str, Any],
) -> None:
    """Write per-sample tool trace for rubric evidence."""
    trace_dir = out_dir / "traces" / sample_id
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "id": sample_id,
        "question": question,
        "image": image_path,
        "runtime_s": round(runtime_s, 3),
        "final": {
            "prediction": prediction,
            "steps": stats.get("steps", 0),
            "tool_retry_count": stats.get("tool_retry_count", 0),
            "stage_trace": stats.get("stage_trace", []),
        },
        "steps": stats.get("tool_steps", []),
    }
    (trace_dir / "trace.json").write_text(json.dumps(trace, indent=2))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    data_path: str,
    output_dir: str,
    *,
    max_samples: int | None = None,
    sample_index: int | None = None,
    seed: int = 42,
    run_agent: bool = True,
    run_baseline: bool = True,
    agent_module: str = "agent",
) -> dict[str, Any]:
    """
    Evaluate agent (and optionally baseline) on a JSONL dataset.

    Returns the metrics dict (also written to <output_dir>/metrics.json).
    """
    _set_seed(seed)

    _agent_mod = importlib.import_module(agent_module)
    solve = _agent_mod.solve
    get_last_run_stats = _agent_mod.get_last_run_stats

    data_path_obj = Path(data_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir = data_path_obj.parent

    logger.info("Loading dataset: %s", data_path_obj)
    samples = load_jsonl(str(data_path_obj))

    if sample_index is not None:
        if sample_index < 0 or sample_index >= len(samples):
            raise ValueError(f"Sample index {sample_index} out of range [0, {len(samples)-1}]")
        samples = [samples[sample_index]]
        logger.info("Running single sample at index %d", sample_index)
    elif max_samples is not None:
        samples = samples[:max_samples]

    n = len(samples)
    logger.info("Evaluating %d samples (seed=%d)", n, seed)

    agent_records: list[dict] = []
    baseline_records: list[dict] = []

    # Create output directory structure upfront
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)

    # Counters
    gt_count = 0
    agent_correct = agent_fail = 0
    base_correct = base_fail = 0
    agent_times: list[float] = []
    base_times: list[float] = []
    agent_steps: list[int] = []
    agent_retry_counts: list[int] = []

    try:
        for idx, sample in enumerate(samples):
            sid = _get_id(sample, idx)

            try:
                image_path = _get_image_path(sample, jsonl_dir)
                question = _get_question(sample)
            except KeyError as exc:
                logger.error("Sample %s skipped: %s", sid, exc)
                continue

            ground_truth = _get_answer(sample)
            has_gt = ground_truth is not None
            if has_gt:
                gt_count += 1

            logger.info("[%d/%d] id=%s  q=%s", idx + 1, n, sid, question[:70])

            # ---- Baseline (run first so agent can use it as a safety net) ----
            base_pred = None
            if run_baseline:
                with Timer() as t:
                    try:
                        base_pred = baseline_solve(image_path, question)
                    except Exception:  # noqa: BLE001
                        logger.error("Baseline failed on %s:\n%s", sid, traceback.format_exc())
                        base_pred = "ERROR: baseline exception"
                        base_fail += 1

                base_times.append(t.elapsed)

                if has_gt and not base_pred.startswith("ERROR"):
                    if _is_correct(base_pred, ground_truth):
                        base_correct += 1

                baseline_records.append(
                    {
                        "id": sid,
                        "prediction": base_pred,
                        "ground_truth": ground_truth,
                        "correct": _is_correct(base_pred, ground_truth) if has_gt else None,
                        "runtime_s": round(t.elapsed, 3),
                    }
                )
                logger.info("  Baseline (%.1fs): %s", t.elapsed, base_pred[:80])

            # ---- Agent -------------------------------------------------------
            if run_agent:
                _baseline_for_fallback = (
                    base_pred
                    if (base_pred and not base_pred.startswith("ERROR"))
                    else None
                )
                with Timer() as t:
                    try:
                        _solve_kwargs = {}
                        import inspect as _inspect
                        if "baseline_fallback" in _inspect.signature(solve).parameters:
                            _solve_kwargs["baseline_fallback"] = _baseline_for_fallback
                        prediction = solve(image_path, question, **_solve_kwargs)
                        stats = get_last_run_stats()
                        steps_taken = stats.get("steps", 1)
                        retry_count = int(stats.get("tool_retry_count", 0))
                    except Exception:  # noqa: BLE001
                        logger.error("Agent failed on %s:\n%s", sid, traceback.format_exc())
                        prediction = "ERROR: agent exception"
                        steps_taken = 0
                        retry_count = 0
                        stats = {}
                        agent_fail += 1

                agent_times.append(t.elapsed)
                agent_steps.append(steps_taken)
                agent_retry_counts.append(float(retry_count))

                if has_gt and not prediction.startswith("ERROR"):
                    if _is_correct(prediction, ground_truth):
                        agent_correct += 1

                agent_records.append(
                    {
                        "id": sid,
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "correct": _is_correct(prediction, ground_truth) if has_gt else None,
                        "steps": steps_taken,
                        "tool_retry_count": retry_count,
                        "tool_steps": stats.get("tool_steps", []),
                        "runtime_s": round(t.elapsed, 3),
                        "used_baseline_fallback": stats.get("used_baseline_fallback", False),
                    }
                )
                _write_agent_trace(
                    out_dir=out_dir,
                    sample_id=sid,
                    image_path=image_path,
                    question=question,
                    prediction=prediction,
                    runtime_s=t.elapsed,
                    stats=stats,
                )
                logger.info("  Agent   (%d steps, %.1fs): %s", steps_taken, t.elapsed, prediction[:80])

            # ---- Baseline (already run above if run_baseline=True) ----------
            if run_baseline and base_pred is None:
                # run_baseline=True but agent was skipped — still need baseline
                with Timer() as t:
                    try:
                        base_pred = baseline_solve(image_path, question)
                    except Exception:  # noqa: BLE001
                        logger.error("Baseline failed on %s:\n%s", sid, traceback.format_exc())
                        base_pred = "ERROR: baseline exception"
                        base_fail += 1

                base_times.append(t.elapsed)

                if has_gt and not base_pred.startswith("ERROR"):
                    if _is_correct(base_pred, ground_truth):
                        base_correct += 1

                baseline_records.append(
                    {
                        "id": sid,
                        "prediction": base_pred,
                        "ground_truth": ground_truth,
                        "correct": _is_correct(base_pred, ground_truth) if has_gt else None,
                        "runtime_s": round(t.elapsed, 3),
                    }
                )
                logger.info("  Baseline (%.1fs): %s", t.elapsed, base_pred[:80])

            # Save checkpoint after every sample
            _save_results(agent_records, baseline_records, out_dir)

    finally:
        # Always save results, even if interrupted or crashed
        logger.info("Saving results...")
        _save_results(agent_records, baseline_records, out_dir)

    # ---- Metrics ---------------------------------------------------------
    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    metrics: dict[str, Any] = {"total_samples": n, "samples_with_gt": gt_count}

    if run_agent:
        baseline_fallback_count = sum(
            1 for rec in agent_records if rec.get("used_baseline_fallback", False)
        )
        metrics["agent"] = {
            "accuracy": round(agent_correct / gt_count, 4) if gt_count else None,
            "correct": agent_correct,
            "failures": agent_fail,
            "failure_rate": round(agent_fail / n, 4) if n else 0.0,
            "avg_runtime_s": _avg(agent_times),
            "avg_steps": _avg([float(s) for s in agent_steps]),
            "avg_tool_retries": _avg(agent_retry_counts),
            "baseline_fallback_count": baseline_fallback_count,
            "baseline_fallback_rate": round(baseline_fallback_count / len(agent_records), 4) if agent_records else 0.0,
        }

    if run_baseline:
        metrics["baseline"] = {
            "accuracy": round(base_correct / gt_count, 4) if gt_count else None,
            "correct": base_correct,
            "failures": base_fail,
            "failure_rate": round(base_fail / n, 4) if n else 0.0,
            "avg_runtime_s": _avg(base_times),
        }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # ---- Summary ---------------------------------------------------------
    sep = "=" * 55
    logger.info("\n%s", sep)
    logger.info("EVALUATION RESULTS  (%s)", data_path_obj.name)
    logger.info(sep)
    logger.info("Samples : %d  (with ground truth: %d)", n, gt_count)
    if run_agent:
        m = metrics["agent"]
        logger.info(
            "Agent    | acc=%-6s  fail=%d  avg_steps=%.1f  avg_time=%.2fs  baseline_fallback=%d",
            m["accuracy"], m["failures"], m["avg_steps"], m["avg_runtime_s"], m["baseline_fallback_count"],
        )
    if run_baseline:
        m = metrics["baseline"]
        logger.info(
            "Baseline | acc=%-6s  fail=%d  avg_time=%.2fs",
            m["accuracy"], m["failures"], m["avg_runtime_s"],
        )
    logger.info("Outputs  : %s", out_dir.resolve())
    logger.info(sep)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the vision agent and baseline on a JSONL dataset."
    )
    p.add_argument("--data", required=True, help="Path to JSONL dataset (e.g. dev.jsonl)")
    p.add_argument("--out", default="outputs/eval", help="Output directory")
    p.add_argument("--max-samples", type=int, default=None, metavar="N",
                   help="Limit evaluation to the first N samples")
    p.add_argument("--sample-index", type=int, default=None, metavar="INDEX",
                   help="Run only a specific sample by index (0-based)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--agent-only", action="store_true",
                   help="Skip the baseline, only run the agent")
    p.add_argument("--baseline-only", action="store_true",
                   help="Skip the agent, only run the baseline")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging")
    p.add_argument("--agent", default="agent", choices=["agent"],
                   help="Agent module to use (default: agent)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for name in ("agent", "baseline", "model_loader", "sandbox", "utils", "__main__"):
            setup_logger(name, logging.DEBUG)

    evaluate(
        data_path=args.data,
        output_dir=args.out,
        max_samples=args.max_samples,
        sample_index=args.sample_index,
        seed=args.seed,
        run_agent=not args.baseline_only,
        run_baseline=not args.agent_only,
        agent_module=args.agent,
    )


if __name__ == "__main__":
    main()
