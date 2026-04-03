"""
baseline.py — Single-step, zero-tool VQA baseline.

Public API
----------
baseline_solve(image_path, question) -> str
    One model call; no code execution, no loop.
"""

from __future__ import annotations

from model_loader import generate_response
from utils import setup_logger

logger = setup_logger(__name__)

_BASELINE_PROMPT = (
    "Answer the question about the image as concisely as possible. "
    "Give only the answer value — no explanation, no sentence.\n"
    "Question: {question}\n"
    "Answer:"
)


def baseline_solve(image_path: str, question: str) -> str:
    """
    Answer a visual question with a single forward pass — no tools.

    Parameters
    ----------
    image_path:
        Path to the image file.
    question:
        Natural-language question.

    Returns
    -------
    str
        The model's direct answer.
    """
    logger.debug("Baseline: %s", question[:80])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_path": image_path},
                {
                    "type": "text",
                    "text": _BASELINE_PROMPT.format(question=question),
                },
            ],
        }
    ]

    try:
        response = generate_response(messages).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("Baseline model error: %s", exc)
        return "ERROR: model inference failed"

    # Strip any residual "Answer:" prefix the model may echo
    if response.lower().startswith("answer:"):
        response = response[7:].strip()

    return response
