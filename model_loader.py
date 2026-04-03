"""
model_loader.py — Vision-language model loading and inference.

Supports:
  • CUDA   → 4-bit quantisation via bitsandbytes (falls back to bfloat16)
  • MPS    → float16, moved to MPS device
  • CPU    → float32

Default model : HuggingFaceTB/SmolVLM-2.2B-Instruct  (~2.5 GB VRAM, 4-bit)
Override via env-vars:
  VQA_MODEL        — HuggingFace hub id
  VQA_MODEL_PATH   — local directory path (takes priority if it exists)
  VQA_MAX_TOKENS   — max new tokens to generate (default 1024)

Public API
----------
generate_response(messages: list[dict]) -> str
    messages format (OpenAI-like):
        [
          {
            "role": "user",
            "content": [
              {"type": "image", "image_path": "/path/to/img.jpg"},
              {"type": "text",  "text": "What is in this image?"}
            ]
          }
        ]
    Text-only content may also be a plain string:
        {"role": "user", "content": "Hello"}
"""

from __future__ import annotations

import os
from typing import Any

import torch
from PIL import Image

from utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME: str = os.environ.get("VQA_MODEL", _DEFAULT_MODEL)
LOCAL_MODEL_PATH: str | None = os.environ.get("VQA_MODEL_PATH")
MAX_NEW_TOKENS: int = int(os.environ.get("VQA_MAX_TOKENS", "1024"))

# Module-level singletons — lazily initialised by get_model()
_model: Any = None
_processor: Any = None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_model_id() -> str:
    if LOCAL_MODEL_PATH and os.path.isdir(LOCAL_MODEL_PATH):
        logger.info("Using local model path: %s", LOCAL_MODEL_PATH)
        return LOCAL_MODEL_PATH
    return MODEL_NAME


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model() -> None:
    """Initialise _model and _processor (called once)."""
    global _model, _processor

    try:  # noqa: PLC0415
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration as AutoModelForVision2Seq
    except ImportError as exc:
        raise ImportError(
            "transformers>=4.45.0 is required. "
            "Run:  pip install --upgrade 'transformers>=4.45.0'  then restart the kernel."
        ) from exc

    model_id = _resolve_model_id()
    device = _get_device()
    logger.info("Loading model '%s' on device '%s'", model_id, device)

    _processor = AutoProcessor.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))

    if device == "cuda":
        _model = _load_cuda(model_id, AutoModelForVision2Seq)
    elif device == "mps":
        _model = _load_mps(model_id, AutoModelForVision2Seq)
    else:
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN"),
        )
        logger.info("Loaded fp32 on CPU")

    _model.eval()
    logger.info("Model ready")


def _load_cuda(model_id: str, ModelClass: type) -> Any:
    """Load with 4-bit quantisation by default; fall back to bfloat16 if bnb fails."""
    try:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = ModelClass.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=os.environ.get("HF_TOKEN"),
        )
        logger.info("Loaded with 4-bit quantisation (CUDA)")
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning("4-bit quantisation failed (%s); falling back to bfloat16", exc)

    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("Loaded bfloat16 on CUDA")
    return model


def _load_mps(model_id: str, ModelClass: type) -> Any:
    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = model.to("mps")
    logger.info("Loaded float16 on MPS")
    return model


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_model() -> tuple[Any, Any]:
    """Return (model, processor), loading them on the first call."""
    global _model, _processor
    if _model is None or _processor is None:
        _load_model()
    return _model, _processor


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _build_smolvlm_input(
    messages: list[dict],
) -> tuple[list[dict], list[Image.Image]]:
    """
    Convert an OpenAI-style messages list into SmolVLM chat-template format.

    SmolVLM expects:
      - messages where image content items are ``{"type": "image"}`` (no path),
        with the actual PIL images passed separately to the processor.
      - Text-only assistant turns are preserved for multi-turn context.

    Returns
    -------
    chat_messages : list[dict]
        Messages with ``{"type": "image"}`` placeholders (no file paths).
    images : list[PIL.Image]
        Ordered list of PIL images extracted from the messages.
    """
    chat_messages: list[dict] = []
    images: list[Image.Image] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            chat_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        # content is a list of typed items
        new_content: list[dict] = []
        for item in content:
            if item["type"] == "image":
                img_path = (
                    item.get("image_path")
                    or item.get("path")
                    or item.get("url", "")
                )
                if img_path and os.path.exists(str(img_path)):
                    images.append(Image.open(img_path).convert("RGB"))
                    new_content.append({"type": "image"})
                # silently skip missing images
            elif item["type"] == "text":
                new_content.append({"type": "text", "text": item["text"]})

        if new_content:
            chat_messages.append({"role": role, "content": new_content})

    return chat_messages, images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(messages: list[dict]) -> str:
    """
    Run a single forward pass through the vision-language model.

    Parameters
    ----------
    messages:
        Chat-style message list.  Each dict has ``role`` and ``content``.
        Content can be a plain string or a list of typed items::

            [
              {"type": "image", "image_path": "path/to/img.jpg"},
              {"type": "text",  "text": "Your prompt here"}
            ]

    Returns
    -------
    str
        The model's generated text (newly generated tokens only, stripped).
    """
    model, processor = get_model()

    chat_messages, images = _build_smolvlm_input(messages)

    # Apply the model's chat template to produce the formatted prompt string
    prompt = processor.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
    )

    if images:
        inputs = processor(text=prompt, images=images, return_tensors="pt")
    else:
        inputs = processor(text=prompt, return_tensors="pt")

    # Move tensors to the model's device
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("VQA_MAX_TOKENS", str(MAX_NEW_TOKENS))),
            do_sample=False,  # deterministic greedy decoding
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    return generated.strip()
