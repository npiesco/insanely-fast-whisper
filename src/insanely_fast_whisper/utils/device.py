import logging

import torch
from transformers.utils import is_flash_attn_2_available


LOGGER = logging.getLogger(__name__)


def resolve_device(device_id: str) -> str:
    normalized_device = str(device_id).strip().lower()

    if normalized_device == "cpu":
        return "cpu"

    if normalized_device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested with --device-id mps, but MPS is not available on this machine.")
        return "mps"

    if normalized_device.startswith("cuda:"):
        normalized_device = normalized_device.split(":", 1)[1]

    if normalized_device.isdigit():
        if torch.cuda.is_available():
            return f"cuda:{normalized_device}"

        LOGGER.warning(
            "CUDA device '%s' was requested but CUDA is unavailable; falling back to CPU.",
            device_id,
        )
        return "cpu"

    raise ValueError(
        f"Unsupported device id '{device_id}'. Use 'cpu', 'mps', a CUDA index like '0', or a CUDA device like 'cuda:0'."
    )


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32

    return torch.float16


def resolve_attention_implementation(device: str, use_flash: bool) -> str:
    if not use_flash:
        return "sdpa"

    if not device.startswith("cuda"):
        LOGGER.warning(
            "Flash Attention 2 was requested on device '%s'; falling back to sdpa because Flash Attention 2 requires CUDA.",
            device,
        )
        return "sdpa"

    if not is_flash_attn_2_available():
        raise RuntimeError(
            "Flash Attention 2 was requested, but it is not available in this environment. Install flash-attn correctly or rerun without --flash True."
        )

    return "flash_attention_2"


def clear_device_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()