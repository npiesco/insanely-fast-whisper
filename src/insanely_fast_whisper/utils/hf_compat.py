import logging
import warnings


LOGGER = logging.getLogger(__name__)


def disable_broken_torchcodec() -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception as error:
        error_message = str(error).splitlines()[0]
        LOGGER.debug(
            "TorchCodec is installed but unusable (%s). Disabling Transformers torchcodec integration for this process.",
            error_message,
        )

        from transformers.pipelines import automatic_speech_recognition
        from transformers.utils import import_utils

        def _torchcodec_unavailable() -> bool:
            return False

        import_utils.is_torchcodec_available = _torchcodec_unavailable
        automatic_speech_recognition.is_torchcodec_available = _torchcodec_unavailable


def suppress_known_pyannote_warnings() -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message=r"\ntorchcodec is not installed correctly so built-in audio decoding will fail\.",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )

    warnings.filterwarnings(
        "ignore",
        message=r"std\(\): degrees of freedom is <= 0\..*",
        category=UserWarning,
        module=r"pyannote\.audio\.models\.blocks\.pooling",
    )