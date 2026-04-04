import logging


LOGGER = logging.getLogger(__name__)


def disable_broken_torchcodec() -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception as error:
        error_message = str(error).splitlines()[0]
        LOGGER.warning(
            "TorchCodec is installed but unusable (%s). Disabling Transformers torchcodec integration for this process.",
            error_message,
        )

        from transformers.pipelines import automatic_speech_recognition
        from transformers.utils import import_utils

        def _torchcodec_unavailable() -> bool:
            return False

        import_utils.is_torchcodec_available = _torchcodec_unavailable
        automatic_speech_recognition.is_torchcodec_available = _torchcodec_unavailable