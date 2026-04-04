import json
import argparse
import logging
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

from .utils.device import clear_device_cache, resolve_attention_implementation, resolve_device, resolve_dtype
from .utils.diarize import load_audio_inputs
from .utils.hf_compat import disable_broken_torchcodec
from .utils.result import build_result


LOGGER = logging.getLogger(__name__)


def _clear_generation_attribute(target, attribute_name: str) -> None:
    if target is not None and hasattr(target, attribute_name):
        setattr(target, attribute_name, None)


def _configure_whisper_generation(pipe, task: str, language: str | None, is_english_only_model: bool) -> None:
    if getattr(pipe.model.config, "model_type", None) != "whisper":
        return

    pipeline_generation_config = getattr(pipe, "generation_config", None)
    model_generation_config = getattr(pipe.model, "generation_config", None)

    for config in (pipeline_generation_config, model_generation_config, pipe.model.config):
        _clear_generation_attribute(config, "forced_decoder_ids")

    # Whisper injects suppress-token processors before delegating to GenerationMixin.generate().
    # Clearing the model defaults avoids GenerationMixin rebuilding the same processors a second time.
    for attribute_name in ("suppress_tokens", "begin_suppress_tokens"):
        _clear_generation_attribute(model_generation_config, attribute_name)

    if is_english_only_model:
        for config in (pipeline_generation_config, model_generation_config):
            _clear_generation_attribute(config, "task")
            _clear_generation_attribute(config, "language")
        return

    for config in (pipeline_generation_config, model_generation_config):
        if config is None:
            continue
        if hasattr(config, "task"):
            config.task = task
        if hasattr(config, "language"):
            config.language = language

parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device to run inference on. Use a CUDA device number like "0", "mps" for Macs with Apple Silicon, or "cpu" for CPU-only systems. If CUDA is unavailable, numeric device IDs fall back to CPU. (default: "0")',
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf-token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)
parser.add_argument(
    "--num-speakers",
    required=False,
    default=None,
    type=int,
    help="Specifies the exact number of speakers present in the audio file. Useful when the exact number of participants in the conversation is known. Must be at least 1. Cannot be used together with --min-speakers or --max-speakers. (default: None)",
)
parser.add_argument(
    "--min-speakers",
    required=False,
    default=None,
    type=int,
    help="Sets the minimum number of speakers that the system should consider during diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be less than or equal to --max-speakers if both are specified. (default: None)",
)
parser.add_argument(
    "--max-speakers",
    required=False,
    default=None,
    type=int,
    help="Defines the maximum number of speakers that the system should consider in diarization. Must be at least 1. Cannot be used together with --num-speakers. Must be greater than or equal to --min-speakers if both are specified. (default: None)",
)

def main():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = parser.parse_args()

    if args.num_speakers is not None and (args.min_speakers is not None or args.max_speakers is not None):
        parser.error("--num-speakers cannot be used together with --min-speakers or --max-speakers.")

    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")

    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")

    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")

    if args.min_speakers is not None and args.max_speakers is not None and args.min_speakers > args.max_speakers:
        if args.min_speakers > args.max_speakers:
            parser.error("--min-speakers cannot be greater than --max-speakers.")

    resolved_device = resolve_device(args.device_id)
    resolved_dtype = resolve_dtype(resolved_device)
    attention_implementation = resolve_attention_implementation(resolved_device, args.flash)
    disable_broken_torchcodec()
    LOGGER.info(
        "Starting transcription with device=%s dtype=%s attention=%s model=%s",
        resolved_device,
        resolved_dtype,
        attention_implementation,
        args.model_name,
    )

    language = None if args.language == "None" else args.language
    is_english_only_model = args.model_name.rsplit("/", 1)[-1].endswith(".en")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        dtype=resolved_dtype,
        device=resolved_device,
        model_kwargs={"attn_implementation": attention_implementation},
    )
    _configure_whisper_generation(pipe, args.task, language, is_english_only_model)

    clear_device_cache(resolved_device)
    # elif not args.flash:
        # pipe.model = pipe.model.to_bettertransformer()

    ts = "word" if args.timestamp == "word" else True

    generate_kwargs = {}
    if not is_english_only_model:
        generate_kwargs["task"] = args.task
        if language is not None:
            generate_kwargs["language"] = language
    elif language is not None or args.task != "transcribe":
        LOGGER.info(
            "Ignoring language/task overrides for English-only model %s",
            args.model_name,
        )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)

        transcription_inputs = load_audio_inputs(inputs=args.file_name)

        outputs = pipe(
            transcription_inputs,
            chunk_length_s=30,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
            return_timestamps=ts,
        )

    if args.hf_token != "no_token":
        from .utils.diarization_pipeline import diarize

        speakers_transcript = diarize(args, outputs, device=resolved_device)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Your file has been transcribed and speaker segmented. Output: {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Your file has been transcribed. Output: {args.transcript_path}"
        )
