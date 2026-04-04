import torch
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import logging

from .diarize import post_process_segments_and_transcripts, diarize_audio, \
    preprocess_inputs
from .device import resolve_device


LOGGER = logging.getLogger(__name__)


def diarize(args, outputs, device=None):
    resolved_device = device or resolve_device(args.device_id)
    diarization_pipeline = Pipeline.from_pretrained(
        checkpoint_path=args.diarization_model,
        use_auth_token=args.hf_token,
    )
    LOGGER.info("Starting diarization with device=%s model=%s", resolved_device, args.diarization_model)
    diarization_pipeline.to(torch.device(resolved_device))

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Segmenting...", total=None)

        inputs, diarizer_inputs = preprocess_inputs(inputs=args.file_name)

        segments = diarize_audio(diarizer_inputs, diarization_pipeline, args.num_speakers, args.min_speakers, args.max_speakers)

        return post_process_segments_and_transcripts(
            segments, outputs["chunks"], group_by_speaker=False
        )
