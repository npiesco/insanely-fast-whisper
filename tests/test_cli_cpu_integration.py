import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path

import pytest


HF_TOKEN_ENV = "INSANELY_FAST_WHISPER_HF_TOKEN"
DIARIZATION_AUDIO_URL_ENV = "INSANELY_FAST_WHISPER_DIARIZATION_AUDIO_URL"
DIARIZATION_MODEL_ENV = "INSANELY_FAST_WHISPER_DIARIZATION_MODEL"
DEFAULT_DIARIZATION_AUDIO_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
GATED_MODEL_ERROR_MARKERS = (
    "GatedRepoError",
    "accept user conditions",
    "restricted and you are not in the authorized list",
    "Could not download Pipeline from",
)


def _extract_gated_model_ids(output: str) -> list[str]:
    return sorted(set(re.findall(r"Access to model ([^ ]+) is restricted", output)))


def _write_silence_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _run_cli(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    cli_executable = shutil.which("insanely-fast-whisper")

    assert cli_executable is not None, "The installed CLI entrypoint must be available during integration tests."

    return subprocess.run(
        [cli_executable, *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_transcribes_with_explicit_cpu_device(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    output_path = tmp_path / "transcript.json"
    _write_silence_wav(audio_path)
    result = _run_cli(
        [
            "--file-name",
            str(audio_path),
            "--transcript-path",
            str(output_path),
            "--device-id",
            "cpu",
            "--model-name",
            "openai/whisper-tiny.en",
            "--batch-size",
            "1",
            "--language",
            "en",
        ]
    )

    assert result.returncode == 0, (
        "CLI should support explicit CPU execution on non-CUDA systems.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert output_path.exists(), "CLI should write the requested transcript file."
    assert "Your file has been transcribed" in result.stdout

    with output_path.open("r", encoding="utf-8") as transcript_file:
        transcript = json.load(transcript_file)

    assert isinstance(transcript["text"], str)


def test_cli_diarizes_with_explicit_cpu_device(tmp_path: Path) -> None:
    hf_token = os.environ.get(HF_TOKEN_ENV)
    if not hf_token:
        pytest.skip(f"Set {HF_TOKEN_ENV} to run the real CPU diarization integration test.")

    audio_source = os.environ.get(DIARIZATION_AUDIO_URL_ENV, DEFAULT_DIARIZATION_AUDIO_URL)
    diarization_model = os.environ.get(DIARIZATION_MODEL_ENV, DEFAULT_DIARIZATION_MODEL)
    output_path = tmp_path / "diarized_transcript.json"

    result = _run_cli(
        [
            "--file-name",
            audio_source,
            "--transcript-path",
            str(output_path),
            "--device-id",
            "cpu",
            "--model-name",
            "openai/whisper-tiny.en",
            "--batch-size",
            "1",
            "--language",
            "en",
            "--hf-token",
            hf_token,
            "--diarization_model",
            diarization_model,
        ],
        timeout=1800,
    )

    combined_output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 and any(marker in combined_output for marker in GATED_MODEL_ERROR_MARKERS):
        gated_model_ids = _extract_gated_model_ids(combined_output)
        if gated_model_ids:
            inaccessible_models = ", ".join(f"'{model_id}'" for model_id in gated_model_ids)
            pytest.skip(
                f"The configured token cannot access required gated model(s): {inaccessible_models}. "
                f"Accept the required model terms or set {DIARIZATION_MODEL_ENV} to an accessible diarization model."
            )
        pytest.skip(
            f"The configured token cannot access diarization model '{diarization_model}' or one of its gated dependencies "
            "(for example 'pyannote/segmentation-3.0'). Accept the required model terms or set "
            f"{DIARIZATION_MODEL_ENV} to an accessible diarization model."
        )

    assert result.returncode == 0, (
        "CLI should support real speaker diarization on explicit CPU execution when a Hugging Face token is available.\n"
        f"Audio source: {audio_source}\n"
        f"Diarization model: {diarization_model}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert output_path.exists(), "CLI should write the requested diarized transcript file."
    assert "speaker segmented" in result.stdout

    with output_path.open("r", encoding="utf-8") as transcript_file:
        transcript = json.load(transcript_file)

    assert isinstance(transcript["text"], str)
    assert isinstance(transcript["chunks"], list)
    assert isinstance(transcript["speakers"], list)
    assert transcript["speakers"], (
        "Expected at least one speaker-attributed segment from the real diarization pipeline. "
        f"Audio source: {audio_source}"
    )
    first_segment = transcript["speakers"][0]
    assert isinstance(first_segment["speaker"], str)
    assert isinstance(first_segment["text"], str)
    assert len(first_segment["timestamp"]) == 2