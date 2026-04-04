import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path


def _write_silence_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def test_cli_transcribes_with_explicit_cpu_device(tmp_path: Path) -> None:
    audio_path = tmp_path / "silence.wav"
    output_path = tmp_path / "transcript.json"
    _write_silence_wav(audio_path)
    cli_executable = shutil.which("insanely-fast-whisper")

    assert cli_executable is not None, "The installed CLI entrypoint must be available during integration tests."

    result = subprocess.run(
        [
            cli_executable,
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
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=900,
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