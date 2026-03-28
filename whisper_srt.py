from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ENGLISH_LANGUAGE_CODES = {"en", "english"}


def format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def build_srt(segments: list[dict]) -> str:
    entries: list[str] = []

    for index, segment in enumerate(segments, start=1):
        text = (segment.get("text") or "").strip()
        if not text:
            continue

        start = format_srt_timestamp(float(segment["start"]))
        end = format_srt_timestamp(float(segment["end"]))
        entries.append(f"{index}\n{start} --> {end}\n{text}")

    return "\n\n".join(entries) + ("\n" if entries else "")


def extract_audio(input_path: Path) -> Path:
    audio_path = input_path.with_suffix(".whisper.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SRT from a video or audio file using Whisper."
    )
    parser.add_argument("input_path", help="Path to the input video or audio file.")
    parser.add_argument(
        "output_srt_path",
        nargs="?",
        help="Optional path for the generated .srt file.",
    )
    parser.add_argument(
        "--source-language",
        default=None,
        help="Source language code, for example: pt, en, es.",
    )
    parser.add_argument(
        "--target-language",
        default=None,
        help=(
            "Target language code. Use the same language as the source to only "
            "transcribe. Whisper translation is only supported to English."
        ),
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate the detected source language to English while keeping subtitle timings.",
    )
    return parser.parse_args()


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    return language.strip().lower().replace("_", "-")


def resolve_task(source_language: str | None, target_language: str | None) -> str:
    if not target_language or target_language == source_language:
        return "transcribe"

    if target_language in ENGLISH_LANGUAGE_CODES:
        return "translate"

    raise ValueError(
        "Whisper only supports direct translation to English. "
        f"Received target language: {target_language}"
    )


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = (
        Path(args.output_srt_path).expanduser().resolve()
        if args.output_srt_path
        else input_path.with_suffix(".srt")
    )
    source_language = normalize_language(args.source_language)
    target_language = normalize_language(args.target_language)

    transcription_input = input_path
    temp_audio_path: Path | None = None

    try:
        if args.translate:
            if target_language and target_language not in ENGLISH_LANGUAGE_CODES:
                print("--translate only supports English output in Whisper.")
                return 1
            target_language = "en"

        task = resolve_task(source_language, target_language)

        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            temp_audio_path = extract_audio(input_path)
            transcription_input = temp_audio_path

        import mlx_whisper

        result = mlx_whisper.transcribe(
            str(transcription_input),
            path_or_hf_repo=DEFAULT_MODEL,
            language=source_language,
            task=task,
            word_timestamps=False,
        )

        segments = result.get("segments") or []
        if not segments:
            print("No transcription segments were returned by Whisper.")
            return 1

        output_path.write_text(build_srt(segments), encoding="utf-8")
        print(f"SRT saved to: {output_path}")
        return 0
    except ValueError as exc:
        print(str(exc))
        return 1
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc))
        return exc.returncode or 1
    finally:
        if temp_audio_path and temp_audio_path.exists():
            temp_audio_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
