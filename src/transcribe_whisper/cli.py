from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
ENGLISH_LANGUAGE_CODES = {"en", "english"}

app = typer.Typer(
    help="Transcribe audio/video files with MLX Whisper on Apple Silicon.",
    no_args_is_help=True,
)


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


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    return language.strip().lower().replace("_", "-")


def resolve_task(source_language: str | None, target_language: str | None) -> str:
    if not target_language or target_language == source_language:
        return "transcribe"

    if target_language in ENGLISH_LANGUAGE_CODES:
        return "translate"

    raise typer.BadParameter(
        "Whisper only supports direct translation to English. "
        f"Received target language: {target_language}"
    )


def require_input(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if not path.exists():
        raise typer.BadParameter(f"Input file not found: {path}")
    if not path.is_file():
        raise typer.BadParameter(f"Input path is not a file: {path}")
    return path


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise typer.BadParameter(
            "ffmpeg is required to accept mixed audio/video formats. "
            "Install it with: brew install ffmpeg"
        )


def normalize_media(
    input_path: Path,
    temp_dir: Path,
    *,
    start_time: float,
    duration: float | None,
) -> Path:
    require_ffmpeg()
    output_path = temp_dir / "input.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if start_time > 0:
        command.extend(["-ss", str(start_time)])
    command.extend([
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ])
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend([
        "-f",
        "wav",
        str(output_path),
    ])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or str(exc)).strip()
        raise typer.BadParameter(f"ffmpeg could not read this media file: {detail}") from exc
    return output_path


def transcribe_with_mlx(
    input_path: Path,
    *,
    model: str,
    source_language: str | None,
    target_language: str | None,
    convert: bool,
    temperature: float,
    condition_on_previous_text: bool,
    hallucination_silence_threshold: float | None,
    start_time: float,
    duration: float | None,
) -> dict:
    task = resolve_task(source_language, target_language)

    try:
        import mlx_whisper

        if not convert and start_time == 0 and duration is None:
            return mlx_whisper.transcribe(
                str(input_path),
                path_or_hf_repo=model,
                language=source_language,
                task=task,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                hallucination_silence_threshold=hallucination_silence_threshold,
                word_timestamps=False,
            )

        with tempfile.TemporaryDirectory(prefix="transcribe-whisper-") as temp_dir_name:
            normalized_path = normalize_media(
                input_path,
                Path(temp_dir_name),
                start_time=start_time,
                duration=duration,
            )
            return mlx_whisper.transcribe(
                str(normalized_path),
                path_or_hf_repo=model,
                language=source_language,
                task=task,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                hallucination_silence_threshold=hallucination_silence_threshold,
                word_timestamps=False,
            )
    except RuntimeError as exc:
        message = str(exc)
        if "No Metal device available" in message:
            raise typer.BadParameter(
                "MLX could not access a Metal GPU. Run this from a normal macOS "
                "Terminal session on Apple Silicon, not a headless/sandboxed shell."
            ) from exc
        raise


def write_text_output(result: dict, output_path: Path) -> None:
    text = (result.get("text") or "").strip()
    output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_srt_output(result: dict, output_path: Path) -> None:
    segments = result.get("segments") or []
    if not segments:
        raise typer.BadParameter("No transcription segments were returned by Whisper.")
    output_path.write_text(build_srt(segments), encoding="utf-8")


@app.command("text")
def text(
    input_path: Annotated[Path, typer.Argument(help="Audio or video file to transcribe.")],
    output_path: Annotated[
        Path | None,
        typer.Argument(help="Optional .txt output path."),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option("--source-language", "-l", help="Source language code, for example: pt, en, es."),
    ] = "pt",
    target_language: Annotated[
        str | None,
        typer.Option(
            "--target-language",
            "-t",
            help="Target language code. Whisper can translate directly only to English.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="MLX Whisper model repo.")] = DEFAULT_MODEL,
    convert: Annotated[
        bool,
        typer.Option(
            "--convert/--no-convert",
            help="Normalize input through ffmpeg before MLX Whisper.",
        ),
    ] = True,
    print_result: Annotated[
        bool,
        typer.Option("--print/--no-print", help="Print the transcription to stdout."),
    ] = True,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Whisper decoding temperature."),
    ] = 0.0,
    condition_on_previous_text: Annotated[
        bool,
        typer.Option(
            "--condition-on-previous-text/--no-condition-on-previous-text",
            help="Use prior generated text as context for the next audio window.",
        ),
    ] = False,
    hallucination_silence_threshold: Annotated[
        float | None,
        typer.Option(
            "--hallucination-silence-threshold",
            help="Suppress likely hallucinations around silent/non-speech spans.",
        ),
    ] = 2.0,
    start_time: Annotated[
        float,
        typer.Option("--start-time", help="Seconds to skip before transcription."),
    ] = 0.0,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Maximum seconds to transcribe."),
    ] = None,
) -> None:
    """Write a plain text transcription."""
    input_file = require_input(input_path)
    source = normalize_language(source_language)
    target = normalize_language(target_language)
    output_file = output_path.expanduser().resolve() if output_path else input_file.with_suffix(".txt")

    result = transcribe_with_mlx(
        input_file,
        model=model,
        source_language=source,
        target_language=target,
        convert=convert,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        hallucination_silence_threshold=hallucination_silence_threshold,
        start_time=start_time,
        duration=duration,
    )
    write_text_output(result, output_file)

    if print_result:
        typer.echo((result.get("text") or "").strip())
    typer.echo(f"Text saved to: {output_file}", err=True)


@app.command("srt")
def srt(
    input_path: Annotated[Path, typer.Argument(help="Audio or video file to transcribe.")],
    output_path: Annotated[
        Path | None,
        typer.Argument(help="Optional .srt output path."),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option("--source-language", "-l", help="Source language code, for example: pt, en, es."),
    ] = "pt",
    target_language: Annotated[
        str | None,
        typer.Option(
            "--target-language",
            "-t",
            help="Target language code. Whisper can translate directly only to English.",
        ),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="MLX Whisper model repo.")] = DEFAULT_MODEL,
    convert: Annotated[
        bool,
        typer.Option(
            "--convert/--no-convert",
            help="Normalize input through ffmpeg before MLX Whisper.",
        ),
    ] = True,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Whisper decoding temperature."),
    ] = 0.0,
    condition_on_previous_text: Annotated[
        bool,
        typer.Option(
            "--condition-on-previous-text/--no-condition-on-previous-text",
            help="Use prior generated text as context for the next audio window.",
        ),
    ] = False,
    hallucination_silence_threshold: Annotated[
        float | None,
        typer.Option(
            "--hallucination-silence-threshold",
            help="Suppress likely hallucinations around silent/non-speech spans.",
        ),
    ] = 2.0,
    start_time: Annotated[
        float,
        typer.Option("--start-time", help="Seconds to skip before transcription."),
    ] = 0.0,
    duration: Annotated[
        float | None,
        typer.Option("--duration", help="Maximum seconds to transcribe."),
    ] = None,
) -> None:
    """Write an SRT subtitle file."""
    input_file = require_input(input_path)
    source = normalize_language(source_language)
    target = normalize_language(target_language)
    output_file = output_path.expanduser().resolve() if output_path else input_file.with_suffix(".srt")

    result = transcribe_with_mlx(
        input_file,
        model=model,
        source_language=source,
        target_language=target,
        convert=convert,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        hallucination_silence_threshold=hallucination_silence_threshold,
        start_time=start_time,
        duration=duration,
    )
    write_srt_output(result, output_file)
    typer.echo(f"SRT saved to: {output_file}")


@app.callback()
def default() -> None:
    """Transcribe audio/video files with MLX Whisper."""


if __name__ == "__main__":
    app()
