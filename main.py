import typer
import torchaudio
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
import subprocess

app = typer.Typer()

# Carregar modelos e processadores globalmente
PROCESSOR = Wav2Vec2Processor.from_pretrained("lgris/wav2vec2-large-xlsr-open-brazilian-portuguese")
MODEL = Wav2Vec2ForCTC.from_pretrained("lgris/wav2vec2-large-xlsr-open-brazilian-portuguese")

# Configurações do PyTorch para CPUs
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to wav format using ffmpeg."""
    if not os.path.exists(output_path):
        # Adiciona filtros de pré-processamento no ffmpeg para melhorar a qualidade do áudio
        command = [
            "ffmpeg", "-i", input_path,
            "-af", "highpass=f=200, lowpass=f=3000",
            output_path, "-y"
        ]
        subprocess.run(command, capture_output=True)
    else:
        typer.echo(f"Output file {output_path} already exists, skipping conversion.")


@app.command("transcribe")
def transcribe(audio_path: str):
    """
    Transcribe an audio file to text using Wav2Vec2 model.
    
    Args:
        audio_path: Path to the .m4a or .wav audio file.
    """
    # Check if input file is m4a and convert to wav if necessary
    if audio_path.endswith(".m4a"):
        wav_path = audio_path.replace(".m4a", ".wav")
        convert_to_wav(audio_path, wav_path)
    else:
        wav_path = audio_path

    # Load the wav file with soundfile
    waveform, sample_rate = sf.read(wav_path, dtype="float32")

    # Convert waveform to PyTorch tensor and adjust to mono if stereo
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)  # Convert stereo to mono by averaging channels

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Transcribe audio in smaller chunks to handle long audios efficiently
    chunk_size = 30 * 16000  # 30 seconds at 16 kHz
    transcription = []
    for start in range(0, waveform.size(0), chunk_size):
        chunk = waveform[start:start + chunk_size]
        inputs = PROCESSOR(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = MODEL(inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription.append(PROCESSOR.decode(predicted_ids[0]))

    # Output transcription to terminal
    typer.echo("\nTranscrição:\n" + " ".join(transcription))


if __name__ == "__main__":
    app()