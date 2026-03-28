import mlx_whisper
import sys
# Usage: python whisper.py <audio_file_path>
if len(sys.argv) != 2:
    print("Usage: python whisper.py <audio_file_path>")
    sys.exit(1)
audio_path = sys.argv[1]
result = mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
print(result["text"])