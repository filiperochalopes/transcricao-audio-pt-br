### Transcrição de Áudio/Vídeo com MLX Whisper no macOS

Este projeto usa `mlx-whisper`, não Faster Whisper. O backend foi pensado para Apple
Silicon/MLX e funciona bem em scripts de terminal e em AppleScript/Automator.

O comando aceita áudio e vídeo em formatos comuns (`.m4a`, `.mp3`, `.wav`, `.aac`,
`.flac`, `.ogg`, `.mp4`, `.mov`, `.mkv`, `.webm` etc.) porque normaliza a entrada
com `ffmpeg` antes de chamar o Whisper.

## Pré-requisitos

1. Instale `uv` e `ffmpeg`:

```sh
brew install uv ffmpeg
```

2. Se quiser gravar áudio interno do macOS, instale o BlackHole:

```sh
brew install --cask blackhole-2ch
```

Depois configure um dispositivo de saída múltipla no app "Configuração de Áudio e
MIDI" e grave com o QuickTime Player.

## Instalação

Para usar de qualquer pasta, instale como ferramenta global do `uv`:

```sh
cd /Users/filipelopes/Desktop/Development/transcribe-whisper
uv tool install --reinstall .
```

Isso cria o executável em `~/.local/bin/transcribe-whisper`. Se o shell não
encontrar o comando, rode:

```sh
uv tool update-shell
```

Para desenvolvimento local do projeto, `uv` também cria e mantém a `.venv`
automaticamente:

```sh
uv sync
```

## Uso

Na primeira execução o modelo será baixado. O idioma padrão é `pt` e a CLI usa
decoding mais conservador para reduzir repetição/hallucination em músicas.

Transcrever para `.txt`:

```sh
transcribe-whisper text /caminho/audio.m4a
```

Escolher o arquivo de saída:

```sh
transcribe-whisper text /caminho/audio.m4a /caminho/saida.txt
```

Gerar `.srt`:

```sh
transcribe-whisper srt /caminho/video.mp4
```

Informar idioma de origem:

```sh
transcribe-whisper text /caminho/audio.m4a --source-language pt
```

Traduzir para inglês, mantendo tempos no `.srt`:

```sh
transcribe-whisper srt /caminho/video.mp4 /caminho/video.en.srt --source-language pt --target-language en
```

Trocar modelo MLX:

```sh
transcribe-whisper text /caminho/audio.m4a --model mlx-community/whisper-small-mlx
```

Se o áudio for música e o modelo repetir palavras da introdução, mantenha o
padrão `--no-condition-on-previous-text`. Para tentar uma transcrição mais
contextual em fala contínua:

```sh
transcribe-whisper text /caminho/audio.m4a --condition-on-previous-text
```

Para pular uma introdução musical ou transcrever só um trecho:

```sh
transcribe-whisper text /caminho/audio.m4a --start-time 30
transcribe-whisper text /caminho/audio.m4a --start-time 60 --duration 30
```

Se estiver dentro da pasta deste projeto, também pode usar:

```sh
uv run transcribe-whisper text /caminho/audio.m4a
```

## Uso simples em app scripts do macOS

Use sempre o caminho absoluto do `uv` para não depender do `PATH` do Automator ou
AppleScript:

```sh
/Users/filipelopes/.local/bin/transcribe-whisper text "$1"
```

Para SRT:

```sh
/Users/filipelopes/.local/bin/transcribe-whisper srt "$1"
```

Os scripts antigos ainda funcionam como atalhos:

```sh
uv run python whisper.py /caminho/audio.m4a
uv run python whisper_srt.py /caminho/video.mp4
```
