### Transcrição de Áudio para Português utilizando ML em MacOS

1. Instalar Blackhole na máquina

```sh
brew install --cask blackhole-2ch
```

2. Configurar MacOS em Ajustes > Configurações de Áudio e MIDI

Cadastrar Dispositivo com Saída Múltipla e colocar a saída normal para ouvir e a saída BlackHole

3. Utilizar o QuickTime Player para gravar

4. Instalar conda

```sh
brew install --cask miniforge
```

5. Criar ambiente python para a aplicação

```sh
conda create -n transcription python=3.12 -y
conda activate transcription
```

6. Instalar dependências

```sh
brew install ffmpeg
pip install -r requirements.txt
```

7. Executar script

A primeira vez demorará um pouco mais pois ele baixará os modelos necessários para a atividade

```sh
python whisper.py /Users/filipelopes/Desktop/curcubita_pepo.m4a
```

8. Gerar `.srt` a partir de vídeo ou áudio com Whisper

O script abaixo aceita arquivos como `.mp4`, `.mov`, `.mkv`, `.m4a`, `.wav` e gera um arquivo `.srt`.

```sh
python whisper_srt.py /caminho/video.mp4
```

Se quiser definir o caminho de saída:

```sh
python whisper_srt.py /caminho/video.mp4 /caminho/saida.srt
```

Se quiser informar o idioma de origem:

```sh
python whisper_srt.py /caminho/video.mp4 --source-language pt
```

Se quiser traduzir mantendo os mesmos tempos, use o idioma de destino. Com Whisper, a tradução nativa é suportada apenas para inglês:

```sh
python whisper_srt.py /caminho/video.mp4 /caminho/video.en.srt --source-language pt --target-language en
```
