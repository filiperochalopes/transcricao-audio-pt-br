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
python main.py /Users/filipelopes/Desktop/curcubita_pepo.m4a
```