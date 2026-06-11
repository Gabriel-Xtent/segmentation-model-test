# segmentation-model-test

Script pra testar modelos de segmentacao YOLO em imagem, video, pasta ou webcam.
Desenha as mascaras e os nomes das classes, com filtro opcional por class name.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

O jeito mais simples (sem digitar caminho nenhum):

1. coloque seu modelo `.pt` na pasta `model/`
2. coloque suas imagens/videos na pasta `inputs/`
3. rode:

```bash
python seg_test.py
```

Ele pega o modelo de `model/`, processa tudo de `inputs/` e salva em `outputs/`.

Voce tambem pode apontar o modelo e a fonte manualmente:

```bash
# imagem (mostra todas as classes) -- abre janela e salva
python seg_test.py --model best.pt --source foto.jpg

# video, mostrando so as classes "person" e "car"
python seg_test.py --model best.pt --source video.mp4 --classes person car

# webcam (indice 0)
python seg_test.py --model best.pt --source 0

# stream RTSP
python seg_test.py --model best.pt --source rtsp://...

# pasta de imagens
python seg_test.py --model best.pt --source ./imagens/

# so listar as classes do modelo
python seg_test.py --model best.pt --list-classes

# servidor sem display (nao abre janela)
python seg_test.py --model best.pt --source video.mp4 --no-show
```

Pressione `q` na janela pra encerrar video/webcam.

## Opcoes

| Flag | Descricao | Padrao |
|------|-----------|--------|
| `--model`, `-m` | Caminho do modelo `.pt` | `model/` |
| `--source`, `-s` | Imagem, video, pasta, indice da webcam ou URL de stream | `inputs/` |
| `--classes`, `-c` | Nomes das classes a mostrar (ex.: `person car`) | todas |
| `--conf` | Confianca minima | `0.25` |
| `--iou` | IoU do NMS | `0.7` |
| `--imgsz` | Tamanho da inferencia | `640` |
| `--device` | `cpu`, `0`, `0,1`... | auto |
| `--output`, `-o` | Pasta de saida | `outputs` |
| `--no-show` | Nao abrir janela (headless) | - |
| `--no-save` | Nao salvar arquivos | - |
| `--list-classes` | So lista as classes e sai | - |

Os resultados anotados sao salvos na pasta `--output`. As classes informadas em
`--classes` sao validadas contra o modelo; se uma nao existir, o script lista as
classes disponiveis e encerra.
