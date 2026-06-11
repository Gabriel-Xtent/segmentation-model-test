#!/usr/bin/env python3
"""Testador de modelos de segmentacao YOLO.

Roda um modelo de segmentacao (.pt) sobre uma imagem, video, pasta de imagens
ou webcam, desenhando as mascaras e os nomes das classes. Permite filtrar
quais classes mostrar usando o nome da classe (class name).

Exemplos:
    # imagem, mostra todas as classes
    python seg_test.py --model yolov8n-seg.pt --source foto.jpg

    # video, mostrando so as classes "person" e "car"
    python seg_test.py --model best.pt --source video.mp4 --classes person car

    # webcam (indice 0)
    python seg_test.py --model best.pt --source 0

    # so listar as classes do modelo e sair
    python seg_test.py --model best.pt --list-classes
"""

import argparse
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
DEFAULT_INPUT_DIR = "inputs"
DEFAULT_MODEL_DIR = "model"


def parse_args():
    p = argparse.ArgumentParser(
        description="Testa modelos de segmentacao YOLO em imagem/video/webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", "-m", default=None,
                   help="Caminho do modelo YOLO (.pt). "
                        "Padrao: usa o modelo que estiver na pasta model/.")
    p.add_argument("--source", "-s", default=None,
                   help="Imagem, video, pasta ou indice da webcam. "
                        "Padrao: processa tudo da pasta inputs/.")
    p.add_argument("--classes", "-c", nargs="+", default=None, metavar="NOME",
                   help="Nomes das classes a mostrar (ex.: person car). Padrao: todas.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confianca minima da deteccao (padrao: 0.25).")
    p.add_argument("--iou", type=float, default=0.7,
                   help="IoU do NMS (padrao: 0.7).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Tamanho da imagem de inferencia (padrao: 640).")
    p.add_argument("--device", default=None,
                   help="Dispositivo: cpu, 0, 0,1 ... (padrao: auto).")
    p.add_argument("--output", "-o", default="outputs",
                   help="Pasta onde salvar os resultados anotados (padrao: outputs).")
    p.add_argument("--no-show", action="store_true",
                   help="Nao abrir janela ao vivo (modo headless / servidor).")
    p.add_argument("--no-save", action="store_true",
                   help="Nao salvar arquivos de saida.")
    p.add_argument("--list-classes", action="store_true",
                   help="Apenas listar as classes do modelo e sair.")
    return p.parse_args()


def resolve_class_ids(model_names, wanted):
    """Converte nomes de classe em ids, validando contra o modelo."""
    # model.names eh um dict {id: nome}
    name_to_id = {name.lower(): cid for cid, name in model_names.items()}
    ids, missing = [], []
    for w in wanted:
        cid = name_to_id.get(w.lower())
        if cid is None:
            missing.append(w)
        else:
            ids.append(cid)
    if missing:
        disponiveis = ", ".join(sorted(model_names.values()))
        sys.exit(
            f"Erro: classe(s) nao encontrada(s) no modelo: {', '.join(missing)}\n"
            f"Classes disponiveis: {disponiveis}"
        )
    return ids


def resolve_model(model_arg):
    """Retorna o caminho do modelo .pt a usar."""
    if model_arg:
        path = Path(model_arg)
        if not path.exists():
            sys.exit(f"Erro: modelo nao encontrado: {model_arg}")
        return str(path)
    # padrao: pega o .pt da pasta model/
    model_dir = Path(DEFAULT_MODEL_DIR)
    model_dir.mkdir(exist_ok=True)
    pts = sorted(model_dir.glob("*.pt"))
    if not pts:
        sys.exit(
            f"Nenhum modelo encontrado em '{model_dir}/'.\n"
            f"Coloque um arquivo .pt em '{model_dir}/' ou use --model."
        )
    if len(pts) > 1:
        nomes = ", ".join(p.name for p in pts)
        sys.exit(
            f"Mais de um modelo em '{model_dir}/': {nomes}\n"
            f"Deixe apenas um, ou escolha com --model {model_dir}/<arquivo>.pt"
        )
    return str(pts[0])


def classify_source(source):
    """Retorna ('webcam'|'image'|'video'|'folder'|'stream', valor)."""
    if source is None:
        # padrao: processa tudo da pasta inputs/
        inputs = Path(DEFAULT_INPUT_DIR)
        inputs.mkdir(exist_ok=True)
        arquivos = [p for p in inputs.iterdir()
                    if p.suffix.lower() in IMG_EXTS | VID_EXTS]
        if not arquivos:
            sys.exit(
                f"Nada pra processar: a pasta '{inputs}/' esta vazia.\n"
                f"Coloque imagens/videos em '{inputs}/' ou use --source."
            )
        return "folder", str(inputs)
    # webcam por indice
    if source.isdigit():
        return "webcam", int(source)
    # stream
    if source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return "stream", source
    path = Path(source)
    if not path.exists():
        sys.exit(f"Erro: source nao encontrado: {source}")
    if path.is_dir():
        return "folder", str(path)
    ext = path.suffix.lower()
    if ext in IMG_EXTS:
        return "image", str(path)
    if ext in VID_EXTS:
        return "video", str(path)
    # deixa o YOLO tentar
    return "video", str(path)


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics nao esta instalado. Instale com:\n"
            "    pip install -r requirements.txt\n"
            "ou:\n"
            "    pip install ultralytics opencv-python"
        )

    model_path = resolve_model(args.model)

    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path)

    # valida que e um modelo de segmentacao
    task = getattr(model, "task", None)
    if task and task != "segment":
        print(f"Aviso: o modelo parece ser de '{task}', nao de segmentacao. "
              f"As mascaras podem nao aparecer.", file=sys.stderr)

    print("\nClasses do modelo:")
    for cid, name in sorted(model.names.items()):
        print(f"  [{cid}] {name}")

    if args.list_classes:
        return

    class_ids = None
    if args.classes:
        class_ids = resolve_class_ids(model.names, args.classes)
        nomes = ", ".join(model.names[c] for c in class_ids)
        print(f"\nFiltrando classes: {nomes}")

    kind, value = classify_source(args.source)
    show = not args.no_show
    save = not args.no_save

    print(f"\nSource: {value}  (tipo: {kind})")
    if save:
        print(f"Salvando em: {args.output}")
    print("Rodando inferencia...  (pressione 'q' na janela para sair)\n")

    # stream=True processa frame a frame (ideal pra video/webcam)
    stream = kind in {"video", "webcam", "stream", "folder"}

    common = dict(
        source=value,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=class_ids,
        device=args.device,
        save=save,
        project=str(Path(args.output).resolve().parent),
        name=Path(args.output).name,
        exist_ok=True,
        verbose=False,
    )

    if not stream:
        # imagem unica (ou poucas) -- nao usa streaming
        results = model.predict(**common)
        if show:
            _show_results(results)
        _print_summary(results, model.names)
        if save and results:
            print(f"\nResultado salvo em: {results[0].save_dir}")
        return

    # video / webcam / pasta -- streaming frame a frame
    results = model.predict(stream=True, **common)
    total, save_dir = _stream_loop(results, model.names, show)
    print(f"\nProcessados {total} frame(s).")
    if save and save_dir:
        print(f"Resultado salvo em: {save_dir}")


def _show_results(results):
    import cv2
    for r in results:
        frame = r.plot()
        cv2.imshow("YOLO segmentation", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def _stream_loop(results, names, show):
    import cv2
    count = 0
    save_dir = None
    try:
        for r in results:
            count += 1
            save_dir = r.save_dir
            _print_frame_counts(r, names, count)
            if show:
                frame = r.plot()
                cv2.imshow("YOLO segmentation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if show:
            cv2.destroyAllWindows()
    return count, save_dir


def _print_frame_counts(r, names, idx):
    if r.boxes is None or len(r.boxes) == 0:
        return
    counts = {}
    for cid in r.boxes.cls.tolist():
        nome = names[int(cid)]
        counts[nome] = counts.get(nome, 0) + 1
    resumo = ", ".join(f"{n}: {q}" for n, q in sorted(counts.items()))
    print(f"  frame {idx}: {resumo}")


def _print_summary(results, names):
    counts = {}
    for r in results:
        if r.boxes is None:
            continue
        for cid in r.boxes.cls.tolist():
            nome = names[int(cid)]
            counts[nome] = counts.get(nome, 0) + 1
    if counts:
        print("\nDeteccoes:")
        for n, q in sorted(counts.items()):
            print(f"  {n}: {q}")
    else:
        print("\nNenhuma deteccao.")


if __name__ == "__main__":
    main()
