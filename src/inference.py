# src/inference.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import utils
import model as m


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cats vs Dogs inference (batch or single image).")
    p.add_argument("--config", type=str, default="configurations/base.yaml",
                   help="Path to YAML config.")
    p.add_argument("--checkpoint", type=str, default="runs/best.pt",
                   help="Path to model weights (.pt).")
    p.add_argument("--input", type=str, required=True,
                   help="Path to an image file or a folder of images.")
    p.add_argument("--output", type=str, default="runs/predictions.jsonl",
                   help="Path to write predictions (JSONL).")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cpu/cuda). If omitted uses config/utils default.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size (default uses config).")
    return p.parse_args()


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Input file must be an image. Got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path not found: {input_path}")

    paths = [p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not paths:
        raise ValueError(f"No images found in: {input_path}")
    return sorted(paths)


def build_transform(image_size: int) -> transforms.Compose:
    # Keep this consistent with your training preprocessing!
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # If you normalized during training, add the same normalization here:
        # transforms.Normalize(mean=[...], std=[...]),
    ])


def load_image(path: Path, tfm: transforms.Compose) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tfm(img)


def batchify(tensors: List[torch.Tensor], batch_size: int) -> List[torch.Tensor]:
    batches = []
    for i in range(0, len(tensors), batch_size):
        batches.append(torch.stack(tensors[i:i + batch_size], dim=0))
    return batches


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)

    # device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = utils.get_device(cfg)

    # batch size
    batch_size = args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"]

    # labels (adjust if you used different label mapping)
    label_map = {0: "cat", 1: "dog"}

    # model
    model = m.build_model()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # inputs
    input_path = Path(args.input)
    img_paths = list_images(input_path)

    image_size = cfg["dataset"]["image_size"]
    tfm = build_transform(image_size)

    # load all tensors (simple + fine for small datasets; for huge folders build a Dataset/DataLoader)
    tensors: List[torch.Tensor] = []
    for p in tqdm(img_paths, desc="Loading images"):
        tensors.append(load_image(p, tfm))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # inference
    idx = 0
    with open(out_path, "w", encoding="utf-8") as f:
        with torch.inference_mode():
            for batch in tqdm(batchify(tensors, batch_size), desc="Running inference"):
                batch = batch.to(device)

                logits = model(batch)                 # shape [B, 2]
                probs = torch.softmax(logits, dim=1)  # shape [B, 2]
                pred = probs.argmax(dim=1)            # shape [B]

                probs_cpu = probs.detach().cpu().tolist()
                pred_cpu = pred.detach().cpu().tolist()

                for j in range(len(pred_cpu)):
                    pth = str(img_paths[idx])
                    pred_id = int(pred_cpu[j])
                    record = {
                        "path": pth,
                        "pred_id": pred_id,
                        "pred_label": label_map.get(pred_id, str(pred_id)),
                        "probs": probs_cpu[j],  # [p_cat, p_dog] if that's your label order
                    }
                    f.write(json.dumps(record) + "\n")
                    idx += 1

    print(f"✅ Wrote {len(img_paths)} predictions to: {out_path}")


if __name__ == "__main__":
    main()
