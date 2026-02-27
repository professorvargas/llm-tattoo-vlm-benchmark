import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_id2name(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    # dict id->name
    if isinstance(obj, dict):
        try:
            k0 = next(iter(obj.keys()))
            int(k0)
            return {int(k): str(v) for k, v in obj.items()}
        except Exception:
            pass

        # dict name->id
        try:
            name2id = {str(k): int(v) for k, v in obj.items()}
            return {v: k for k, v in name2id.items()}
        except Exception:
            raise ValueError(f"Formato inesperado em {path}")

    # list index->name
    if isinstance(obj, list):
        return {i: str(name) for i, name in enumerate(obj)}

    raise ValueError(f"Formato inesperado em {path}")


def bbox_from_mask(m: np.ndarray):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test_open", "test_closed", "both"], default="both")
    ap.add_argument("--min-area", type=int, default=200, help="Ignora regiões muito pequenas")
    ap.add_argument("--pad", type=int, default=6, help="Padding no bounding box")
    ap.add_argument("--out-dir", default="datasets/crops_gt")
    ap.add_argument("--id2name", default="datasets/tssd2023_id2name.json")
    ap.add_argument("--bg", choices=["black", "white"], default="black", help="Cor do fundo fora da máscara (black=0, white=255)")
    args = ap.parse_args()

    id2name = load_id2name(Path(args.id2name))
    root = Path("datasets")
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    splits = []
    if args.split in ("test_open", "both"):
        splits.append("test_open")
    if args.split in ("test_closed", "both"):
        splits.append("test_closed")

    for split_name in splits:
        img_dir = root / split_name / "images"
        mask_dir = root / split_name / "mask_ids"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask_ids: {mask_dir}")

        out_split = out_base / split_name
        out_split.mkdir(parents=True, exist_ok=True)

        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        for img_path in imgs:
            base = img_path.stem
            mask_path = mask_dir / f"{base}_ids.npy"
            if not mask_path.exists():
                # fallback
                cands = list(mask_dir.glob(f"{base}*ids*.npy"))
                if not cands:
                    continue
                mask_path = cands[0]

            img = Image.open(img_path).convert("RGB")
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = mask.squeeze()

            h, w = mask.shape[:2]
            if img.size != (w, h):
                # se divergir, ignora para não gerar recorte errado
                continue

            classes = sorted([int(c) for c in np.unique(mask) if int(c) != 0])

            out_img_dir = out_split / base
            out_img_dir.mkdir(parents=True, exist_ok=True)

            for cid in classes:
                name = id2name.get(cid, f"class_{cid}")
                if name == "background":
                    continue

                binmask = (mask == cid)
                area = int(binmask.sum())
                if area < args.min_area:
                    continue

                bb = bbox_from_mask(binmask)
                if not bb:
                    continue
                x1, y1, x2, y2 = bb

                pad = args.pad
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)

                crop_img = img.crop((x1, y1, x2 + 1, y2 + 1))
                crop_mask = binmask[y1:y2 + 1, x1:x2 + 1]

                crop_np = np.array(crop_img)
                bg = 0 if args.bg == "black" else 255
                crop_np[~crop_mask] = bg
                out = Image.fromarray(crop_np)

                out_path = out_img_dir / f"{name}.png"
                out.save(out_path)

        print(f"[ok] crops written for split={split_name} -> {out_split}")


if __name__ == "__main__":
    main()
