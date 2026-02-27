import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def _read_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normaliza colunas esperadas (tenta ser tolerante com nomes)
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("image_id", "img_id", "image"):
            col_map[c] = "image_id"
        elif lc in ("split",):
            col_map[c] = "split"
        elif lc in ("gt_labels", "gt", "ground_truth", "groundtruth"):
            col_map[c] = "gt_labels"
        elif lc in ("pred_labels", "pred", "prediction", "predicted"):
            col_map[c] = "pred_labels"
        elif lc in ("f1", "micro_f1", "f1_score"):
            col_map[c] = "f1"
        elif lc in ("jaccard", "iou", "mean_iou"):
            col_map[c] = "jaccard"
        elif lc in ("fp", "false_positives"):
            col_map[c] = "fp"
        elif lc in ("fn", "false_negatives"):
            col_map[c] = "fn"

    df = df.rename(columns=col_map)

    required = {"image_id", "split", "gt_labels", "pred_labels", "f1", "jaccard", "fp", "fn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} está sem colunas necessárias: {sorted(missing)}.\n"
                         f"Colunas encontradas: {list(df.columns)}")
    return df


def _find_file_by_image_id(root: Path, image_id: str, exts=(".jpg", ".jpeg", ".png")) -> Path | None:
    # tenta match direto
    for ext in exts:
        p = root / f"{image_id}{ext}"
        if p.exists():
            return p
    # busca recursiva
    for ext in exts:
        hits = list(root.rglob(f"{image_id}{ext}"))
        if hits:
            return hits[0]
    return None


def _find_mask(root: Path, image_id: str) -> Path | None:
    # tenta padrões comuns
    candidates = [
        root / f"{image_id}_mask.png",
        root / f"{image_id}_mask.jpg",
        root / f"{image_id}_mask.jpeg",
        root / f"{image_id}_mask.bmp",
    ]
    for c in candidates:
        if c.exists():
            return c
    # busca algo contendo image_id e "mask"
    hits = list(root.rglob(f"*{image_id}*mask*.*"))
    return hits[0] if hits else None


def _load_pil(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def _plot_crops(ax, crop_paths: list[Path], title: str):
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if not crop_paths:
        ax.text(0.5, 0.5, "(sem crops)", ha="center", va="center")
        return

    # monta uma “tira” simples (1 linha) se poucas; senão, 2 linhas
    n = len(crop_paths)
    rows = 1 if n <= 5 else 2
    cols = (n + rows - 1) // rows

    # subgrid manual (sem complicar)
    for i, cp in enumerate(crop_paths[: rows * cols]):
        r = i // cols
        c = i % cols
        # coordenadas normalizadas
        w = 1 / cols
        h = 1 / rows
        x0 = c * w
        y0 = 1 - (r + 1) * h

        im = _load_pil(cp)
        # inset axes
        iax = ax.inset_axes([x0 + 0.01, y0 + 0.01, w - 0.02, h - 0.02])
        iax.imshow(im)
        iax.axis("off")


def _fmt_block(title: str, row: pd.Series) -> str:
    return (
        f"{title}\n"
        f"GT:   {row['gt_labels']}\n"
        f"Pred: {row['pred_labels']}\n"
        f"F1={row['f1']:.3f} | Jaccard={row['jaccard']:.3f} | FP={int(row['fp'])} | FN={int(row['fn'])}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test_open", "test_closed"])
    ap.add_argument("--image-ids", nargs="+", required=True)
    ap.add_argument("--baseline-metrics", required=True)
    ap.add_argument("--black-metrics", required=True)
    ap.add_argument("--white-metrics", required=True)
    ap.add_argument("--crops-black-dir", required=True)
    ap.add_argument("--crops-white-dir", required=True)
    ap.add_argument("--datasets-dir", default="datasets", help="raiz onde estão test_open/test_closed")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    split = args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_base = _read_metrics(Path(args.baseline_metrics))
    df_blk  = _read_metrics(Path(args.black_metrics))
    df_wht  = _read_metrics(Path(args.white_metrics))

    ds_root = Path(args.datasets_dir)
    split_root = ds_root / split

    crops_black_root = Path(args.crops_black_dir)
    crops_white_root = Path(args.crops_white_dir)

    for image_id in args.image_ids:
        base_row = df_base[(df_base["split"] == split) & (df_base["image_id"] == image_id)]
        blk_row  = df_blk[(df_blk["split"] == split) & (df_blk["image_id"] == image_id)]
        wht_row  = df_wht[(df_wht["split"] == split) & (df_wht["image_id"] == image_id)]

        if base_row.empty or blk_row.empty or wht_row.empty:
            print(f"[skip] faltou linha em algum csv para {split}/{image_id}")
            continue

        base_row = base_row.iloc[0]
        blk_row  = blk_row.iloc[0]
        wht_row  = wht_row.iloc[0]

        img_path = _find_file_by_image_id(split_root, image_id)
        mask_path = _find_mask(split_root, image_id)

        if img_path is None:
            print(f"[skip] não achei imagem original para {split}/{image_id} dentro de {split_root}")
            continue

        img = _load_pil(img_path)
        mask = _load_pil(mask_path) if mask_path and mask_path.exists() else None

        crop_black = sorted((crops_black_root / image_id).glob("*.png"))
        crop_white = sorted((crops_white_root / image_id).glob("*.png"))

        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(f"{split}/{image_id} — Baseline vs GT-crops (black) vs GT-crops (white)", fontsize=14)

        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 0.9], width_ratios=[1.2, 1])

        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(img)
        ax_img.set_title("Imagem original", fontsize=11)
        ax_img.axis("off")

        ax_mask = fig.add_subplot(gs[0, 1])
        if mask is not None:
            ax_mask.imshow(mask)
            ax_mask.set_title("GT mask (cores do dataset)", fontsize=11)
        else:
            ax_mask.text(0.5, 0.5, "(mask não encontrada)", ha="center", va="center")
        ax_mask.axis("off")

        # Texto do baseline por cima (abaixo do topo)
        fig.text(0.02, 0.60, _fmt_block("BASELINE (imagem inteira)", base_row), fontsize=10, family="monospace")

        ax_blk_crops = fig.add_subplot(gs[1, 0])
        _plot_crops(ax_blk_crops, crop_black, "GT-crops (fundo preto)")

        ax_blk_txt = fig.add_subplot(gs[1, 1])
        ax_blk_txt.axis("off")
        ax_blk_txt.text(0.01, 0.9, _fmt_block("AGREGADO (crops preto)", blk_row),
                        fontsize=10, family="monospace", va="top")

        ax_wht_crops = fig.add_subplot(gs[2, 0])
        _plot_crops(ax_wht_crops, crop_white, "GT-crops (fundo branco)")

        ax_wht_txt = fig.add_subplot(gs[2, 1])
        ax_wht_txt.axis("off")
        ax_wht_txt.text(0.01, 0.9, _fmt_block("AGREGADO (crops branco)", wht_row),
                        fontsize=10, family="monospace", va="top")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"fig_{split}_{image_id}_baseline_vs_black_vs_white.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
