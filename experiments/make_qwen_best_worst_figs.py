#!/usr/bin/env python3
import argparse, csv, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image


def read_vocab(vocab_path: Path):
    lines = [l.strip() for l in vocab_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, seen = [], set()
    for x in lines:
        if x not in seen:
            out.append(x); seen.add(x)
    return out, set(out)


def parse_labels(val, vocab_set):
    if val is None:
        return {"unknown"}

    if isinstance(val, dict):
        for k in ("labels", "label", "prediction", "pred", "output"):
            if k in val:
                return parse_labels(val[k], vocab_set)
        val = json.dumps(val, ensure_ascii=False)

    if isinstance(val, list):
        items = []
        for x in val:
            if isinstance(x, str):
                items.extend(re.split(r"[,;\n|]+", x))
            else:
                items.append(str(x))
    else:
        items = re.split(r"[,;\n|]+", str(val))

    cleaned = []
    for t in items:
        t = t.strip().lower()
        if t:
            cleaned.append(t)

    out = {t for t in cleaned if t in vocab_set}
    return out if out else {"unknown"}


def metrics(gt, pred):
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
    jac  = tp / (tp + fp + fn) if (tp+fp+fn) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, jaccard=jac)


def find_image_path(split, image_id):
    img_dir = Path("datasets")/split/"images"
    cand = sorted(img_dir.glob(f"{image_id}.*"))
    return cand[0] if cand else None


def find_mask_path(split, image_id):
    # Se não existir, o script só mostra “mask não encontrada” (sem quebrar).
    for sub in ("mask_rgb", "masks_rgb", "mask", "masks"):
        mdir = Path("datasets")/split/sub
        if mdir.exists():
            cand = sorted(mdir.glob(f"{image_id}_mask.*")) + sorted(mdir.glob(f"{image_id}.*"))
            if cand:
                return cand[0]
    return None


def gt_labels_and_crops(split, image_id):
    d = Path("datasets/crops_gt")/split/image_id
    crops = sorted(d.glob("*.png"))
    labs = [p.stem for p in crops]
    return (set(labs) if labs else {"unknown"}), crops


def load_baseline_preds(jsonl_path, vocab_set):
    m = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            img = r.get("image") or r.get("image_path") or r.get("path")
            if not img:
                continue
            image_id = Path(img).stem
            pred = parse_labels(r.get("output") or r.get("json_obj") or r.get("json_text"), vocab_set)
            m[image_id] = pred
    return m


def load_crop_preds(jsonl_path, vocab_set):
    per_img = {}
    per_crop = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            crop = r.get("crop_image") or r.get("image") or r.get("path")
            if not crop:
                continue
            p = Path(crop)
            parts = p.parts

            if "test_open" in parts:
                idx = parts.index("test_open")
            elif "test_closed" in parts:
                idx = parts.index("test_closed")
            else:
                continue

            # padrão esperado: datasets/crops_gt/<split>/<image_id>/<label>.png
            if idx + 1 >= len(parts):
                continue
            image_id = parts[idx+1]

            pred = parse_labels(r.get("output") or r.get("json_obj") or r.get("json_text"), vocab_set)

            crop_key = p.as_posix()
            per_crop[crop_key] = pred
            per_img.setdefault(image_id, set()).update(pred)

    return per_img, per_crop


def read_rank_csv(path, topk):
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= topk:
                break
            image_id = (row.get("image_id") or row.get("image") or row.get("id") or "").strip()
            if image_id:
                out.append(image_id)
    return out


def draw_crop_grid(fig, subspec, crop_paths, crop_pred_map, vocab_set, title):
    ax_bg = fig.add_subplot(subspec)
    ax_bg.axis("off")
    ax_bg.set_title(title, fontsize=10, loc="left")

    n = len(crop_paths)
    if n == 0:
        ax_bg.text(0.0, 0.5, "(sem crops encontrados)", fontsize=9)
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    gs = GridSpecFromSubplotSpec(rows, cols, subplot_spec=subspec, wspace=0.15, hspace=0.25)

    for i, p in enumerate(crop_paths):
        r = i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        try:
            im = Image.open(p).convert("RGB")
            ax.imshow(im)
        except Exception:
            ax.text(0.5, 0.5, "erro ao abrir", ha="center", va="center")
        ax.axis("off")

        pred = crop_pred_map.get(p.as_posix(), {"unknown"})
        pred = {x for x in pred if x in vocab_set} or {"unknown"}
        ax.set_title(",".join(sorted(pred)), fontsize=8)


def save_one(out_png, split, image_id, vocab_set,
             baseline_pred, black_pred_img, black_pred_crop,
             white_pred_img,  white_pred_crop):
    img_path = find_image_path(split, image_id)
    mask_path = find_mask_path(split, image_id)
    gt, gt_crop_paths = gt_labels_and_crops(split, image_id)

    gt_crop_paths_white = []
    for p in gt_crop_paths:
        gt_crop_paths_white.append(Path(str(p).replace("datasets/crops_gt/", "datasets/crops_gt_white/")))

    pred_base  = baseline_pred.get(image_id, {"unknown"})
    pred_black = black_pred_img.get(image_id, {"unknown"})
    pred_white = white_pred_img.get(image_id, {"unknown"})

    m_base  = metrics(gt, pred_base)
    m_black = metrics(gt, pred_black)
    m_white = metrics(gt, pred_white)

    fig = plt.figure(figsize=(13, 10), dpi=150)
    fig.suptitle(f"{split}/{image_id} — Baseline vs GT-crops (preto) vs GT-crops (branco)", fontsize=12)

    gs = GridSpec(4, 2, figure=fig, height_ratios=[3.2, 0.9, 2.2, 2.2], width_ratios=[3, 2])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1]); ax1.axis("off")
    ax0.set_title("Imagem original", fontsize=10, loc="left")
    ax1.set_title("GT mask (cores do dataset)", fontsize=10, loc="left")

    if img_path and img_path.exists():
        ax0.imshow(Image.open(img_path).convert("RGB"))
    else:
        ax0.text(0.5, 0.5, "imagem não encontrada", ha="center", va="center")

    if mask_path and mask_path.exists():
        ax1.imshow(Image.open(mask_path).convert("RGB"))
    else:
        ax1.text(0.5, 0.5, "mask não encontrada", ha="center", va="center")

    ax_txt = fig.add_subplot(gs[1, :]); ax_txt.axis("off")
    ax_txt.text(
        0.0, 0.9,
        "Baseline (imagem inteira)\n"
        f"GT: {', '.join(sorted(gt))}\n"
        f"Pred: {', '.join(sorted(pred_base))}\n"
        f"F1={m_base['f1']:.3f} | Jaccard={m_base['jaccard']:.3f} | FP={m_base['fp']} | FN={m_base['fn']}",
        fontsize=10, va="top"
    )

    draw_crop_grid(fig, gs[2, 0], gt_crop_paths, black_pred_crop, vocab_set,
                   "GT-crops (segmentação GT) + predição por crop (VLM) + agregação — fundo preto")
    axb = fig.add_subplot(gs[2, 1]); axb.axis("off")
    axb.text(
        0.0, 0.9,
        "Agregado (união das labels dos crops)\n"
        f"GT: {', '.join(sorted(gt))}\n"
        f"Pred: {', '.join(sorted(pred_black))}\n"
        f"F1={m_black['f1']:.3f} | Jaccard={m_black['jaccard']:.3f} | FP={m_black['fp']} | FN={m_black['fn']}",
        fontsize=10, va="top"
    )

    draw_crop_grid(fig, gs[3, 0], gt_crop_paths_white, white_pred_crop, vocab_set,
                   "GT-crops (segmentação GT) + predição por crop (VLM) + agregação — fundo branco")
    axw = fig.add_subplot(gs[3, 1]); axw.axis("off")
    axw.text(
        0.0, 0.9,
        "Agregado (união das labels dos crops)\n"
        f"GT: {', '.join(sorted(gt))}\n"
        f"Pred: {', '.join(sorted(pred_white))}\n"
        f"F1={m_white['f1']:.3f} | Jaccard={m_white['jaccard']:.3f} | FP={m_white['fp']} | FN={m_white['fn']}",
        fontsize=10, va="top"
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="runs/qwen2_5_vl")
    ap.add_argument("--split", required=True, choices=["test_open", "test_closed"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vocab_path = Path("runs/_shared_links/vocab_35.txt")
    _, vocab_set = read_vocab(vocab_path)

    baseline_jsonl = model_dir/"baseline"/args.split/"raw"/f"p1_baseline_{args.split}.jsonl"
    black_jsonl    = model_dir/"crops_black"/args.split/"raw"/f"p1_crop_{args.split}.jsonl"
    white_jsonl    = model_dir/"crops_white"/args.split/"raw"/f"p1_crop_white_{args.split}.jsonl"

    candidates = set()
    for cond in ("baseline", "crops_black", "crops_white"):
        ev = model_dir/cond/args.split/"eval"
        candidates.update(read_rank_csv(ev/"best_images.csv", args.topk))
        candidates.update(read_rank_csv(ev/"worst_images.csv", args.topk))
    candidates = sorted(candidates)

    baseline_pred = load_baseline_preds(baseline_jsonl, vocab_set)
    black_pred_img, black_pred_crop = load_crop_preds(black_jsonl, vocab_set)
    white_pred_img, white_pred_crop = load_crop_preds(white_jsonl, vocab_set)

    out_dir = Path(args.out_dir) if args.out_dir else (model_dir/"_figures_best_worst"/args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split={args.split} | candidates={len(candidates)} | out={out_dir}")

    for i, image_id in enumerate(candidates, 1):
        out_png = out_dir / f"fig_{args.split}_{image_id}_baseline_vs_black_vs_white.png"
        save_one(out_png, args.split, image_id, vocab_set,
                 baseline_pred, black_pred_img, black_pred_crop,
                 white_pred_img,  white_pred_crop)
        if i % 20 == 0:
            print(f"[{i}/{len(candidates)}] saved {out_png.name}")

    print("Done.")


if __name__ == "__main__":
    main()
