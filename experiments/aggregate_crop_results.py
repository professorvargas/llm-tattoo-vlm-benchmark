#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

def load_id2name(path: Path) -> Dict[int, str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        try:
            k0 = next(iter(obj.keys()))
            int(k0)
            return {int(k): str(v) for k, v in obj.items()}
        except Exception:
            name2id = {str(k): int(v) for k, v in obj.items()}
            return {v: k for k, v in name2id.items()}
    if isinstance(obj, list):
        return {i: str(name) for i, name in enumerate(obj)}
    raise ValueError(f"Unexpected id2name format: {path}")

def allowed_label_set(id2name: Dict[int, str]) -> Set[str]:
    s = set(id2name.values())
    s.discard("background")
    return set(x.lower() for x in s)

def find_key(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None

def extract_image_path(rec: dict) -> Optional[str]:
    k = find_key(rec, ["image_path", "image", "path", "image_file", "file"])
    if k:
        return rec.get(k)
    for kk in ["meta", "input", "data"]:
        if isinstance(rec.get(kk), dict):
            k2 = find_key(rec[kk], ["image_path", "image", "path", "file"])
            if k2:
                return rec[kk].get(k2)
    return None

def extract_json_obj(rec: dict) -> Optional[dict]:
    if isinstance(rec.get("json_obj"), dict):
        return rec["json_obj"]
    if isinstance(rec.get("parsed_json"), dict):
        return rec["parsed_json"]
    if isinstance(rec.get("json"), dict):
        return rec["json"]
    return None

def split_from_path(p: Path) -> Optional[str]:
    parts = set(p.parts)
    if "test_open" in parts: return "test_open"
    if "test_closed" in parts: return "test_closed"
    return None

def compute_gt_labels_from_mask(split: str, image_id: str, id2name: Dict[int, str], min_area: int) -> Set[str]:
    mask_dir = Path("datasets") / split / "mask_ids"
    mask_path = mask_dir / f"{image_id}_ids.npy"
    if not mask_path.exists():
        cands = list(mask_dir.glob(f"{image_id}*ids*.npy"))
        if not cands:
            return set()
        mask_path = cands[0]

    mask = np.load(mask_path)
    if mask.ndim == 3:
        mask = mask.squeeze()

    labels: Set[str] = set()
    for cid in np.unique(mask):
        cid_int = int(cid)
        if cid_int == 0:
            continue
        name = id2name.get(cid_int, f"class_{cid_int}")
        if name == "background":
            continue
        area = int((mask == cid_int).sum())
        if area >= min_area:
            labels.add(name.lower())
    return labels

def f1_from_pr(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2 * p * r / (p + r))

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    inter = len(a & b); uni = len(a | b)
    return inter / uni if uni else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test_open", "test_closed", "both"], default="both")
    ap.add_argument("--jsonl-open", default="experiments/p1_crop_test_open.jsonl")
    ap.add_argument("--jsonl-closed", default="experiments/p1_crop_test_closed.jsonl")
    ap.add_argument("--id2name", default="datasets/tssd2023_id2name.json")
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--out-dir", default="datasets")
    args = ap.parse_args()

    id2name = load_id2name(Path(args.id2name))
    allowed = allowed_label_set(id2name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_by_image: Dict[Tuple[str, str], Set[str]] = {}

    def ingest_jsonl(jsonl_path: Path):
        if not jsonl_path.exists():
            print(f"[warn] missing jsonl: {jsonl_path}")
            return
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                img_path_str = extract_image_path(rec)
                if not img_path_str:
                    continue
                p = Path(img_path_str)

                sp = rec.get("split") or split_from_path(p)
                if sp not in ("test_open", "test_closed"):
                    continue

                # crops: .../<split>/<image_id>/<gt_label>.png
                image_id = p.parent.name

                obj = extract_json_obj(rec) if rec.get("json_ok", True) else None
                labels = []
                if isinstance(obj, dict):
                    if isinstance(obj.get("labels"), list):
                        labels = obj.get("labels") or []
                    elif isinstance(obj.get("label"), str):
                        labels = [obj.get("label")]
                if isinstance(obj, list):
                    labels = obj

                labels_norm = []
                for lab in labels:
                    if not isinstance(lab, str):
                        continue
                    lab = lab.strip().lower()
                    if lab == "spider" and "spide" in allowed:
                        lab = "spide"
                    if lab in allowed:
                        labels_norm.append(lab)

                key = (sp, image_id)
                pred_by_image.setdefault(key, set()).update(labels_norm)

    splits = []
    if args.split in ("test_open", "both"): splits.append("test_open")
    if args.split in ("test_closed", "both"): splits.append("test_closed")

    if "test_open" in splits: ingest_jsonl(Path(args.jsonl_open))
    if "test_closed" in splits: ingest_jsonl(Path(args.jsonl_closed))

    rows_full = []
    sum_tp = sum_fp = sum_fn = 0
    jaccs: List[float] = []
    n_images = 0

    for sp in splits:
        img_dir = Path("datasets") / sp / "images"
        images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        for img_path in images:
            image_id = img_path.stem
            gt = compute_gt_labels_from_mask(sp, image_id, id2name, args.min_area)
            pred = pred_by_image.get((sp, image_id), set())

            if not gt:
                continue

            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = f1_from_pr(prec, rec)
            jac = jaccard(gt, pred)

            sum_tp += tp; sum_fp += fp; sum_fn += fn
            jaccs.append(jac); n_images += 1

            rows_full.append({
                "split": sp,
                "image_id": image_id,
                "gt_labels": ";".join(sorted(gt)),
                "pred_labels": ";".join(sorted(pred)),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 6),
                "recall": round(rec, 6),
                "f1": round(f1, 6),
                "jaccard": round(jac, 6),
                "hallucinations_fp": fp,
                "omissions_fn": fn,
            })

    pred_out = out_dir / "pred_crop_labels_per_image.csv"
    with pred_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_id", "pred_labels"])
        for (sp, image_id), labs in sorted(pred_by_image.items()):
            w.writerow([sp, image_id, ";".join(sorted(labs))])

    full_out = out_dir / "metrics_crop_full.csv"
    if rows_full:
        with full_out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows_full[0].keys())
            dw = csv.DictWriter(f, fieldnames=fieldnames)
            dw.writeheader()
            dw.writerows(rows_full)

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) else 0.0
    micro_f1 = f1_from_pr(micro_p, micro_r)
    mean_j = float(np.mean(jaccs)) if jaccs else 0.0

    summary_out = out_dir / "summary_crop.csv"
    with summary_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_images", "micro_precision", "micro_recall", "micro_f1", "mean_jaccard", "tp", "fp", "fn", "min_area"])
        w.writerow([n_images, round(micro_p, 6), round(micro_r, 6), round(micro_f1, 6), round(mean_j, 6), sum_tp, sum_fp, sum_fn, args.min_area])

    base_path = out_dir / "summary_p1_v2.csv"
    compare_out = out_dir / "summary_compare_p1_vs_crop.csv"
    if base_path.exists():
        with base_path.open("r", encoding="utf-8") as f_in, compare_out.open("w", newline="", encoding="utf-8") as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames or []
            extra = ["crop_micro_precision", "crop_micro_recall", "crop_micro_f1", "crop_mean_jaccard", "crop_tp", "crop_fp", "crop_fn", "crop_min_area"]
            out_fields = fieldnames + [e for e in extra if e not in fieldnames]
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()
            for row in reader:
                row.update({
                    "crop_micro_precision": round(micro_p, 6),
                    "crop_micro_recall": round(micro_r, 6),
                    "crop_micro_f1": round(micro_f1, 6),
                    "crop_mean_jaccard": round(mean_j, 6),
                    "crop_tp": sum_tp,
                    "crop_fp": sum_fp,
                    "crop_fn": sum_fn,
                    "crop_min_area": args.min_area,
                })
                writer.writerow(row)

    print("[ok] wrote:")
    print(f"- {pred_out}")
    print(f"- {full_out}")
    print(f"- {summary_out}")
    if base_path.exists():
        print(f"- {compare_out}")

if __name__ == "__main__":
    main()
