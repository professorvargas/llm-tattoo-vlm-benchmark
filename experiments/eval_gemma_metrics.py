import os, json, csv, math, statistics
from pathlib import Path
from collections import defaultdict

VOCAB_PATH = Path("runs/_shared_links/vocab_35.txt")
VOCAB = [x.strip() for x in VOCAB_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
VOCAB_SET = set(VOCAB)
L = len(VOCAB)

def extract_image_id_from_path(p: str):
    # baseline: datasets/test_open/images/<id>.jpg
    name = Path(p).name
    return Path(name).stem

def extract_image_id_from_crop_path(p: str):
    # crops: datasets/crops_gt(_white)/test_open/<image_id>/<label>.png
    parts = Path(p).parts
    # achar "test_open" ou "test_closed"
    for i, token in enumerate(parts):
        if token in ("test_open", "test_closed"):
            if i + 1 < len(parts):
                return parts[i+1]
    # fallback
    return Path(p).parent.name

def normalize_pred_labels(x):
    if x is None:
        return set(["unknown"])
    if isinstance(x, list):
        out = [str(t).strip() for t in x if str(t).strip()]
    else:
        # string
        out = [t.strip() for t in str(x).split(",") if t.strip()]
    out = [t for t in out if t in VOCAB_SET]
    return set(out) if out else set(["unknown"])

def load_gt_by_image(split: str):
    # GT por imagem = união dos labels (stems) dos crops_gt (independente de black/white)
    root = Path("datasets/crops_gt") / split
    gt = {}
    for img_dir in root.iterdir():
        if not img_dir.is_dir():
            continue
        labels = set()
        for f in img_dir.glob("*.png"):
            labels.add(f.stem)
        if labels:
            gt[img_dir.name] = labels
    return gt

def prf(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    jacc = tp/(tp+fp+fn) if (tp+fp+fn)>0 else 0.0
    return prec, rec, f1, jacc

def row_metrics(pred: set, gold: set):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    prec, rec, f1, jacc = prf(tp, fp, fn)

    # métricas “qualitativas-proxy”
    # - Alucinação (proxy) = FP
    # - Completude (proxy) = Recall
    # - Especificidade/controle: HammingLoss e Overprediction
    hamming = (fp + fn) / L
    overpred = (len(pred) / len(gold)) if len(gold) > 0 else float("inf")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1, "jaccard": jacc,
        "hamming_loss": hamming,
        "overprediction": overpred,
        "pred_size": len(pred),
        "gold_size": len(gold),
    }

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize(rows, key="f1"):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {}
    vals_sorted = sorted(vals)
    def pct(p):
        if not vals_sorted: return None
        k = int(math.ceil(p*len(vals_sorted))) - 1
        k = max(0, min(k, len(vals_sorted)-1))
        return vals_sorted[k]
    return {
        "n": len(vals),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "p90": float(pct(0.90)),
        "min": float(vals_sorted[0]),
        "max": float(vals_sorted[-1]),
    }

def eval_baseline(split: str, run_jsonl: Path, out_dir: Path):
    gt = load_gt_by_image(split)
    rows = []
    for r in read_jsonl(run_jsonl):
        img_path = r.get("image") or r.get("img") or r.get("image_path") or None
        if img_path is None:
            # fallback: procurar string .jpg no json
            img_path = next((v for v in r.values() if isinstance(v,str) and v.lower().endswith((".jpg",".png",".jpeg",".webp"))), None)
        if not img_path:
            continue
        image_id = extract_image_id_from_path(img_path)
        gold = gt.get(image_id)
        if not gold:
            continue
        pred = normalize_pred_labels(r.get("pred_labels") or r.get("labels") or r.get("pred") or r.get("output"))
        m = row_metrics(pred, gold)
        rows.append({
            "image_id": image_id,
            "split": split,
            "condition": "baseline",
            "seconds": r.get("seconds"),
            "pred_labels": ",".join(sorted(pred)),
            "gold_labels": ",".join(sorted(gold)),
            **m
        })

    # best/worst
    perfect = [x for x in rows if x["fp"]==0 and x["fn"]==0]
    worst = sorted(rows, key=lambda x: (x["f1"], -x["fp"], x["fn"]))[:25]

    write_csv(out_dir/"metrics_per_image.csv", rows,
              ["image_id","split","condition","seconds","pred_labels","gold_labels",
               "tp","fp","fn","precision","recall","f1","jaccard","hamming_loss","overprediction","pred_size","gold_size"])
    write_csv(out_dir/"best_images.csv", perfect,
              ["image_id","f1","tp","fp","fn","pred_labels","gold_labels","seconds"])
    write_csv(out_dir/"worst_images.csv", worst,
              ["image_id","f1","tp","fp","fn","pred_labels","gold_labels","seconds"])

    summary = {
        "split": split,
        "condition": "baseline",
        "metric_summary_f1": summarize(rows, "f1"),
        "metric_summary_fp": summarize(rows, "fp"),
        "metric_summary_fn": summarize(rows, "fn"),
        "n_perfect": len(perfect),
    }
    (out_dir/"labels_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def eval_crops(split: str, condition: str, run_jsonl: Path, out_dir: Path):
    gt_img = load_gt_by_image(split)

    # per-crop
    crop_rows = []
    by_image_pred = defaultdict(set)
    by_image_sec = defaultdict(float)
    by_image_n = defaultdict(int)

    for r in read_jsonl(run_jsonl):
        crop_path = r.get("crop_image")
        if not crop_path:
            continue
        image_id = extract_image_id_from_crop_path(crop_path)
        gold_crop = set([Path(crop_path).stem])
        pred = normalize_pred_labels(r.get("pred_labels") or r.get("labels") or r.get("pred") or r.get("output"))
        m = row_metrics(pred, gold_crop)

        sec = r.get("seconds")
        try:
            sec_f = float(sec) if sec is not None else None
        except Exception:
            sec_f = None

        crop_rows.append({
            "image_id": image_id,
            "crop_path": crop_path,
            "split": split,
            "condition": condition,
            "seconds": sec_f,
            "pred_labels": ",".join(sorted(pred)),
            "gold_label": next(iter(gold_crop)),
            **m
        })

        by_image_pred[image_id] |= pred
        if sec_f is not None:
            by_image_sec[image_id] += sec_f
        by_image_n[image_id] += 1

    # per-image (union)
    img_rows = []
    for image_id, pred_union in by_image_pred.items():
        gold = gt_img.get(image_id)
        if not gold:
            continue
        m = row_metrics(pred_union, gold)
        img_rows.append({
            "image_id": image_id,
            "split": split,
            "condition": condition,
            "seconds_total": by_image_sec.get(image_id, None),
            "n_crops": by_image_n.get(image_id, 0),
            "pred_labels": ",".join(sorted(pred_union)),
            "gold_labels": ",".join(sorted(gold)),
            **m
        })

    perfect = [x for x in img_rows if x["fp"]==0 and x["fn"]==0]
    worst = sorted(img_rows, key=lambda x: (x["f1"], -x["fp"], x["fn"]))[:25]

    write_csv(out_dir/"metrics_per_crop.csv", crop_rows,
              ["image_id","crop_path","split","condition","seconds","pred_labels","gold_label",
               "tp","fp","fn","precision","recall","f1","jaccard","hamming_loss","overprediction","pred_size","gold_size"])
    write_csv(out_dir/"metrics_per_image.csv", img_rows,
              ["image_id","split","condition","seconds_total","n_crops","pred_labels","gold_labels",
               "tp","fp","fn","precision","recall","f1","jaccard","hamming_loss","overprediction","pred_size","gold_size"])
    write_csv(out_dir/"best_images.csv", perfect,
              ["image_id","f1","tp","fp","fn","pred_labels","gold_labels","seconds_total","n_crops"])
    write_csv(out_dir/"worst_images.csv", worst,
              ["image_id","f1","tp","fp","fn","pred_labels","gold_labels","seconds_total","n_crops"])

    summary = {
        "split": split,
        "condition": condition,
        "metric_summary_f1": summarize(img_rows, "f1"),
        "metric_summary_fp": summarize(img_rows, "fp"),
        "metric_summary_fn": summarize(img_rows, "fn"),
        "n_perfect": len(perfect),
    }
    (out_dir/"labels_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

def main():
    base = Path("runs/gemma3")

    tasks = [
        ("baseline","test_closed", base/"baseline/test_closed/raw/p1_baseline_test_closed.jsonl", base/"baseline/test_closed/eval"),
        ("baseline","test_open",   base/"baseline/test_open/raw/p1_baseline_test_open.jsonl",   base/"baseline/test_open/eval"),

        ("crops_black","test_closed", base/"crops_black/test_closed/raw/p1_crop_test_closed.jsonl", base/"crops_black/test_closed/eval"),
        ("crops_black","test_open",   base/"crops_black/test_open/raw/p1_crop_test_open.jsonl",   base/"crops_black/test_open/eval"),

        ("crops_white","test_closed", base/"crops_white/test_closed/raw/p1_crop_white_test_closed.jsonl", base/"crops_white/test_closed/eval"),
        ("crops_white","test_open",   base/"crops_white/test_open/raw/p1_crop_white_test_open.jsonl",   base/"crops_white/test_open/eval"),
    ]

    all_sum = []
    for condition, split, jsonl_path, out_dir in tasks:
        if not jsonl_path.exists():
            print("SKIP missing:", jsonl_path)
            continue
        if condition == "baseline":
            s = eval_baseline(split, jsonl_path, out_dir)
        else:
            s = eval_crops(split, condition, jsonl_path, out_dir)
        all_sum.append(s)

    (base/"_summary_labels_gemma.json").write_text(json.dumps(all_sum, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", base/"_summary_labels_gemma.json")

if __name__ == "__main__":
    main()
