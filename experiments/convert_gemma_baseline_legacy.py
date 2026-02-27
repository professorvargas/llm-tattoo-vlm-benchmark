import json, math, statistics
from pathlib import Path

# fontes legadas (já existem dentro do baseline do gemma3)
src_open   = Path("runs/gemma3/baseline/test_open/raw/full_results_v2_brilhador_labels.jsonl")
src_closed = Path("runs/gemma3/baseline/test_closed/raw/full_results_v2_brilhador_labels.jsonl")

assert src_open.exists(),   f"Não achei: {src_open}"
assert src_closed.exists(), f"Não achei: {src_closed}"

out_open   = Path("runs/gemma3/baseline/test_open/raw/p1_baseline_test_open.jsonl")
out_closed = Path("runs/gemma3/baseline/test_closed/raw/p1_baseline_test_closed.jsonl")

out_open.parent.mkdir(parents=True, exist_ok=True)
out_closed.parent.mkdir(parents=True, exist_ok=True)

def pick_labels(r):
    # já no formato novo?
    if isinstance(r.get("pred_labels"), list):
        return r["pred_labels"] or ["unknown"]

    # formato legado: json_obj.labels
    jo = r.get("json_obj")
    if isinstance(jo, dict) and isinstance(jo.get("labels"), list):
        labs = [str(x).strip() for x in jo["labels"] if str(x).strip()]
        return labs or ["unknown"]

    # fallback
    return ["unknown"]

def pick_image(r):
    return r.get("image") or r.get("img") or r.get("image_name") or r.get("filename")

def pick_seconds(r):
    s = r.get("seconds")
    try:
        return float(s) if s is not None else None
    except Exception:
        return None

def convert(src: Path, split: str):
    recs, secs = [], []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            img = pick_image(r)
            labs = pick_labels(r)
            sec = pick_seconds(r)
            if sec is not None:
                secs.append(sec)

            recs.append({
                "image": img,
                "pred_labels": labs,
                "seconds": sec,
                "split": split,
                "condition": "baseline",
                "model_src": "Gemma3",
            })
    return recs, secs

def write_jsonl(path: Path, recs):
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pctl(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    idx = max(0, min(len(vals)-1, int(math.ceil(q*len(vals))) - 1))
    return vals[idx]

def timing_summary(split, secs, n):
    total = sum(secs) if secs else None
    return {
        "model_id": "gemma3",
        "split": split,
        "condition": "baseline",
        "n_images": n,
        "total_seconds": round(total, 3) if total is not None else None,
        "mean_seconds_per_image": round(statistics.mean(secs), 6) if secs else None,
        "median_seconds_per_image": round(statistics.median(secs), 6) if secs else None,
        "p90_seconds_per_image": round(pctl(secs, 0.90), 6) if secs else None,
        "throughput_images_per_second": round(n/total, 6) if total else None,
    }

open_recs, open_secs = convert(src_open, "test_open")
closed_recs, closed_secs = convert(src_closed, "test_closed")

write_jsonl(out_open, open_recs)
write_jsonl(out_closed, closed_recs)

Path("runs/gemma3/baseline/test_open/eval/timing_summary.json").write_text(
    json.dumps(timing_summary("test_open", open_secs, len(open_recs)), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Path("runs/gemma3/baseline/test_closed/eval/timing_summary.json").write_text(
    json.dumps(timing_summary("test_closed", closed_secs, len(closed_recs)), ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("OK")
print(" open :", out_open,   "linhas:", len(open_recs))
print(" closed:", out_closed, "linhas:", len(closed_recs))
