import json, math, statistics
from pathlib import Path

BASE = Path("runs/gemma3")

TASKS = [
    ("crops_black", "test_open",   "p1_crop_test_open.jsonl"),
    ("crops_black", "test_closed", "p1_crop_test_closed.jsonl"),
    ("crops_white", "test_open",   "p1_crop_white_test_open.jsonl"),
    ("crops_white", "test_closed", "p1_crop_white_test_closed.jsonl"),
]

SEC_KEYS = ["seconds", "sec", "elapsed", "elapsed_s", "latency_s", "time_s", "dt"]

def pctl(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    idx = max(0, min(len(vals)-1, int(math.ceil(q*len(vals))) - 1))
    return vals[idx]

def get_seconds(rec):
    # ignore registros explicitamente inválidos
    if rec.get("ok") is False:
        return None
    for k in SEC_KEYS:
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                pass
    return None

for cond, split, fname in TASKS:
    raw_path = BASE / cond / split / "raw" / fname
    eval_dir = BASE / cond / split / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print("SKIP (raw missing):", raw_path)
        continue

    secs = []
    n_total = 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)
            s = get_seconds(rec)
            if s is not None:
                secs.append(s)

    total = sum(secs) if secs else None
    mean_all = statistics.mean(secs) if secs else None
    median = statistics.median(secs) if secs else None
    p90 = pctl(secs, 0.90) if secs else None
    mean_excl_first = (statistics.mean(secs[1:]) if len(secs) > 1 else mean_all)

    summary = {
        "model_id": "gemma3",
        "split": split,
        "condition": cond,
        "n_crops_ok": len(secs),
        "n_total_lines": n_total,
        "total_seconds": round(total, 3) if total is not None else None,
        "mean_seconds_per_crop": round(mean_all, 6) if mean_all is not None else None,
        "mean_seconds_excluding_first": round(mean_excl_first, 6) if mean_excl_first is not None else None,
        "median_seconds_per_crop": round(median, 6) if median is not None else None,
        "p90_seconds_per_crop": round(p90, 6) if p90 is not None else None,
        "throughput_crops_per_second": round((len(secs)/total), 6) if total else None,
        "source_raw": str(raw_path),
    }

    out = eval_dir / "timing_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", out, "| secs:", len(secs), "lines:", n_total)

