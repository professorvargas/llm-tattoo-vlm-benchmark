import json, time, statistics, os
from pathlib import Path

import torch
from transformers import pipeline

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE_ID = 0
DTYPE = torch.bfloat16

VOCAB_PATH = Path("runs/_shared_links/vocab_35.txt")
IN_JSONL = Path("runs/gemma3/crops_white/test_open/raw/p1_crop_white_test_open.jsonl")

OUT_JSONL = Path("runs/qwen2_5_vl/crops_white/test_open/raw/p1_crop_white_test_open.jsonl")
OUT_LOGDIR = Path("runs/qwen2_5_vl/crops_white/test_open/logs")
OUT_EVALDIR = Path("runs/qwen2_5_vl/crops_white/test_open/eval")
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
OUT_LOGDIR.mkdir(parents=True, exist_ok=True)
OUT_EVALDIR.mkdir(parents=True, exist_ok=True)

vocab = [x.strip() for x in VOCAB_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
vocab_set = set(vocab)
vocab_str = ", ".join(vocab)

PROMPT = (
    "Return ONLY a comma-separated list of labels from this vocabulary:\n"
    f"{vocab_str}\n"
    "Rules:\n"
    "- Use ONLY labels from the list (exact spelling)\n"
    "- If uncertain or none apply, return: unknown\n"
    "- No extra words, no punctuation other than commas\n"
)

def normalize_labels(text: str):
    parts = [t.strip() for t in text.strip().split(",") if t.strip()]
    parts = [p for p in parts if p in vocab_set]
    if not parts:
        return ["unknown"]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def extract_first_image_path(obj):
    img_ext = (".png",".jpg",".jpeg",".webp")
    found = []
    def walk(x):
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            if x.lower().endswith(img_ext):
                found.append(x)
    walk(obj)
    return found[0] if found else None

pipe = pipeline(
    task="image-text-to-text",
    model=MODEL_ID,
    device=DEVICE_ID,
    dtype=DTYPE
)

times = []
t_total0 = time.time()
n = 0

with IN_JSONL.open("r", encoding="utf-8") as fin, OUT_JSONL.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec_in = json.loads(line)
        crop_path = extract_first_image_path(rec_in)
        if not crop_path or not os.path.exists(crop_path):
            rec_out = {
                "crop_image": crop_path,
                "ok": False,
                "error": "crop_path_missing",
                "seconds": None,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "path": crop_path},
                {"type": "text", "text": PROMPT},
            ],
        }]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = pipe(text=messages, max_new_tokens=64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.time() - t0
        times.append(dt)
        n += 1

        assistant_text = out[0]["generated_text"][-1]["content"].strip()
        labels = normalize_labels(assistant_text)

        rec_out = {
            "crop_image": crop_path,
            "prompt_name": "P1_controlled_vocab",
            "pred_labels": labels,
            "raw_text": assistant_text,
            "seconds": round(dt, 6),
            "split": rec_in.get("split", "test_open"),
            "temperature": rec_in.get("temperature", None),
            "model_src": "Qwen",
        }
        fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        if n % 200 == 0:
            print(f"[{n}] last_seconds={dt:.3f}")

t_total = time.time() - t_total0

if times:
    mean_all = sum(times)/len(times)
    median = statistics.median(times)
    p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else None
    mean_excl_first = (sum(times[1:])/(len(times)-1)) if len(times) > 1 else mean_all
else:
    mean_all = median = p90 = mean_excl_first = None

summary = {
    "model_id": MODEL_ID,
    "split": "test_open",
    "condition": "crops_white",
    "n_crops_ok": len(times),
    "n_total_lines": sum(1 for _ in IN_JSONL.open("r", encoding="utf-8")),
    "total_seconds": round(t_total, 3),
    "mean_seconds_per_crop": (round(mean_all, 6) if mean_all is not None else None),
    "mean_seconds_excluding_first": (round(mean_excl_first, 6) if mean_excl_first is not None else None),
    "median_seconds_per_crop": (round(median, 6) if median is not None else None),
    "p90_seconds_per_crop": (round(p90, 6) if p90 is not None else None),
    "throughput_crops_per_second": (round(len(times)/t_total, 6) if times else None),
}

(OUT_EVALDIR / "timing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", OUT_JSONL)
print("Saved:", OUT_EVALDIR / "timing_summary.json")
print("Summary:", summary)
