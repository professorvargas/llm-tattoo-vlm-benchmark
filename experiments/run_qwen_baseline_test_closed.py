import json, time, statistics
from pathlib import Path

import torch
from transformers import pipeline

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE_ID = 0  # cuda:0
DTYPE = torch.bfloat16

VOCAB_PATH = Path("runs/_shared_links/vocab_35.txt")
IMG_DIR = Path("datasets/test_closed/images")

OUT_JSONL = Path("runs/qwen2_5_vl/baseline/test_closed/raw/p1_baseline_test_closed.jsonl")
OUT_LOGDIR = Path("runs/qwen2_5_vl/baseline/test_closed/logs")
OUT_EVALDIR = Path("runs/qwen2_5_vl/baseline/test_closed/eval")
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
OUT_LOGDIR.mkdir(parents=True, exist_ok=True)
OUT_EVALDIR.mkdir(parents=True, exist_ok=True)

# vocab
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
    # de-dup preserve order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

# pipeline (já funciona no seu ambiente)
pipe = pipeline(
    task="image-text-to-text",
    model=MODEL_ID,
    device=DEVICE_ID,
    dtype=DTYPE
)

imgs = sorted(IMG_DIR.glob("*.jpg"))
if not imgs:
    raise SystemExit(f"No .jpg found in {IMG_DIR}")

times = []
t_total0 = time.time()

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for i, img_path in enumerate(imgs, 1):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "path": str(img_path)},
                {"type": "text", "text": PROMPT},
            ],
        }]

        # timing preciso em GPU (sincroniza)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()
        out = pipe(text=messages, max_new_tokens=64)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.time() - t0
        times.append(dt)

        # pega texto do assistant
        assistant_text = out[0]["generated_text"][-1]["content"].strip()
        labels = normalize_labels(assistant_text)

        rec = {
            "image": img_path.name,
            "prompt_name": "P1_controlled_vocab",
            "pred_labels": labels,
            "raw_text": assistant_text,
            "seconds": round(dt, 6),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if i % 50 == 0:
            print(f"[{i}/{len(imgs)}] last_seconds={dt:.3f}")

t_total = time.time() - t_total0

# resumo de custo computacional (tempo)
mean_all = sum(times)/len(times)
median = statistics.median(times)
p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else None
mean_excl_first = (sum(times[1:])/(len(times)-1)) if len(times) > 1 else mean_all

summary = {
    "model_id": MODEL_ID,
    "split": "test_closed",
    "condition": "baseline",
    "n_images": len(imgs),
    "total_seconds": round(t_total, 3),
    "mean_seconds_per_image": round(mean_all, 6),
    "mean_seconds_excluding_first": round(mean_excl_first, 6),
    "median_seconds_per_image": round(median, 6),
    "p90_seconds_per_image": (round(p90, 6) if p90 is not None else None),
    "throughput_images_per_second": round(len(imgs)/t_total, 6),
}

(OUT_EVALDIR / "timing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", OUT_JSONL)
print("Saved:", OUT_EVALDIR / "timing_summary.json")
print("Summary:", summary)
