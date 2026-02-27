#!/usr/bin/env python3
import json, time, statistics, os, re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

VOCAB_PATH = Path("runs/_shared_links/vocab_35.txt")

def read_vocab():
    vocab = [x.strip() for x in VOCAB_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    return vocab, set(vocab)

def build_prompt(vocab):
    vocab_str = ", ".join(vocab)
    return (
        "Return ONLY a comma-separated list of labels from this vocabulary:\n"
        f"{vocab_str}\n"
        "Rules:\n"
        "- Use ONLY labels from the list (exact spelling)\n"
        "- If uncertain or none apply, return: unknown\n"
        "- No extra words, no punctuation other than commas\n"
    )

def normalize_labels(text: str, vocab_set):
    parts = [t.strip() for t in re.split(r"[,;\n|]+", (text or "").strip()) if t.strip()]
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
    # mesmo padrão do Qwen: vasculha rec_in e pega o 1º caminho de imagem
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

class LlamaVLM:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=DTYPE,
        )
        self.model.eval()

    @torch.inference_mode()
    def infer_text(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        in_len = inputs["input_ids"].shape[-1]
        gen = out[:, in_len:]
        text = self.processor.decode(gen[0], skip_special_tokens=True)
        return (text or "").strip()

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_timing(out_evaldir: Path, summary: dict):
    out_evaldir.mkdir(parents=True, exist_ok=True)
    (out_evaldir / "timing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def run_baseline(vlm: LlamaVLM, split: str, out_base: Path, limit: int | None = None):
    vocab, vocab_set = read_vocab()
    prompt = build_prompt(vocab)

    img_dir = Path("datasets") / split / "images"
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"No .jpg found in {img_dir}")

    if limit:
        imgs = imgs[:limit]

    out_jsonl = out_base / "raw" / f"p1_baseline_{split}.jsonl"
    out_eval  = out_base / "eval"

    times = []
    records = []
    t_total0 = time.time()

    for i, img_path in enumerate(imgs, 1):
        t0 = time.time()
        assistant_text = vlm.infer_text(str(img_path), prompt, max_new_tokens=64)
        dt = time.time() - t0
        times.append(dt)

        labels = normalize_labels(assistant_text, vocab_set)

        records.append({
            "image": img_path.name,  # igual ao Qwen baseline
            "prompt_name": "P1_controlled_vocab",
            "pred_labels": labels,
            "raw_text": assistant_text,
            "seconds": round(dt, 6),
            "model_src": "LLaMA",
            "split": split,
            "condition": "baseline",
        })

        if i % 50 == 0:
            print(f"[baseline {split}] [{i}/{len(imgs)}] last_seconds={dt:.3f}")

    t_total = time.time() - t_total0

    mean_all = sum(times)/len(times) if times else None
    median = statistics.median(times) if times else None
    p90 = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else None
    mean_excl_first = (sum(times[1:])/(len(times)-1)) if len(times) > 1 else mean_all

    summary = {
        "model_id": MODEL_ID,
        "split": split,
        "condition": "baseline",
        "n_images": len(imgs),
        "total_seconds": round(t_total, 3),
        "mean_seconds_per_image": (round(mean_all, 6) if mean_all is not None else None),
        "mean_seconds_excluding_first": (round(mean_excl_first, 6) if mean_excl_first is not None else None),
        "median_seconds_per_image": (round(median, 6) if median is not None else None),
        "p90_seconds_per_image": (round(p90, 6) if p90 is not None else None),
        "throughput_images_per_second": (round(len(times)/t_total, 6) if times else None),
    }

    write_jsonl(out_jsonl, records)
    write_timing(out_eval, summary)
    print("Saved:", out_jsonl)
    print("Saved:", out_eval / "timing_summary.json")

def run_crops(vlm: LlamaVLM, split: str, condition: str, in_jsonl: Path, out_base: Path, out_name: str, limit: int | None = None):
    vocab, vocab_set = read_vocab()
    prompt = build_prompt(vocab)

    out_jsonl = out_base / "raw" / out_name
    out_eval  = out_base / "eval"
    out_log   = out_base / "logs"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_eval.mkdir(parents=True, exist_ok=True)
    out_log.mkdir(parents=True, exist_ok=True)

    times = []
    records = []
    t_total0 = time.time()
    n_total = 0
    n_ok = 0

    with in_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            if limit and n_ok >= limit:
                break

            rec_in = json.loads(line)
            crop_path = extract_first_image_path(rec_in)
            if not crop_path or not os.path.exists(crop_path):
                records.append({
                    "crop_image": crop_path,
                    "ok": False,
                    "error": "crop_path_missing",
                    "seconds": None,
                    "split": split,
                    "condition": condition,
                    "model_src": "LLaMA",
                })
                continue

            t0 = time.time()
            assistant_text = vlm.infer_text(crop_path, prompt, max_new_tokens=64)
            dt = time.time() - t0
            times.append(dt)
            n_ok += 1

            labels = normalize_labels(assistant_text, vocab_set)

            records.append({
                "crop_image": crop_path,
                "prompt_name": "P1_controlled_vocab",
                "pred_labels": labels,
                "raw_text": assistant_text,
                "seconds": round(dt, 6),
                "split": rec_in.get("split", split),
                "temperature": rec_in.get("temperature", None),
                "condition": condition,
                "model_src": "LLaMA",
            })

            if n_ok % 200 == 0:
                print(f"[{condition} {split}] [{n_ok}] last_seconds={dt:.3f}")

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
        "split": split,
        "condition": condition,
        "n_crops_ok": len(times),
        "n_total_lines": n_total,
        "total_seconds": round(t_total, 3),
        "mean_seconds_per_crop": (round(mean_all, 6) if mean_all is not None else None),
        "mean_seconds_excluding_first": (round(mean_excl_first, 6) if mean_excl_first is not None else None),
        "median_seconds_per_crop": (round(median, 6) if median is not None else None),
        "p90_seconds_per_crop": (round(p90, 6) if p90 is not None else None),
        "throughput_crops_per_second": (round(len(times)/t_total, 6) if times else None),
    }

    write_jsonl(out_jsonl, records)
    write_timing(out_eval, summary)
    print("Saved:", out_jsonl)
    print("Saved:", out_eval / "timing_summary.json")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_images", type=int, default=0, help="0 = sem limite (baseline)")
    ap.add_argument("--limit_crops", type=int, default=0, help="0 = sem limite (crops)")
    ap.add_argument("--only", default="", help="ex: baseline:test_open,crops_black:test_open")
    args = ap.parse_args()

    base = Path("runs/llama3_2_vision")
    vlm = LlamaVLM(MODEL_ID)

    tasks = [
        ("baseline","test_closed", None, base/"baseline/test_closed", f"p1_baseline_test_closed.jsonl"),
        ("baseline","test_open",   None, base/"baseline/test_open",   f"p1_baseline_test_open.jsonl"),

        ("crops_black","test_closed", Path("runs/gemma3/crops_black/test_closed/raw/p1_crop_test_closed.jsonl"),
         base/"crops_black/test_closed", "p1_crop_test_closed.jsonl"),
        ("crops_black","test_open",   Path("runs/gemma3/crops_black/test_open/raw/p1_crop_test_open.jsonl"),
         base/"crops_black/test_open",   "p1_crop_test_open.jsonl"),

        ("crops_white","test_closed", Path("runs/gemma3/crops_white/test_closed/raw/p1_crop_white_test_closed.jsonl"),
         base/"crops_white/test_closed", "p1_crop_white_test_closed.jsonl"),
        ("crops_white","test_open",   Path("runs/gemma3/crops_white/test_open/raw/p1_crop_white_test_open.jsonl"),
         base/"crops_white/test_open",   "p1_crop_white_test_open.jsonl"),
    ]

    only = set([x.strip() for x in args.only.split(",") if x.strip()]) if args.only else None

    for condition, split, in_jsonl, out_base, out_name in tasks:
        key = f"{condition}:{split}"
        if only and key not in only:
            continue

        if condition == "baseline":
            run_baseline(vlm, split, out_base, limit=(args.limit_images or None))
        else:
            if not in_jsonl.exists():
                print("SKIP missing:", in_jsonl)
                continue
            run_crops(vlm, split, condition, in_jsonl, out_base, out_name, limit=(args.limit_crops or None))

if __name__ == "__main__":
    main()
