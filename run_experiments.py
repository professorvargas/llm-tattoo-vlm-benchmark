import argparse
import base64
import json
import os
import time
from pathlib import Path
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


BASE_URL_DEFAULT = "http://localhost:11434"
MODEL_DEFAULT = "gemma3:12b"

# Vocabulário oficial (labels) do Brilhador / TSSD2023 (sem background)
BRILHADOR_LABELS = [
    "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond", "dog", "eagle",
    "fire", "fish", "flower", "fox", "gun", "heart", "key", "knife", "leaf", "lion",
    "mermaid", "octopus", "owl", "ribbon", "rope", "scorpion", "shark", "shield", "skull",
    "snake", "spider", "star", "tiger", "water", "wolf", "unknown"
]

PROMPTS = {
    "P0_free_description": (
        "Analyse the image and describe in detail what you see. "
        "Be factual and do not invent details."
    ),

    # Seu P1 atual (mantido para histórico / comparação qualitativa)
    "P1_controlled_vocab": (
        "You are analyzing a tattoo image for an academic experiment.\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with keys:\n"
        "  - has_tattoo: boolean\n"
        "  - style: one of [traditional, neo-traditional, realism, blackwork, tribal, watercolor, geometric, lettering, other]\n"
        "  - dominant_elements: list of up to 5 items from [text, face, animal, flower, skull, heart, symbol, abstract, landscape, other]\n"
        "  - open_set_unknown: boolean  (true if the tattoo content seems outside the vocabulary above)\n"
        "  - short_caption: string (max 20 words)\n"
        "Rules:\n"
        "- If uncertain, pick 'other' and set open_set_unknown=true.\n"
        "- Never include keys other than the 5 listed.\n"
    ),

    # NOVO P1: o que você precisa para comparar com os IDs do Brilhador
    "P1_brilhador_labels_only": (
        "You are analyzing a tattoo image for an academic experiment.\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with EXACT keys:\n"
        "  - labels: list of 0 to 10 items\n"
        "  - unknown: boolean\n"
        "\n"
        "The 'labels' MUST be chosen ONLY from this controlled vocabulary:\n"
        f"{BRILHADOR_LABELS}\n"
        "\n"
        "Rules:\n"
        "- Use only lowercase strings exactly matching the vocabulary items.\n"
        "- Do NOT output generic words like 'animal', 'face', 'abstract'.\n"
        "- If you are not confident about a specific label, do not guess; set unknown=true.\n"
        "- If the tattoo contains something outside the vocabulary, set unknown=true and you may include 'unknown' in labels.\n"
        "- Avoid hallucination: output only what is clearly present.\n"
        "- 'labels' should NOT include 'background'.\n"
    ),
}

def extract_json_object(text: str):
    """Extract a JSON object from model output (handles ```json fences and extra text)."""
    if not text:
        return None, False

    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    if t.startswith("{") and t.endswith("}"):
        return t, True

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        return m.group(0).strip(), True

    return None, False

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def guess_mime(image_path: Path) -> str:
    suf = image_path.suffix.lower()
    if suf == ".png":
        return "image/png"
    return "image/jpeg"

def run_one(llm: ChatOllama, image_path: Path, prompt_text: str) -> str:
    image_b64 = image_to_base64(image_path)
    mime = guess_mime(image_path)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": f"data:{mime};base64,{image_b64}"},
        ]
    )
    out = llm.invoke([msg]).content
    return out

def iter_images(folder: Path, recursive: bool = False):
    """Yield image files from a folder.

    - recursive=False: only immediate files (default for datasets/<split>/images)
    - recursive=True: traverse subfolders (needed for GT-crops in datasets/crops_gt/<split>/<image_id>/*.png)
    """
    exts = {".jpg", ".jpeg", ".png"}
    if recursive:
        files = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in exts]
        for p in sorted(files):
            yield p
    else:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE_URL_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--split", choices=["test_closed", "test_open", "both"], default="both")
    ap.add_argument("--prompt-id", choices=list(PROMPTS.keys()) + ["all"], default="all")
    ap.add_argument("--images-dir", default="", help="Optional override for images folder (e.g., datasets/crops_gt/test_open)")

    # IMPORTANTE:
    # --limit continua sendo "limite de registros" (imagem x prompt).
    # Se quiser limitar por imagens, use --limit-images (abaixo).
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (records = image x prompt)")
    ap.add_argument("--limit-images", type=int, default=0, help="0 = no limit (limits number of images per split)")
    ap.add_argument("--out", default="experiments/results.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    root = Path("datasets")

    def split_images_folder(split_name: str) -> Path:
        # Default behavior: datasets/<split>/images
        # Override (for GT-crops): pass --images-dir pointing directly to the folder to iterate.
        # NOTE: when using --images-dir, run one split at a time (e.g., --split test_open).
        if args.images_dir:
            return Path(args.images_dir)
        return root / split_name / "images"

    splits = []
    if args.split in ("test_closed", "both"):
        splits.append(("test_closed", split_images_folder("test_closed")))
    if args.split in ("test_open", "both"):
        splits.append(("test_open", split_images_folder("test_open")))

    for name, folder in splits:
        if not folder.exists():
            raise FileNotFoundError(f'Missing folder: {folder}')

    # If --images-dir points to GT-crops, images are inside subfolders (one folder per original image).
    # Auto-enable recursive scan in that case.
    recursive_scan = bool(args.images_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = ChatOllama(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    prompt_ids = list(PROMPTS.keys()) if args.prompt_id == "all" else [args.prompt_id]

    # qualquer prompt que começa com P1_ deve ser JSON
    JSON_PROMPTS = {pid for pid in PROMPTS.keys() if pid.startswith("P1_")}

    n_done = 0
    with out_path.open("a", encoding="utf-8") as f:
        for split_name, folder in splits:
            img_count = 0
            for img_path in iter_images(folder, recursive=recursive_scan):
                img_count += 1
                if args.limit_images and img_count > args.limit_images:
                    break

                for pid in prompt_ids:
                    t0 = time.time()
                    try:
                        txt = run_one(llm, img_path, PROMPTS[pid])
                        ok = True
                        err = ""
                    except Exception as e:
                        txt = ""
                        ok = False
                        err = repr(e)
                    dt = time.time() - t0

                    json_text = ""
                    json_ok = False
                    json_obj = None

                    if pid in JSON_PROMPTS:
                        jt, okj = extract_json_object(txt)
                        json_text = jt or ""
                        json_ok = bool(okj)
                        if json_ok:
                            try:
                                json_obj = json.loads(json_text)
                            except Exception:
                                json_ok = False
                                json_obj = None

                    rec = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "split": split_name,
                        "image": str(img_path),
                        "model": args.model,
                        "prompt_id": pid,
                        "temperature": args.temperature,
                        "ok": ok,
                        "seconds": round(dt, 3),
                        "error": err,
                        "output": txt,
                        "json_ok": json_ok,
                        "json_text": json_text,
                        "json_obj": json_obj,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()

                    n_done += 1
                    if n_done % 10 == 0:
                        print(f"[progress] records={n_done} last={split_name}/{img_path.name} prompt={pid} ok={ok} time={dt:.2f}s")

                    if args.limit and n_done >= args.limit:
                        print(f"[done] limit reached: {args.limit} records")
                        return

    print(f"[done] wrote: {out_path} records={n_done}")

if __name__ == "__main__":
    main()
