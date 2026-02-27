#!/usr/bin/env python3
import argparse, re, os
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

RE_HDR  = re.compile(r"^## image_id:\s*`([^`]+)`\s*\(([^)]+)\)\s*$")
RE_GT   = re.compile(r"^\*\*GT \(image-level\):\*\*\s*`([^`]+)`\s*$")
RE_CROP = re.compile(r"^\*\*Crop label:\*\*\s*`([^`]+)`\s*$")
RE_IMG  = re.compile(r'<img\s+src="([^"]+)"')
RE_ROW  = re.compile(r'^\|\s*`([^`]+)`\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$')

def split_labels(s: str):
    s = (s or "").strip().lower()
    if not s or s in {"—","-","none","null"}:
        return []
    parts = re.split(r"[;,]\s*|,\s*|\s+", s)
    return [p for p in (p.strip() for p in parts) if p]

def classify(target: str, pred: str, fp: str):
    target = (target or "").strip().lower()
    pred_set = set(split_labels(pred))
    fp_set   = set(split_labels(fp))
    fp_no_unk = {x for x in fp_set if x != "unknown"}
    has_target = target in pred_set
    has_unknown = ("unknown" in pred_set) or ("unknown" in fp_set)

    if has_target:
        if not fp_no_unk and not has_unknown:
            return "Perfect"
        if not fp_no_unk and has_unknown:
            return "Perfect+Unknown"
        if fp_no_unk and has_unknown:
            return "Hallucination+Unknown"
        return "Hallucination"
    else:
        if pred_set == {"unknown"}:
            return "Unknown"
        if "unknown" in pred_set:
            return "Unknown+Other"
        if pred_set:
            return "Hallucination"
        return "Miss"

def parse_cell(cell: str):
    parts = re.findall(r"`([^`]*)`", cell)
    pred = parts[0].strip() if len(parts) > 0 else ""
    fp   = parts[1].strip() if len(parts) > 1 else ""
    nfp  = parts[2].strip() if len(parts) > 2 else ""
    return pred, fp, nfp

def load_font(size=18, mono=False):
    candidates = []
    if mono:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def open_rgb(p: Path):
    if not p or not p.exists():
        return None
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def paste_fit(canvas, img, box):
    if img is None:
        return
    x0,y0,x1,y1 = box
    w = x1-x0
    h = y1-y0
    iw, ih = img.size
    scale = min(w/iw, h/ih)
    nw, nh = int(iw*scale), int(ih*scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    px = x0 + (w-nw)//2
    py = y0 + (h-nh)//2
    canvas.paste(resized, (px, py))

def page_canvas(dpi=150):
    W = int(11.69 * dpi)  # A4 landscape
    H = int(8.27  * dpi)
    return Image.new("RGB", (W, H), (255,255,255))

def draw_table(draw, x, y, w, header, rows, font, font_h, pad=6):
    # colunas: Model | Baseline | Black | White
    col_w = [int(0.16*w), int(0.28*w), int(0.28*w), int(0.28*w)]
    # header
    hh = font_h + 2*pad
    draw.rectangle([x, y, x+w, y+hh], outline=(120,120,120), fill=(235,235,235))
    cx = x
    for j, text in enumerate(header):
        draw.text((cx+pad, y+pad), text, fill=(0,0,0), font=font)
        cx += col_w[j]
        draw.line([cx, y, cx, y+hh], fill=(180,180,180), width=1)

    row_y = y + hh + 8
    line_h = font_h + 3
    row_h = line_h * 6 + 8

    for r in rows:
        draw.rectangle([x, row_y, x+w, row_y+row_h], outline=(210,210,210))
        cx = x
        for j, cell in enumerate(r):
            tx = cx+pad
            ty = row_y+pad
            for ln in cell.split("\n"):
                draw.text((tx, ty), ln[:110], fill=(0,0,0), font=font)
                ty += line_h
            cx += col_w[j]
            draw.line([cx, row_y, cx, row_y+row_h], fill=(230,230,230), width=1)
        row_y += row_h + 6

def derive_paths(root: Path, split: str, image_id: str, crop_label: str):
    # fallback quando o MD não tem <img src=...>
    base = None
    imgdir = root / "datasets" / split / "images"
    for ext in (".jpg",".jpeg",".png",".webp"):
        p = imgdir / f"{image_id}{ext}"
        if p.exists():
            base = p; break
    if base is None:
        hits = list(imgdir.glob(f"{image_id}.*"))
        base = hits[0] if hits else None

    mask = root / "datasets" / split / "mask_rgb" / f"{image_id}_mask.jpg"
    if not mask.exists():
        hits = list((root / "datasets" / split).rglob(f"{image_id}_mask.*"))
        mask = hits[0] if hits else None

    cb = root / "datasets" / "crops_gt" / split / image_id / crop_label
    cw = root / "datasets" / "crops_gt_white" / split / image_id / crop_label
    return base, cb, cw, mask

def parse_panels(md_path: Path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    panels = []
    cur = None
    for ln in lines:
        m = RE_HDR.match(ln)
        if m:
            if cur: panels.append(cur)
            cur = {"image_id": m.group(1), "split": m.group(2), "gt":"", "crop":"", "imgs":[], "rows":[]}
            continue
        if not cur:
            continue
        mg = RE_GT.match(ln)
        if mg:
            cur["gt"] = mg.group(1); continue
        mc = RE_CROP.match(ln)
        if mc:
            cur["crop"] = mc.group(1); continue
        mi = RE_IMG.search(ln)
        if mi:
            cur["imgs"].append(mi.group(1)); continue
        mr = RE_ROW.match(ln)
        if mr:
            cur["rows"].append((mr.group(1), mr.group(2), mr.group(3), mr.group(4)))
            continue
    if cur: panels.append(cur)
    return panels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--root", default=".", help="project root (para fallback de paths)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--limit_images", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    md_path = Path(args.md).resolve()
    md_dir = md_path.parent.resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    panels = parse_panels(md_path)

    # agrupa por (split, image_id)
    by_img = defaultdict(list)
    for p in panels:
        by_img[(p["split"], p["image_id"])].append(p)

    keys = sorted(by_img.keys())
    if args.limit_images and args.limit_images > 0:
        keys = keys[:args.limit_images]

    fontT = load_font(22, mono=False)
    fontS = load_font(14, mono=False)
    fontM = load_font(12, mono=True)

    for (split, image_id) in keys:
        crops = sorted(by_img[(split, image_id)], key=lambda x: x["crop"])
        gt_lbl = crops[0]["gt"] if crops else ""

        pages = []

        # --- summary page (baseline + mask) ---
        page = page_canvas(args.dpi)
        draw = ImageDraw.Draw(page)
        W,H = page.size
        draw.text((30, 15), f"{image_id} | {split}", fill=(0,0,0), font=fontT)
        draw.text((30, 45), f"GT (image-level): {gt_lbl}", fill=(0,0,0), font=fontS)
        draw.text((30, 68), f"#crops: {len(crops)}", fill=(0,0,0), font=fontS)

        # baseline/mask: tenta pegar do <img src> se existir, senão deriva do dataset
        base_path = mask_path = None
        if crops and crops[0]["imgs"]:
            # ordem típica: baseline, crop_black, crop_white, mask
            imgs = crops[0]["imgs"]
            if len(imgs) >= 1: base_path = (md_dir / imgs[0]).resolve()
            if len(imgs) >= 4: mask_path = (md_dir / imgs[3]).resolve()

        if not base_path or not base_path.exists() or not mask_path or not mask_path.exists():
            b, _, _, m = derive_paths(root, split, image_id, crops[0]["crop"] if crops else "unknown.png")
            base_path = b if b else base_path
            mask_path = m if m else mask_path

        base_img = open_rgb(base_path) if base_path else None
        mask_img = open_rgb(mask_path) if mask_path else None

        margin = 30; gap = 20; top = 110
        box_h = H - top - margin
        box_w = (W - 2*margin - gap)//2
        draw.text((margin, top-18), "Baseline", fill=(0,0,0), font=fontS)
        draw.text((margin+box_w+gap, top-18), "Mask", fill=(0,0,0), font=fontS)
        paste_fit(page, base_img, (margin, top, margin+box_w, top+box_h))
        paste_fit(page, mask_img, (margin+box_w+gap, top, margin+2*box_w+gap, top+box_h))

        pages.append(page)

        # --- one page per crop ---
        for p in crops:
            crop_label = p["crop"]
            target = Path(crop_label).stem

            # paths: prefer imgs[] if exists, else derive
            cb_path = cw_path = None
            if p["imgs"] and len(p["imgs"]) >= 3:
                cb_path = (md_dir / p["imgs"][1]).resolve()
                cw_path = (md_dir / p["imgs"][2]).resolve()
            if not cb_path or not cb_path.exists() or not cw_path or not cw_path.exists():
                _, cb, cw, _ = derive_paths(root, split, image_id, crop_label)
                cb_path = cb if cb else cb_path
                cw_path = cw if cw else cw_path

            cb_img = open_rgb(cb_path) if cb_path else None
            cw_img = open_rgb(cw_path) if cw_path else None

            page = page_canvas(args.dpi)
            draw = ImageDraw.Draw(page)
            W,H = page.size

            draw.text((30, 12), f"{image_id} | {split} | crop: {crop_label} (target={target})", fill=(0,0,0), font=fontT)
            draw.text((30, 42), f"GT (image-level): {p['gt']}", fill=(0,0,0), font=fontS)

            # images
            margin = 30; gap = 20; top = 75
            img_h = int(0.38*H)
            img_w = (W - 2*margin - gap)//2
            draw.text((margin, top-16), "Crop black", fill=(0,0,0), font=fontS)
            draw.text((margin+img_w+gap, top-16), "Crop white", fill=(0,0,0), font=fontS)
            paste_fit(page, cb_img, (margin, top, margin+img_w, top+img_h))
            paste_fit(page, cw_img, (margin+img_w+gap, top, margin+2*img_w+gap, top+img_h))

            # table
            header = ["Model", "Baseline", "Black", "White"]
            rows = []
            for (model, bcell, kcell, wcell) in p["rows"]:
                bpred,bfp,bn = parse_cell(bcell)
                kpred,kfp,kn = parse_cell(kcell)
                wpred,wfp,wn = parse_cell(wcell)

                def pack(pred, fp, nfp):
                    fp_out = fp if fp else "—"
                    cls = classify(target, pred, fp)
                    return f"cls:{cls}\npred:{pred[:70]}\nfp:{fp_out[:70]}\nn_fp:{nfp}"

                rows.append([model, pack(bpred,bfp,bn), pack(kpred,kfp,kn), pack(wpred,wfp,wn)])

            tbl_top = top + img_h + 25
            draw_table(draw, margin, tbl_top, W-2*margin, header, rows, fontM, font_h=12, pad=6)

            pages.append(page)

        pdf_path = outdir / split / f"{image_id}.pdf"
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pages[0].save(pdf_path, "PDF", resolution=args.dpi, save_all=True, append_images=pages[1:])
        print(f"[OK] {pdf_path}  (pages={len(pages)})")

if __name__ == "__main__":
    main()
