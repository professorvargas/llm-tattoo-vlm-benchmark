#!/usr/bin/env python3
import argparse, re
from pathlib import Path

RE_IMAGE = re.compile(r"^## image_id:\s*`([^`]+)`\s*\(([^)]+)\)\s*$")
RE_CROP  = re.compile(r"^\*\*Crop label:\*\*\s*`([^`]+)`\s*$")

def find_baseline(root: Path, split: str, image_id: str):
    d = root / "datasets" / split / "images"
    for ext in (".jpg",".jpeg",".png",".webp"):
        p = d / f"{image_id}{ext}"
        if p.exists(): return p
    # fallback
    hits = list(d.glob(f"{image_id}.*"))
    return hits[0] if hits else None

def find_mask(root: Path, split: str, image_id: str):
    d = root / "datasets" / split
    hits = list(d.rglob(f"{image_id}_mask.*"))
    return hits[0] if hits else None

def rel_from(outdir: Path, p: Path):
    return p.resolve().relative_to(outdir.resolve().anchor) if False else Path(
        str(p.resolve())
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    inp = Path(args.inp).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_md = outdir / "index.md"

    lines = inp.read_text(encoding="utf-8").splitlines()
    out = []

    cur_img = None
    cur_split = None

    for i, ln in enumerate(lines):
        m = RE_IMAGE.match(ln)
        if m:
            cur_img = m.group(1).strip()
            cur_split = m.group(2).strip()

        out.append(ln)

        mc = RE_CROP.match(ln)
        if mc and cur_img and cur_split:
            crop_label = mc.group(1).strip()

            base = find_baseline(root, cur_split, cur_img)
            mask = find_mask(root, cur_split, cur_img)

            cb = root / "datasets" / "crops_gt" / cur_split / cur_img / crop_label
            cw = root / "datasets" / "crops_gt_white" / cur_split / cur_img / crop_label

            # compute relative paths from outdir
            def rel(p):
                if not p or not p.exists(): return "(not_found)"
                return Path(
                    str(p.resolve().relative_to(root.parent.resolve()))
                    if False else str(Path.relpath(p, outdir))
                )

            # Path.relpath exists on os.path; emulate safely:
            import os
            def rel2(p):
                if not p or not p.exists(): return "(not_found)"
                return os.path.relpath(str(p), str(outdir))

            block = f"""
<!--IMAGES_BEGIN-->
**Images (baseline / crop black / crop white / mask)**

<table>
  <tr>
    <td><b>Baseline</b><br><img src="{rel2(base)}" width="240"></td>
    <td><b>Crop black</b><br><img src="{rel2(cb)}" width="240"></td>
    <td><b>Crop white</b><br><img src="{rel2(cw)}" width="240"></td>
    <td><b>Mask</b><br><img src="{rel2(mask)}" width="240"></td>
  </tr>
</table>
<!--IMAGES_END-->
""".rstrip("\n")
            out.append(block)

    out_md.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] Wrote: {out_md} (no assets folder, references datasets directly)")

if __name__ == "__main__":
    main()
