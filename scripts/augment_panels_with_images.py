#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path

RE_IMAGE = re.compile(r"^## image_id:\s*`([^`]+)`\s*\(([^)]+)\)\s*$")
RE_CROP  = re.compile(r"^\*\*Crop label:\*\*\s*`([^`]+)`\s*$")

def build_index_images(root: Path, split: str):
    """
    Index baseline images and masks for a split, if folders exist.
    - baseline: datasets/<split>/images/<image_id>.(jpg|png|...)
    - mask:     any file under datasets/<split> matching <image_id>_mask.(jpg|png|...)
    """
    base_dir = root / "datasets" / split / "images"
    split_root = root / "datasets" / split

    baseline = {}
    if base_dir.exists():
        for p in base_dir.iterdir():
            if p.is_file():
                baseline[p.stem] = p

    masks = {}
    if split_root.exists():
        for p in split_root.rglob("*_mask.*"):
            if p.is_file():
                stem = p.stem
                if stem.endswith("_mask"):
                    img_id = stem[:-5]  # remove "_mask"
                    masks[img_id] = p

    return baseline, masks

def safe_stem(name: str) -> str:
    # "eagle.png" -> "eagle"
    return Path(name).stem.replace(" ", "_")

def copy_asset(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True, help="Input panels_custom.md")
    ap.add_argument("--outdir", required=True, help="Output directory (will be created)")
    ap.add_argument("--root", default=".", help="Project root (default: .)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    inp = Path(args.inp).resolve()
    outdir = Path(args.outdir).resolve()
    out_md = outdir / "index.md"
    assets_dir = outdir / "assets"

    outdir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    lines = inp.read_text(encoding="utf-8").splitlines()

    # First pass: collect which splits appear, so we can index once per split.
    splits = set()
    for ln in lines:
        m = RE_IMAGE.match(ln)
        if m:
            splits.add(m.group(2).strip())

    idx_baseline = {}
    idx_masks = {}
    for sp in splits:
        b, ms = build_index_images(root, sp)
        idx_baseline[sp] = b
        idx_masks[sp] = ms

    current_image_id = None
    current_split = None

    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m_img = RE_IMAGE.match(ln)
        if m_img:
            current_image_id = m_img.group(1).strip()
            current_split = m_img.group(2).strip()

        out.append(ln)

        m_crop = RE_CROP.match(ln)
        if m_crop and current_image_id and current_split:
            # avoid duplicating if already inserted
            next_ln = lines[i+1] if i+1 < len(lines) else ""
            if "<!--IMAGES_BEGIN-->" not in next_ln:
                crop_label = m_crop.group(1).strip()
                crop_stem = safe_stem(crop_label)

                # Resolve baseline
                base_src = idx_baseline.get(current_split, {}).get(current_image_id)
                # Resolve mask
                mask_src = idx_masks.get(current_split, {}).get(current_image_id)

                # Resolve crops (black/white) by convention from your repo layout
                crop_black_src = root / "datasets" / "crops_gt" / current_split / current_image_id / crop_label
                crop_white_src = root / "datasets" / "crops_gt_white" / current_split / current_image_id / crop_label

                # Prepare destination paths (self-contained export)
                base_dst = None
                mask_dst = None
                cb_dst = None
                cw_dst = None

                img_folder = assets_dir / current_split / current_image_id
                if base_src and base_src.exists():
                    base_dst = img_folder / f"baseline{base_src.suffix.lower()}"
                    copy_asset(base_src, base_dst)

                if mask_src and mask_src.exists():
                    mask_dst = img_folder / f"mask{mask_src.suffix.lower()}"
                    copy_asset(mask_src, mask_dst)

                if crop_black_src.exists():
                    cb_dst = img_folder / f"crop_black_{crop_stem}{crop_black_src.suffix.lower()}"
                    copy_asset(crop_black_src, cb_dst)

                if crop_white_src.exists():
                    cw_dst = img_folder / f"crop_white_{crop_stem}{crop_white_src.suffix.lower()}"
                    copy_asset(crop_white_src, cw_dst)

                # Build HTML block (works inside Markdown and lets us control sizes)
                def rel(p: Path):
                    return p.relative_to(outdir).as_posix()

                base_ref = rel(base_dst) if base_dst else "(not_found)"
                cb_ref   = rel(cb_dst)   if cb_dst   else "(not_found)"
                cw_ref   = rel(cw_dst)   if cw_dst   else "(not_found)"
                mask_ref = rel(mask_dst) if mask_dst else "(not_found)"

                block = f"""
<!--IMAGES_BEGIN-->
**Images (baseline / crop black / crop white / mask)**

<table>
  <tr>
    <td><b>Baseline</b><br><img src="{base_ref}" width="240"></td>
    <td><b>Crop black</b><br><img src="{cb_ref}" width="240"></td>
    <td><b>Crop white</b><br><img src="{cw_ref}" width="240"></td>
    <td><b>Mask</b><br><img src="{mask_ref}" width="240"></td>
  </tr>
</table>
<!--IMAGES_END-->
""".rstrip("\n")
                out.append(block)

        i += 1

    out_md.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] Wrote: {out_md}")
    print(f"[OK] Assets: {assets_dir}")

if __name__ == "__main__":
    main()
