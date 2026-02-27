#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def pick_one(df: pd.DataFrame):
    """Escolhe uma linha representativa (menor seconds)"""
    if df is None or df.empty:
        return None
    if "seconds" in df.columns:
        tmp = df.copy()
        tmp["seconds_num"] = pd.to_numeric(tmp["seconds"], errors="coerce")
        tmp = tmp.sort_values(["seconds_num"], ascending=True, na_position="last")
        row = tmp.iloc[0].to_dict()
        return row
    return df.iloc[0].to_dict()


def norm_str(x):
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def row_to_export(image_id, split, gt, model, variant, crop_label, crop_file, rowdict):
    if rowdict is None:
        return {
            "image_id": image_id,
            "split": split,
            "gt_labels_image": gt,
            "model": model,
            "variant": variant,
            "crop_label": crop_label,
            "crop_file": crop_file,
            "pred_labels": "",
            "fp_labels_not_in_gt": "",
            "tp_labels_in_gt": "",
            "n_pred": "",
            "n_tp": "",
            "n_fp": "",
            "seconds": "",
        }
    return {
        "image_id": image_id,
        "split": split,
        "gt_labels_image": gt,
        "model": model,
        "variant": variant,
        "crop_label": crop_label,
        "crop_file": crop_file,
        "pred_labels": norm_str(rowdict.get("pred_labels")),
        "fp_labels_not_in_gt": norm_str(rowdict.get("fp_labels_not_in_gt")),
        "tp_labels_in_gt": norm_str(rowdict.get("tp_labels_in_gt")),
        "n_pred": rowdict.get("n_pred", ""),
        "n_tp": rowdict.get("n_tp", ""),
        "n_fp": rowdict.get("n_fp", ""),
        "seconds": rowdict.get("seconds", ""),
    }


def make_markdown_for_image(image_id, split, gt, crop_label_file, models_block):
    lines = []
    lines.append(f"## image_id: `{image_id}` ({split})")
    lines.append(f"**GT (image-level):** `{gt}`")
    lines.append("")
    lines.append(f"**Crop label:** `{crop_label_file}`")
    lines.append("")
    lines.append("| Model | Baseline (pred / fp / n_fp) | Crop black (pred / fp / n_fp) | Crop white (pred / fp / n_fp) |")
    lines.append("|---|---|---|---|")

    for m, base, cb, cw in models_block:
        def cell(r):
            if r is None:
                return "—"
            pred = norm_str(r.get("pred_labels"))
            fp = norm_str(r.get("fp_labels_not_in_gt"))
            nfp = norm_str(r.get("n_fp"))
            return f"`{pred}`<br>fp:`{fp}`<br>n_fp:`{nfp}`"

        lines.append(f"| `{m}` | {cell(base)} | {cell(cb)} | {cell(cw)} |")

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, default="DL_all_vlms_baseline_black_white_tables.xlsx")
    ap.add_argument("--out_csv", type=str, default="panels_export.csv")
    ap.add_argument("--out_md", type=str, default="panels_export.md")

    ap.add_argument("--split", type=str, default="", help="test_open ou test_closed (opcional)")
    ap.add_argument("--label", type=str, default="eagle", help="label do crop (sem .png), ex: eagle")

    ap.add_argument("--image_ids", type=str, default="", help="lista separada por vírgula: id1,id2,id3")
    ap.add_argument("--examples", action="store_true", help="buscar N exemplos cujo GT contém a label")
    ap.add_argument("--n", type=int, default=3, help="quantidade de exemplos com --examples")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    if not xlsx.exists():
        raise SystemExit(f"XLSX não encontrado: {xlsx}")

    df = pd.read_excel(xlsx, sheet_name="per_crop")

    # normalizar tipos
    for c in ["split", "variant", "model", "image_id", "crop_file", "gt_labels_image"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    split = args.split.strip() or None
    label = args.label.strip().lower()
    crop_label_file = f"{label}.png"

    # escolher image_ids
    if args.examples:
        base = df[df["variant"] == "baseline"].copy()
        if split:
            base = base[base["split"] == split]
        hits = base[base["gt_labels_image"].str.lower().str.contains(rf"(^|;){label}(;|$)", regex=True)]
        image_ids = hits["image_id"].drop_duplicates().head(args.n).tolist()
        if not image_ids:
            raise SystemExit(f"Nenhuma imagem encontrada com label '{label}' no GT (split={split or 'qualquer'}).")
    else:
        if not args.image_ids.strip():
            raise SystemExit("Passe --image_ids id1,id2 OU use --examples.")
        image_ids = [x.strip() for x in args.image_ids.split(",") if x.strip()]

    export_rows = []
    md_parts = ["# VLM Panels Export", ""]

    # modelos disponíveis no arquivo
    models_all = sorted(df["model"].dropna().astype(str).unique())

    for image_id in image_ids:
        d_img = df[df["image_id"] == image_id].copy()
        if split:
            d_img = d_img[d_img["split"] == split]

        if d_img.empty:
            continue

        split_used = d_img.iloc[0]["split"]
        gt = d_img.iloc[0].get("gt_labels_image", "")

        models_block = []

        for m in models_all:
            dm = d_img[d_img["model"] == m]

            base = pick_one(dm[dm["variant"] == "baseline"])
            cb = pick_one(dm[(dm["variant"] == "crops_black") & (dm["crop_file"].str.lower() == crop_label_file)])
            cw = pick_one(dm[(dm["variant"] == "crops_white") & (dm["crop_file"].str.lower() == crop_label_file)])

            models_block.append((m, base, cb, cw))

            # CSV rows (um por variante)
            export_rows.append(row_to_export(image_id, split_used, gt, m, "baseline", label, "__full_image__", base))
            export_rows.append(row_to_export(image_id, split_used, gt, m, "crops_black", label, crop_label_file, cb))
            export_rows.append(row_to_export(image_id, split_used, gt, m, "crops_white", label, crop_label_file, cw))

        md_parts.append(make_markdown_for_image(image_id, split_used, gt, crop_label_file, models_block))

    # salvar
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    pd.DataFrame(export_rows).to_csv(out_csv, index=False)
    out_md.write_text("\n".join(md_parts), encoding="utf-8")

    print(f"[DONE] CSV: {out_csv.resolve()}")
    print(f"[DONE] MD : {out_md.resolve()}")


if __name__ == "__main__":
    main()
