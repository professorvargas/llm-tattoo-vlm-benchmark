#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_SPLIT_RE = re.compile(r"[;,|/]\s*")


def split_labels(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            out.extend(split_labels(v))
        return out

    s = str(x).strip()
    if not s:
        return []
    s = re.sub(r'[\[\]\(\)\{\}"\']', "", s)
    parts = _SPLIT_RE.split(s)

    labels = []
    for p in parts:
        p = p.strip()
        if p:
            labels.append(p)

    seen = set()
    norm = []
    for l in labels:
        ll = l.strip().lower()
        if not ll or ll in {"none", "null", "nan"}:
            continue
        if ll not in seen:
            seen.add(ll)
            norm.append(ll)
    return norm


def join_labels(labels: List[str]) -> str:
    if not labels:
        return ""
    return ";".join(sorted(set(labels)))


def pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = lower_map.get(cand.lower())
        if c:
            return c
    return None


def build_gt_map(gt_dir: Path) -> Dict[Tuple[str, str], List[str]]:
    gt_map: Dict[Tuple[str, str], List[str]] = {}
    for gt_root_name in ["crops_gt_black", "crops_gt_white", "crops_gt"]:
        root = gt_dir / gt_root_name
        if not root.exists():
            continue
        for split in ["test_open", "test_closed"]:
            split_dir = root / split
            if not split_dir.exists():
                continue
            for image_dir in split_dir.iterdir():
                if not image_dir.is_dir():
                    continue
                image_id = image_dir.name
                labels = []
                for p in image_dir.glob("*.png"):
                    lab = p.stem.strip().lower()
                    if lab:
                        labels.append(lab)
                key = (split, image_id)
                prev = set(gt_map.get(key, []))
                gt_map[key] = sorted(prev.union(set(labels)))
    return gt_map


def try_read_run_info(meta_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    run_info = meta_dir / "run_info.txt"
    if not run_info.exists():
        return out
    try:
        for line in run_info.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _find_first_existing_csv(search_root: Path, filenames: List[str]) -> Optional[Path]:
    for name in filenames:
        direct = search_root / name
        if direct.exists():
            return direct
    for name in filenames:
        hits = sorted(search_root.rglob(name))
        if hits:
            return hits[0]
    return None


def load_crops_eval(eval_dir: Path) -> pd.DataFrame:
    search_root = eval_dir if eval_dir.exists() else eval_dir.parent
    candidates = ["metrics_per_crop.csv", "metrics_crop_full.csv", "pred_crop_labels_per_image.csv"]

    paths: List[Path] = []
    for name in candidates:
        p = search_root / name
        if p.exists():
            paths.append(p)
    for name in candidates:
        for p in sorted(search_root.rglob(name)):
            if p not in paths:
                paths.append(p)

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if len(df) > 0:
            df.attrs["__source_path__"] = str(p)
            return df

    for p in paths:
        try:
            df = pd.read_csv(p)
            df.attrs["__source_path__"] = str(p)
            return df
        except Exception:
            continue

    return pd.DataFrame()


def load_baseline_eval(eval_dir: Path) -> pd.DataFrame:
    search_root = eval_dir if eval_dir.exists() else eval_dir.parent
    f = _find_first_existing_csv(search_root, ["metrics_per_image.csv"])
    if f:
        df = pd.read_csv(f)
        df.attrs["__source_path__"] = str(f)
        return df
    return pd.DataFrame()


# -------------------------
# Gemma raw parsing (corrigido para o seu formato de JSONL)
# -------------------------
def _normalize_split_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if s in {"test_open", "open", "test-open", "testopen"}:
        return "test_open"
    if s in {"test_closed", "closed", "test-closed", "testclosed"}:
        return "test_closed"
    return str(v).strip()


def _infer_image_id_from_path(crop_path: str) -> str:
    p = Path(crop_path)
    parts = list(p.parts)
    for split_name in ("test_open", "test_closed"):
        if split_name in parts:
            i = parts.index(split_name)
            if i + 1 < len(parts):
                return parts[i + 1]
    # fallback
    parent = p.parent.name
    return parent if parent else p.stem


def _extract_crop_path(obj: dict) -> Optional[str]:
    # seu jsonl usa "image" para o path do crop
    for k in ["crop_image", "crop_path", "image", "image_path", "file", "path", "crop"]:
        v = obj.get(k)
        if v:
            return str(v)
    return None


def _extract_pred_labels(obj: dict) -> List[str]:
    # 1) direto
    if obj.get("pred_labels") is not None:
        return split_labels(obj.get("pred_labels"))

    # 2) json_obj (seu jsonl tem json_obj={"labels":[...], ...})
    jo = obj.get("json_obj")
    if isinstance(jo, dict):
        if jo.get("labels") is not None:
            return split_labels(jo.get("labels"))
        if jo.get("pred_labels") is not None:
            return split_labels(jo.get("pred_labels"))

    # 3) output com bloco ```json ... ```
    out = obj.get("output")
    if isinstance(out, str) and out:
        m = re.search(r"```json\s*(\{.*?\})\s*```", out, flags=re.DOTALL | re.IGNORECASE)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if isinstance(parsed, dict) and parsed.get("labels") is not None:
                    return split_labels(parsed.get("labels"))
            except Exception:
                pass

    return []


def _find_jsonl_files(split_base: Path) -> List[Path]:
    # tenta raw/ primeiro, mas se estiver vazio procura recursivamente
    raw_dir = split_base / "raw"
    files = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    if files:
        return files
    return sorted(split_base.rglob("*.jsonl"))


def build_gemma_crops_df_from_raw(split_base: Path, split_expected: str, variant: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    condition = "black" if "black" in variant else ("white" if "white" in variant else "")

    jsonl_files = _find_jsonl_files(split_base)
    if not jsonl_files:
        return pd.DataFrame()

    for jf in jsonl_files:
        with jf.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                crop_path = _extract_crop_path(obj)
                if not crop_path:
                    continue

                line_split = _normalize_split_value(obj.get("split"))
                if not line_split:
                    s = crop_path.lower()
                    if "test_open" in s:
                        line_split = "test_open"
                    elif "test_closed" in s:
                        line_split = "test_closed"
                    else:
                        line_split = split_expected

                if line_split != split_expected:
                    continue

                labels = _extract_pred_labels(obj)
                if not labels:
                    # ainda inclui a linha (pra você enxergar), mas pred_labels vazio
                    pred = ""
                else:
                    pred = join_labels(labels)

                image_id = _infer_image_id_from_path(crop_path)

                rows.append({
                    "image_id": str(image_id),
                    "crop_path": str(crop_path),
                    "split": split_expected,
                    "condition": condition,
                    "seconds": obj.get("seconds", None),
                    "pred_labels": pred,
                })

    return pd.DataFrame(rows)


# -------------------------
# Standardização
# -------------------------
def standardize_crops_df(
    df: pd.DataFrame,
    split: str,
    model: str,
    variant: str,
    prompt_id_default: str,
    gt_map: Dict[Tuple[str, str], List[str]],
) -> pd.DataFrame:
    if df.empty:
        return df

    c_image = pick_col(df, "image_id", "image")
    c_crop = pick_col(df, "crop_file", "crop_path", "crop_image", "image", "crop", "crop_name", "crop_filename")
    c_pred = pick_col(df, "pred_labels", "predicted_labels", "pred_labels_crop", "pred_labels_image")
    c_gtimg = pick_col(df, "gt_labels_image", "gt_labels", "gt_image_labels", "gold_label")
    c_sec = pick_col(df, "seconds", "sec", "time_s", "elapsed_s", "elapsed_seconds")
    c_prompt = pick_col(df, "prompt_id", "prompt")

    if c_image is None:
        raise RuntimeError(f"Não encontrei coluna image_id em {model}/{variant}/{split}.")

    if c_crop is None:
        df = df.copy()
        df["crop_path"] = ""
        c_crop = "crop_path"

    out = pd.DataFrame(index=df.index)
    out["image_id"] = df[c_image].astype(str)
    out["split"] = split
    out["model"] = model
    out["variant"] = variant

    crop_src = df[c_crop].astype(str)
    crop_file = crop_src.apply(lambda s: Path(s).name if s and s != "nan" else "")
    crop_file = crop_file.apply(lambda s: s if (not s or s.lower().endswith(".png")) else (s + ".png"))
    out["crop_file"] = crop_file

    out["crop_gt_label_from_filename"] = out["crop_file"].apply(lambda s: Path(s).stem.lower() if s else "")
    out["pred_labels"] = df[c_pred].astype(str) if c_pred is not None else ""

    if c_gtimg is not None:
        out["gt_labels_image"] = df[c_gtimg].astype(str)
    else:
        out["gt_labels_image"] = out.apply(lambda r: join_labels(gt_map.get((split, r["image_id"]), [])), axis=1)

    out["seconds"] = pd.to_numeric(df[c_sec], errors="coerce") if c_sec is not None else pd.NA
    out["prompt_id"] = df[c_prompt].astype(str) if c_prompt is not None else str(prompt_id_default)

    pred_list = out["pred_labels"].apply(split_labels)
    gt_list = out["gt_labels_image"].apply(split_labels)

    tp, fp, n_pred, n_tp, n_fp = [], [], [], [], []
    for p, g in zip(pred_list, gt_list):
        ps = set(p)
        gs = set(g)
        tp_set = ps.intersection(gs)
        fp_set = ps.difference(gs)
        tp.append(join_labels(list(tp_set)))
        fp.append(join_labels(list(fp_set)))
        n_pred.append(len(ps))
        n_tp.append(len(tp_set))
        n_fp.append(len(fp_set))

    out["tp_labels_in_gt"] = tp
    out["fp_labels_not_in_gt"] = fp
    out["n_pred"] = n_pred
    out["n_tp"] = n_tp
    out["n_fp"] = n_fp

    return out


def standardize_baseline_df(
    df: pd.DataFrame,
    split: str,
    model: str,
    prompt_id_default: str,
    gt_map: Dict[Tuple[str, str], List[str]],
) -> pd.DataFrame:
    if df.empty:
        return df

    c_image = pick_col(df, "image_id", "image")
    c_pred = pick_col(df, "pred_labels", "predicted_labels", "pred_labels_image")
    c_gtimg = pick_col(df, "gt_labels_image", "gt_labels", "gt_image_labels")
    c_sec = pick_col(df, "seconds", "sec", "time_s", "elapsed_s", "elapsed_seconds")
    c_prompt = pick_col(df, "prompt_id", "prompt")

    if c_image is None:
        raise RuntimeError(f"Não encontrei coluna image_id no baseline de {model}/{split}.")

    out = pd.DataFrame(index=df.index)
    out["image_id"] = df[c_image].astype(str)
    out["split"] = split
    out["model"] = model
    out["variant"] = "baseline"
    out["crop_file"] = "__full_image__"
    out["crop_gt_label_from_filename"] = ""
    out["pred_labels"] = df[c_pred].astype(str) if c_pred is not None else ""

    if c_gtimg is not None:
        out["gt_labels_image"] = df[c_gtimg].astype(str)
    else:
        out["gt_labels_image"] = out.apply(lambda r: join_labels(gt_map.get((split, r["image_id"]), [])), axis=1)

    out["seconds"] = pd.to_numeric(df[c_sec], errors="coerce") if c_sec is not None else pd.NA
    out["prompt_id"] = df[c_prompt].astype(str) if c_prompt is not None else str(prompt_id_default)

    pred_list = out["pred_labels"].apply(split_labels)
    gt_list = out["gt_labels_image"].apply(split_labels)

    tp, fp, n_pred, n_tp, n_fp = [], [], [], [], []
    for p, g in zip(pred_list, gt_list):
        ps = set(p)
        gs = set(g)
        tp_set = ps.intersection(gs)
        fp_set = ps.difference(gs)
        tp.append(join_labels(list(tp_set)))
        fp.append(join_labels(list(fp_set)))
        n_pred.append(len(ps))
        n_tp.append(len(tp_set))
        n_fp.append(len(fp_set))

    out["tp_labels_in_gt"] = tp
    out["fp_labels_not_in_gt"] = fp
    out["n_pred"] = n_pred
    out["n_tp"] = n_tp
    out["n_fp"] = n_fp

    return out


def discover_runs_dir(start_dir: Path) -> Optional[Path]:
    candidates = []
    for needle in ["qwen2_5_vl", "llama3_2_vision", "gemma"]:
        for p in start_dir.rglob(needle):
            if p.is_dir():
                candidates.append(p.parent)
    candidates = list(dict.fromkeys(candidates))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="", help="Diretório que contém as pastas dos modelos")
    ap.add_argument("--gt_dir", type=str, default="datasets", help="Diretório base com crops_gt_black/crops_gt_white")
    ap.add_argument("--out_xlsx", type=str, default="DL_all_vlms_baseline_black_white_tables.xlsx", help="Arquivo XLSX de saída")
    args = ap.parse_args()

    repo = Path(".").resolve()
    runs_dir: Optional[Path] = None

    if args.runs_dir:
        runs_dir = Path(args.runs_dir).expanduser().resolve()
        if not runs_dir.exists():
            print(f"[WARN] runs_dir informado não existe: {runs_dir} (vou tentar descobrir automaticamente)")
            runs_dir = None

    if runs_dir is None:
        auto = discover_runs_dir(repo)
        if auto is None:
            raise SystemExit("Não consegui descobrir automaticamente o runs_dir. Passe --runs_dir apontando para a pasta com qwen2_5_vl/llama3_2_vision/gemma3.")
        runs_dir = auto

    gt_dir = Path(args.gt_dir).expanduser().resolve()
    gt_map = build_gt_map(gt_dir)

    print(f"[INFO] runs_dir = {runs_dir}")
    print(f"[INFO] gt_dir   = {gt_dir}  (gt_map entries={len(gt_map)})")

    rows_all: List[pd.DataFrame] = []

    for model_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        variants_present = {p.name for p in model_dir.iterdir() if p.is_dir()}
        if not ({"baseline", "crops_black", "crops_white"} & variants_present):
            continue

        model_name = model_dir.name
        print(f"\n[MODEL] {model_name}")

        prompt_id_default = ""
        for v in ["crops_black", "crops_white", "baseline"]:
            vdir = model_dir / v
            if not vdir.exists():
                continue
            for split in ["test_open", "test_closed"]:
                mdir = vdir / split / "meta"
                if mdir.exists():
                    info = try_read_run_info(mdir)
                    if "prompt_id" in info:
                        prompt_id_default = info["prompt_id"]
                    elif "prompt" in info and not prompt_id_default:
                        prompt_id_default = info["prompt"]
                    if prompt_id_default:
                        break
            if prompt_id_default:
                break

        # baseline
        bdir = model_dir / "baseline"
        if bdir.exists():
            for split in ["test_open", "test_closed"]:
                split_base = bdir / split
                eval_dir = split_base / "eval"

                dfb = load_baseline_eval(eval_dir)
                if dfb.empty:
                    print(f"  [WARN] baseline/{split}: metrics_per_image.csv não encontrado.")
                    continue

                st = standardize_baseline_df(dfb, split, model_name, prompt_id_default, gt_map)
                rows_all.append(st)
                print(f"  [OK] baseline/{split}: {len(st)} linhas")

        # crops
        for variant in ["crops_black", "crops_white"]:
            vdir = model_dir / variant
            if not vdir.exists():
                continue

            for split in ["test_open", "test_closed"]:
                split_base = vdir / split
                eval_dir = split_base / "eval"

                if model_name.lower().startswith("gemma"):
                    dfc = build_gemma_crops_df_from_raw(split_base, split_expected=split, variant=variant)
                    src_note = f"{split_base} (raw jsonl)" if not dfc.empty else ""
                    if dfc.empty:
                        dfc = load_crops_eval(eval_dir)
                        src_note = dfc.attrs.get("__source_path__", "")
                else:
                    dfc = load_crops_eval(eval_dir)
                    src_note = dfc.attrs.get("__source_path__", "")

                if dfc.empty:
                    print(f"  [WARN] {variant}/{split}: nenhum CSV/raw encontrado.")
                    continue

                if src_note:
                    print(f"    [SRC] {variant}/{split}: {src_note}")

                st = standardize_crops_df(dfc, split, model_name, variant, prompt_id_default, gt_map)
                rows_all.append(st)
                print(f"  [OK] {variant}/{split}: {len(st)} linhas")

    if not rows_all:
        raise SystemExit("Nenhum resultado encontrado. Verifique --runs_dir.")

    per_crop = pd.concat(rows_all, ignore_index=True)

    # -------------------------
    # Deduplicar BASELINE por imagem
    # (mantém a linha mais rápida: menor seconds)
    # -------------------------
    baseline_mask = per_crop["variant"].fillna("") == "baseline"

    if baseline_mask.any():
        base = per_crop[baseline_mask].copy()
        other = per_crop[~baseline_mask].copy()

        # seconds pode estar NaN -> empurra pro fim
        base["seconds_num"] = pd.to_numeric(base.get("seconds", pd.NA), errors="coerce")
        base["seconds_num"] = base["seconds_num"].fillna(float("inf"))

        # ordena para manter primeiro o menor seconds
        base = base.sort_values(["model", "split", "image_id", "seconds_num"], ascending=[True, True, True, True])

        # remove duplicatas por imagem
        base = base.drop_duplicates(subset=["model", "split", "image_id"], keep="first")

        # volta sem coluna auxiliar
        base = base.drop(columns=["seconds_num"], errors="ignore")

        per_crop = pd.concat([base, other], ignore_index=True)

    for c in ["split", "variant", "model", "prompt_id"]:
        if c in per_crop.columns:
            per_crop[c] = per_crop[c].fillna("").astype(str)

    desired_cols = [
        "split", "variant", "image_id", "crop_file", "crop_gt_label_from_filename",
        "pred_labels", "gt_labels_image", "tp_labels_in_gt", "fp_labels_not_in_gt",
        "n_pred", "n_tp", "n_fp", "seconds", "model", "prompt_id"
    ]
    for c in desired_cols:
        if c not in per_crop.columns:
            per_crop[c] = ""
    per_crop = per_crop[desired_cols]

    tmp = per_crop[["model", "variant", "split", "image_id", "crop_file", "fp_labels_not_in_gt", "gt_labels_image"]].copy()
    tmp["fp_list"] = tmp["fp_labels_not_in_gt"].apply(split_labels)
    fp_long = tmp.explode("fp_list").dropna(subset=["fp_list"])
    fp_long = fp_long[fp_long["fp_list"].astype(str).str.len() > 0].rename(columns={"fp_list": "fp_label"})

    if fp_long.empty:
        fp_sources = pd.DataFrame(columns=["model", "variant", "split", "image_id", "fp_label",
                                           "crops_that_predicted_fp", "n_crops", "gt_labels_image"])
    else:
        fp_sources = fp_long.groupby(
            ["model", "variant", "split", "image_id", "fp_label"],
            as_index=False,
            dropna=False
        ).agg(
            crops_that_predicted_fp=("crop_file", lambda s: ";".join(sorted(set(map(str, s))))),
            n_crops=("crop_file", lambda s: len(set(map(str, s)))),
            gt_labels_image=("gt_labels_image", "first"),
        )

    out_xlsx = Path(args.out_xlsx).expanduser().resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        per_crop.to_excel(w, sheet_name="per_crop", index=False)
        fp_sources.to_excel(w, sheet_name="fp_sources", index=False)

    print(f"\n[DONE] XLSX gerado em: {out_xlsx}")
    print(f"       per_crop rows = {len(per_crop)}")
    print(f"       fp_sources rows = {len(fp_sources)}")


if __name__ == "__main__":
    main()
