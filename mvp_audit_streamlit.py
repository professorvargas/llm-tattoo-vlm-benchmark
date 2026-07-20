from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image

MODELS = {
    "Gemma3:12b": "runs/gemma3",
    "Qwen2.5-VL-7B": "runs/qwen2_5_vl",
    "LLaMA 3.2 Vision 11B": "runs/llama3_2_vision",
}

VARIANTS = {
    # Keep the internal key "baseline" because it matches the existing
    # folders and JSONL filenames. Only the interface label changes.
    "baseline": "Original image",
    "crops_black": "Crop Black",
    "crops_white": "Crop White",
}


def existing_path(*candidates: Path) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


@st.cache_data(show_spinner=False)
def discover_cases(repo_root: str, split: str) -> List[str]:
    root = Path(repo_root)
    cases: set[str] = set()

    crops_gt = root / "datasets" / "crops_gt" / split
    if crops_gt.exists():
        cases.update(p.name for p in crops_gt.iterdir() if p.is_dir())

    panels_assets = root / "panels_custom_with_images" / "assets" / split
    if panels_assets.exists():
        cases.update(p.name for p in panels_assets.iterdir() if p.is_dir())

    return sorted(cases)


def open_image_if_exists(path: Optional[Path]) -> Optional[Image.Image]:
    if not path or not path.exists():
        return None
    try:
        return Image.open(path)
    except Exception:
        return None


def parse_list_like(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        # separadores reais do seu projeto: ; e ,
        if ";" in text or "," in text:
            parts = re.split(r"[;,]", text)
            return [x.strip() for x in parts if x.strip()]

        return [text]

    return [str(value).strip()]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def case_image_path(repo_root: str, split: str, case_id: str) -> Optional[Path]:
    root = Path(repo_root)
    images_dir = root / "datasets" / split / "images"
    for ext in IMAGE_EXTS:
        p = images_dir / f"{case_id}{ext}"
        if p.exists():
            return p
    return None


def case_mask_path(repo_root: str, split: str, case_id: str) -> Optional[Path]:
    root = Path(repo_root)

    # prioridade 1: pasta demo/panel
    panel_dir = root / "panels_custom_with_images" / "assets" / split / case_id
    for name in ["mask.jpg", "mask.png"]:
        p = panel_dir / name
        if p.exists():
            return p

    # prioridade 2: máscara do dataset
    mask_dir = root / "datasets" / split / "mask_rgb"
    for ext in IMAGE_EXTS:
        p = mask_dir / f"{case_id}_mask{ext}"
        if p.exists():
            return p

    return None


def gt_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])

def gt_black_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt_black" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])


def gt_white_crop_paths(repo_root: str, split: str, case_id: str) -> List[Path]:
    root = Path(repo_root)
    crop_dir = root / "datasets" / "crops_gt_white" / split / case_id
    if not crop_dir.exists():
        return []
    return sorted([p for p in crop_dir.glob("*.png") if p.is_file()])

def crop_pairs_by_label(repo_root: str, split: str, case_id: str) -> List[Dict[str, Optional[Path]]]:
    black_map = {p.stem: p for p in gt_black_crop_paths(repo_root, split, case_id)}
    white_map = {p.stem: p for p in gt_white_crop_paths(repo_root, split, case_id)}
    labels = sorted(set(black_map.keys()) | set(white_map.keys()))

    pairs = []
    for label in labels:
        pairs.append({
            "label": label,
            "black": black_map.get(label),
            "white": white_map.get(label),
        })
    return pairs


def gt_labels_from_crops(repo_root: str, split: str, case_id: str) -> List[str]:
    labels = []
    for p in gt_crop_paths(repo_root, split, case_id):
        label = p.stem.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


CASE_KEYS = ["image_id", "image", "case_id", "id", "file", "filename", "img"]
LABEL_KEYS = [
    "pred_labels",
    "labels_pred",
    "labels",
    "predicted_labels",
    "pred",
    "prediction",
    "mapped_labels",
    "parsed_labels",
]
GT_KEYS = ["gt_labels", "gt", "ground_truth", "gold_labels"]
UNKNOWN_KEYS = ["unknown", "has_unknown", "unknown_present"]
FP_KEYS = ["fp", "fp_img", "fp_per_image", "false_positives", "fp_count"]
FN_KEYS = ["fn", "fn_img", "fn_per_image", "false_negatives", "fn_count"]


def row_for_case(df: pd.DataFrame, case_id: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    for col in df.columns:
        if col.lower() in CASE_KEYS or any(k in col.lower() for k in CASE_KEYS):
            series = df[col].astype(str)
            mask = series.str.contains(re.escape(case_id), regex=True, na=False)
            if mask.any():
                return df[mask].iloc[0]
    joined = df.astype(str).agg(" | ".join, axis=1)
    mask = joined.str.contains(re.escape(case_id), regex=True, na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None


def extract_from_row(row: Optional[pd.Series], preferred_keys: Iterable[str]) -> Any:
    if row is None:
        return None
    lowered = {str(col).lower(): col for col in row.index}
    for key in preferred_keys:
        for col_lower, col_orig in lowered.items():
            if key == col_lower or key in col_lower:
                return row[col_orig]
    return None


@st.cache_data(show_spinner=False)
def load_variant_tables(repo_root: str, model_dir: str, split: str, variant: str) -> Dict[str, pd.DataFrame]:
    root = Path(repo_root)
    base = root / model_dir / variant / split / "eval"
    return {
        "metrics_per_image": read_csv_safe(base / "metrics_per_image.csv"),
        "metrics_per_crop": read_csv_safe(base / "metrics_per_crop.csv"),
        "pred_crop_labels": read_csv_safe(base / "pred_crop_labels_per_image.csv"),
        "best_images": read_csv_safe(base / "best_images.csv"),
        "worst_images": read_csv_safe(base / "worst_images.csv"),
    }


def preferred_jsonl_filename(variant: str, split: str) -> str:
    if variant == "baseline":
        return f"p1_baseline_{split}.jsonl"
    if variant == "crops_black":
        return f"p1_crop_{split}.jsonl"
    if variant == "crops_white":
        return f"p1_crop_white_{split}.jsonl"
    return ""

@st.cache_data(show_spinner=False)
def load_variant_jsonl(repo_root: str, model_dir: str, split: str, variant: str) -> List[Dict[str, Any]]:
    root = Path(repo_root)
    raw_dir = root / model_dir / variant / split / "raw"
    if not raw_dir.exists():
        return []

    preferred = raw_dir / preferred_jsonl_filename(variant, split)
    if preferred.exists():
        return read_jsonl_safe(preferred)

    jsonls = sorted(raw_dir.glob("*.jsonl"))
    if not jsonls:
        return []

    return read_jsonl_safe(jsonls[0])


def extract_jsonl_case(records: List[Dict[str, Any]], case_id: str) -> Optional[Dict[str, Any]]:
    for record in records:
        haystack = json.dumps(record, ensure_ascii=False)
        if case_id in haystack:
            return record
    return None

def first_case_row(case_id: str, *dfs: pd.DataFrame) -> Optional[pd.Series]:
    for df in dfs:
        row = row_for_case(df, case_id)
        if row is not None:
            return row
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_label_set(values: List[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def collect_variant_prediction(repo_root: str, model_name: str, split: str, variant: str, case_id: str) -> Dict[str, Any]:
    model_dir = MODELS[model_name]
    tables = load_variant_tables(repo_root, model_dir, split, variant)

    row_labels = first_case_row(
        case_id,
        tables["pred_crop_labels"],
        tables["metrics_per_image"],
        tables["metrics_per_crop"],
        tables["best_images"],
        tables["worst_images"],
    )

    row_metrics = first_case_row(
        case_id,
        tables["metrics_per_image"],
        tables["metrics_per_crop"],
        tables["best_images"],
        tables["worst_images"],
        tables["pred_crop_labels"],
    )

    labels = parse_list_like(extract_from_row(row_labels, LABEL_KEYS))

    gt_labels = parse_list_like(extract_from_row(row_metrics, GT_KEYS))
    if not gt_labels:
        gt_labels = parse_list_like(extract_from_row(row_labels, GT_KEYS))

    fp = extract_from_row(row_metrics, FP_KEYS)
    fn = extract_from_row(row_metrics, FN_KEYS)

    unknown = extract_from_row(row_metrics, UNKNOWN_KEYS)
    if is_missing(unknown):
        unknown = extract_from_row(row_labels, UNKNOWN_KEYS)

    if not gt_labels:
        gt_labels = gt_labels_from_crops(repo_root, split, case_id)

    if not labels:
        record = extract_jsonl_case(load_variant_jsonl(repo_root, model_dir, split, variant), case_id)
        if record:
            for key in ["json_obj", "parsed", "prediction", "pred"]:
                candidate = record.get(key)
                if isinstance(candidate, dict):
                    labels = parse_list_like(candidate.get("dominant_elements") or candidate.get("labels"))
                    if is_missing(unknown) and "unknown" in candidate:
                        unknown = candidate.get("unknown")
                    if labels:
                        break

            if not labels:
                labels = parse_list_like(record.get("labels") or record.get("pred_labels"))

            if not labels and isinstance(record.get("json_obj"), dict):
                labels = parse_list_like(record["json_obj"].get("labels"))
                if is_missing(unknown) and "unknown" in record["json_obj"]:
                    unknown = record["json_obj"]["unknown"]

    if is_missing(unknown):
        unknown = any(str(x).strip().lower() == "unknown" for x in labels)

    # fallback: calcula FP/FN se a tabela não trouxer esses valores
    if gt_labels and labels:
        pred_set = normalize_label_set(labels)
        gt_set = normalize_label_set(gt_labels)

        if is_missing(fp):
            fp = len(pred_set - gt_set)

        if is_missing(fn):
            fn = len(gt_set - pred_set)

    row = row_metrics if row_metrics is not None else row_labels

    return {
        "labels": labels,
        "gt_labels": gt_labels,
        "fp": fp,
        "fn": fn,
        "unknown": unknown,
        "row": row,
    }

def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def build_row_details(payload: Dict[str, Any], case_id: str, split: str) -> pd.DataFrame:
    details: Dict[str, Any] = {}

    row = payload.get("row")
    if row is not None:
        details.update({str(col): row[col] for col in row.index})

    pred_labels = payload.get("labels", []) or []
    gt_labels = payload.get("gt_labels", []) or []

    pred_set = normalize_label_set(pred_labels)
    gt_set = normalize_label_set(gt_labels)

    tp = len(pred_set & gt_set)

    fp = payload.get("fp")
    fn = payload.get("fn")

    if is_missing(fp):
        fp = len(pred_set - gt_set)
    if is_missing(fn):
        fn = len(gt_set - pred_set)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    jaccard = safe_div(tp, tp + fp + fn)

    pred_size = len(pred_set)
    gold_size = len(gt_set)
    overprediction = safe_div(pred_size, gold_size)

    # Ajuste este valor se seu benchmark usar outro tamanho de vocabulário
    label_space_size = 35
    hamming_loss = safe_div(fp + fn, label_space_size)

    # Preenche campos faltantes sem sobrescrever os já existentes
    details.setdefault("split", split)
    details.setdefault("image_id", case_id)
    details.setdefault("pred_labels", ";".join(pred_labels))
    details.setdefault("gt_labels", ";".join(gt_labels))
    details.setdefault("tp", tp)
    details.setdefault("fp", fp)
    details.setdefault("fn", fn)
    details.setdefault("precision", precision)
    details.setdefault("recall", recall)
    details.setdefault("f1", f1)
    details.setdefault("jaccard", jaccard)
    details.setdefault("hamming_loss", hamming_loss)
    details.setdefault("overprediction", overprediction)
    details.setdefault("pred_size", pred_size)
    details.setdefault("gold_size", gold_size)
    details.setdefault("unknown", payload.get("unknown"))

    preferred_order = [
        "split",
        "image_id",
        "pred_labels",
        "gt_labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "hamming_loss",
        "overprediction",
        "pred_size",
        "gold_size",
        "unknown",
    ]

    ordered_keys = [k for k in preferred_order if k in details]
    ordered_keys += [k for k in details.keys() if k not in ordered_keys]

    return pd.DataFrame({
        "field": ordered_keys,
        "value": [details[k] for k in ordered_keys],
    })

def panel_assets(repo_root: str, split: str, case_id: str) -> Dict[str, Optional[Path]]:
    root = Path(repo_root)
    panel_dir = root / "panels_custom_with_images" / "assets" / split / case_id

    baseline = existing_path(
        panel_dir / "baseline.jpg",
        panel_dir / "baseline.png",
        case_image_path(repo_root, split, case_id),
    )

    mask = existing_path(
        panel_dir / "mask.jpg",
        panel_dir / "mask.png",
        case_mask_path(repo_root, split, case_id),
    )

    # prioridade 1: painéis demo
    black_candidates = sorted(panel_dir.glob("crop_black_*"))
    white_candidates = sorted(panel_dir.glob("crop_white_*"))

    # prioridade 2: datasets completos
    if not black_candidates:
        black_candidates = gt_black_crop_paths(repo_root, split, case_id)

    if not white_candidates:
        white_candidates = gt_white_crop_paths(repo_root, split, case_id)

    
    return {
    "baseline": baseline,
    "mask": mask,
    "crop_black": black_candidates[0] if black_candidates else None,
    "crop_white": white_candidates[0] if white_candidates else None,
}


def compute_priority(repo_root: str, split: str, case_id: str) -> Dict[str, Any]:
    all_predictions: Dict[str, Dict[str, Any]] = {}
    flags: List[str] = []
    score = 0

    for model_name in MODELS:
        all_predictions[model_name] = {}
        base = collect_variant_prediction(repo_root, model_name, split, "baseline", case_id)
        black = collect_variant_prediction(repo_root, model_name, split, "crops_black", case_id)
        white = collect_variant_prediction(repo_root, model_name, split, "crops_white", case_id)
        all_predictions[model_name]["baseline"] = base
        all_predictions[model_name]["crops_black"] = black
        all_predictions[model_name]["crops_white"] = white

        base_set = set(base["labels"])
        black_set = set(black["labels"])
        white_set = set(white["labels"])

        if "unknown" in {x.lower() for x in base_set | black_set | white_set}:
            flags.append(f"{model_name}: unknown present")
            score += 2
        if black_set != white_set and (black_set or white_set):
            flags.append(f"{model_name}: flip black/white")
            score += 2
        if base_set != black_set or base_set != white_set:
            flags.append(f"{model_name}: baseline/crops divergence")
            score += 1

        for variant_name, payload in [("baseline", base), ("crops_black", black), ("crops_white", white)]:
            fp = payload.get("fp")
            try:
                if fp is not None and float(fp) >= 3:
                    flags.append(f"{model_name} {variant_name}: high FP ({fp})")
                    score += 2
            except Exception:
                pass

    return {"score": score, "flags": flags, "predictions": all_predictions}


@st.cache_data(show_spinner=False)
def build_case_ranking(repo_root: str, split: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_id in discover_cases(repo_root, split):
        priority = compute_priority(repo_root, split, case_id)
        rows.append({
            "case_id": case_id,
            "priority_score": priority["score"],
            "n_flags": len(priority["flags"]),
            "flags": " | ".join(priority["flags"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["priority_score", "n_flags", "case_id"], ascending=[False, False, True]).reset_index(drop=True)

def theoretical_max_priority() -> int:
    # por modelo: unknown(2) + flip(2) + divergence(1) + high FP em 3 variantes (6)
    per_model_max = 2 + 2 + 1 + (3 * 2)
    return len(MODELS) * per_model_max  # 33 com 3 modelos


def observed_max_priority(ranking: pd.DataFrame) -> int:
    if ranking.empty:
        return 0
    return int(ranking["priority_score"].max())


def priority_band(score: int) -> str:
    # faixas simples e fáceis de explicar para o auditor
    if score <= 3:
        return "Very low"
    if score <= 9:
        return "Low"
    if score <= 16:
        return "Medium"
    if score <= 20:
        return "High"
    return "Critical"

def audit_log_path(repo_root: str) -> Path:
    path = Path(repo_root) / "audit_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "audit_log.csv"


def append_audit_decision(repo_root: str, payload: Dict[str, Any]) -> None:
    out = audit_log_path(repo_root)
    frame = pd.DataFrame([payload])
    if out.exists():
        frame.to_csv(out, mode="a", header=False, index=False)
    else:
        frame.to_csv(out, index=False)


def show_image(title: str, image_path: Optional[Path]) -> None:
    st.markdown(f"**{title}**")
    image = open_image_if_exists(image_path)
    if image is None:
        st.caption("Image not found.")
    else:
        st.image(image, width=320)
        st.caption(str(image_path))

# -----------------------------------------------------------------------------
# Data hygiene and expert-review helpers
# -----------------------------------------------------------------------------

def _find_case_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    exact = {str(c).lower(): str(c) for c in df.columns}
    for key in CASE_KEYS:
        if key in exact:
            return exact[key]
    for col in df.columns:
        lowered = str(col).lower()
        if any(key in lowered for key in ["image_id", "case_id", "filename", "image"]):
            return str(col)
    return None


def _clean_eval_table(df: pd.DataFrame, split: str, *, deduplicate: bool) -> pd.DataFrame:
    if df.empty:
        return df

    cleaned = df.copy()
    split_columns = [c for c in cleaned.columns if str(c).lower() == "split"]
    if split_columns:
        split_col = split_columns[0]
        split_mask = cleaned[split_col].astype(str).str.strip().eq(split)
        if split_mask.any():
            cleaned = cleaned.loc[split_mask].copy()

    if deduplicate:
        case_col = _find_case_column(cleaned)
        if case_col:
            cleaned = cleaned.drop_duplicates(subset=[case_col], keep="last")

    return cleaned.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_variant_tables(repo_root: str, model_dir: str, split: str, variant: str) -> Dict[str, pd.DataFrame]:
    """Load evaluation tables and remove duplicated cross-split records safely."""
    root = Path(repo_root)
    base = root / model_dir / variant / split / "eval"
    return {
        "metrics_per_image": _clean_eval_table(
            read_csv_safe(base / "metrics_per_image.csv"), split, deduplicate=True
        ),
        "metrics_per_crop": _clean_eval_table(
            read_csv_safe(base / "metrics_per_crop.csv"), split, deduplicate=False
        ),
        "pred_crop_labels": _clean_eval_table(
            read_csv_safe(base / "pred_crop_labels_per_image.csv"), split, deduplicate=True
        ),
        "best_images": _clean_eval_table(
            read_csv_safe(base / "best_images.csv"), split, deduplicate=True
        ),
        "worst_images": _clean_eval_table(
            read_csv_safe(base / "worst_images.csv"), split, deduplicate=True
        ),
    }


def load_label_vocabulary(repo_root: str) -> List[str]:
    root = Path(repo_root)
    candidates = [
        root / "data_meta" / "tssd2023_id2name.json",
        root / "datasets" / "tssd2023_id2name.json",
        root / "datasets" / "data_meta" / "tssd2023_id2name.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                labels = [str(value).strip() for value in payload.values()]
                return sorted({label for label in labels if label and label != "background"})
            if isinstance(payload, list):
                return sorted({str(value).strip() for value in payload if str(value).strip()})
        except Exception:
            continue

    # Stable fallback based on the benchmark vocabulary.
    return sorted({
        "anchor", "bird", "branch", "butterfly", "cat", "crown", "diamond",
        "dog", "eagle", "fire", "fish", "flower", "fox", "gun", "heart",
        "key", "knife", "leaf", "lion", "mermaid", "octopus", "owl",
        "ribbon", "rope", "scorpion", "shark", "shield", "skull", "snake",
        "spider", "star", "tiger", "water", "wolf", "unknown",
    })


def expert_review_log_path(repo_root: str) -> Path:
    directory = Path(repo_root) / "audit_logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "expert_review_log.csv"


def read_expert_review_log(repo_root: str) -> pd.DataFrame:
    path = expert_review_log_path(repo_root)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def append_expert_review(repo_root: str, payload: Dict[str, Any]) -> None:
    path = expert_review_log_path(repo_root)
    current = read_expert_review_log(repo_root)
    new_row = pd.DataFrame([{key: "" if value is None else value for key, value in payload.items()}])

    # Rewrite rather than append positionally. This keeps the CSV valid even if
    # the schema evolves during prototyping.
    if current.empty:
        combined = new_row
    else:
        combined = pd.concat([current, new_row], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)


def latest_reviews_by_case(review_log: pd.DataFrame) -> pd.DataFrame:
    if review_log.empty or "case_id" not in review_log.columns:
        return pd.DataFrame(columns=["case_id", "review_status", "review_action", "reviewer", "timestamp"])

    ordered = review_log.copy()
    if "timestamp" in ordered.columns:
        ordered["_parsed_time"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
        ordered = ordered.sort_values("_parsed_time", na_position="first")
    latest = ordered.drop_duplicates(subset=["case_id"], keep="last")

    keep = [c for c in ["case_id", "review_status", "review_action", "reviewer", "timestamp"] if c in latest.columns]
    return latest[keep].copy()


def build_review_queue(ranking: pd.DataFrame, review_log: pd.DataFrame) -> pd.DataFrame:
    queue = ranking.copy()
    latest = latest_reviews_by_case(review_log)
    if not latest.empty:
        queue = queue.merge(latest, on="case_id", how="left")

    for column, default in [
        ("review_status", "Pending"),
        ("review_action", ""),
        ("reviewer", ""),
        ("timestamp", ""),
    ]:
        if column not in queue.columns:
            queue[column] = default
        else:
            queue[column] = queue[column].replace("", pd.NA).fillna(default)

    queue["band"] = queue["priority_score"].apply(priority_band)
    return queue


def latest_case_review(review_log: pd.DataFrame, case_id: str) -> Optional[pd.Series]:
    if review_log.empty or "case_id" not in review_log.columns:
        return None
    case_rows = review_log[review_log["case_id"].astype(str) == str(case_id)].copy()
    if case_rows.empty:
        return None
    if "timestamp" in case_rows.columns:
        case_rows["_parsed_time"] = pd.to_datetime(case_rows["timestamp"], errors="coerce", utc=True)
        case_rows = case_rows.sort_values("_parsed_time", na_position="first")
    return case_rows.iloc[-1]


def labels_to_text(labels: List[str]) -> str:
    return ", ".join(labels) if labels else "None"


def label_badges(labels: List[str], *, muted: bool = False) -> None:
    if not labels:
        st.caption("No labels available.")
        return
    css_class = "label-badge muted" if muted else "label-badge"
    chips = "".join(
        f'<span class="{css_class}">{str(label).strip()}</span>'
        for label in labels
        if str(label).strip()
    )
    st.markdown(f'<div class="label-row">{chips}</div>', unsafe_allow_html=True)


def pretty_flag(flag: str) -> str:
    text = str(flag)
    text = text.replace(": unknown present", ": detected an unknown label")
    text = text.replace(": flip black/white", ": predictions changed between black and white crops")
    text = text.replace(": baseline/crops divergence", ": original-image and crop predictions disagree")
    text = text.replace("crops_black", "black crop")
    text = text.replace("crops_white", "white crop")
    text = text.replace("baseline", "original image")
    text = text.replace("high FP", "high overprediction")
    return text


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def variant_agreement_status(variants: Dict[str, Dict[str, Any]]) -> str:
    sets = [normalize_label_set(variants[name].get("labels", [])) for name in VARIANTS]
    if sets[0] == sets[1] == sets[2]:
        return "Stable across contexts"
    return "Context-sensitive"


def model_evidence_frame(priority: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, variants in priority["predictions"].items():
        stability = variant_agreement_status(variants)
        for variant_key, variant_title in VARIANTS.items():
            payload = variants[variant_key]
            rows.append({
                "Model": model_name,
                "Input": variant_title,
                "Predicted labels": labels_to_text(payload.get("labels", [])),
                "FP": payload.get("fp", ""),
                "FN": payload.get("fn", ""),
                "Unknown": bool_value(payload.get("unknown")),
                "Context assessment": stability,
            })
    return pd.DataFrame(rows)


def review_status_for_action(action: str) -> str:
    mapping = {
        "Confirm current reference": "Reviewed",
        "Propose revised reference": "Revision proposed",
        "Mark case as ambiguous": "Ambiguous",
        "Send case to re-annotation": "Re-annotation requested",
        "Exclude from high-confidence pool": "Excluded",
    }
    return mapping.get(action, "Reviewed")


def safe_multiselect_defaults(options: List[str], current: List[str]) -> List[str]:
    option_set = set(options)
    return [label for label in current if label in option_set]


def show_image_card(title: str, image_path: Optional[Path], *, caption_path: bool = False) -> None:
    st.markdown(f"#### {title}")
    image = open_image_if_exists(image_path)
    if image is None:
        st.info("Image not available for this case.")
        return
    st.image(image, width="stretch")
    if caption_path:
        st.caption(str(image_path))


def render_priority_reasons(priority: Dict[str, Any]) -> None:
    st.markdown("### Why this case was prioritized")
    if not priority["flags"]:
        st.success("No priority indicators were detected by the current heuristics.")
        return
    for flag in priority["flags"]:
        st.markdown(f'<div class="reason-item">{pretty_flag(flag)}</div>', unsafe_allow_html=True)


def render_latest_review_card(latest: Optional[pd.Series]) -> None:
    if latest is None:
        st.info("This case has not been reviewed yet.")
        return
    action = latest.get("review_action", "")
    status = latest.get("review_status", "Reviewed")
    reviewer = latest.get("reviewer", "")
    timestamp = latest.get("timestamp", "")
    st.markdown(
        f"""
        <div class="review-summary-card">
          <div class="eyebrow">LATEST EXPERT DECISION</div>
          <div class="review-summary-title">{status}</div>
          <div class="review-summary-text">{action}</div>
          <div class="review-summary-meta">{reviewer or 'Reviewer not recorded'} · {timestamp or 'Time not recorded'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ta-ink: #172033;
            --ta-muted: #667085;
            --ta-line: #e4e7ec;
            --ta-soft: #f7f8fa;
            --ta-accent: #2f6fed;
            --ta-accent-soft: #eef4ff;
            --ta-warning-soft: #fff7e6;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.6rem;
            padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--ta-line);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.3rem;
        }
        .app-kicker {
            color: var(--ta-accent);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .app-title {
            color: var(--ta-ink);
            font-size: clamp(2rem, 3.2vw, 3.5rem);
            font-weight: 760;
            line-height: 1.04;
            margin: 0;
        }
        .app-subtitle {
            color: var(--ta-muted);
            font-size: 1.05rem;
            margin-top: 0.55rem;
            margin-bottom: 1.1rem;
        }
        .case-id {
            color: var(--ta-muted);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }
        .section-card {
            border: 1px solid var(--ta-line);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: white;
            margin-bottom: 0.8rem;
        }
        .label-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.4rem 0 0.75rem 0;
        }
        .label-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.62rem;
            background: var(--ta-accent-soft);
            border: 1px solid #cddcff;
            color: #234f9d;
            font-size: 0.86rem;
            font-weight: 650;
        }
        .label-badge.muted {
            background: #f2f4f7;
            border-color: #e4e7ec;
            color: #475467;
        }
        .reason-item {
            border-left: 4px solid #f79009;
            background: var(--ta-warning-soft);
            border-radius: 7px;
            padding: 0.72rem 0.85rem;
            margin: 0.48rem 0;
            color: #7a2e0e;
        }
        .review-summary-card {
            border: 1px solid #cddcff;
            background: var(--ta-accent-soft);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.9rem;
        }
        .eyebrow {
            color: #475467;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.08em;
        }
        .review-summary-title {
            color: var(--ta-ink);
            font-size: 1.25rem;
            font-weight: 760;
            margin-top: 0.2rem;
        }
        .review-summary-text {
            color: #344054;
            margin-top: 0.25rem;
        }
        .review-summary-meta {
            color: var(--ta-muted);
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--ta-line);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            background: white;
        }
        [data-testid="stMetricLabel"] {
            color: var(--ta-muted);
        }
        [data-testid="stForm"] {
            border: 1px solid var(--ta-line);
            border-radius: 14px;
            padding: 1rem;
            background: var(--ta-soft);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--ta-line);
            border-radius: 10px;
            overflow: hidden;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--ta-line);
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="TattooAudit — Expert Review",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    split = "test_open"
    auto_repo_root = str(Path(__file__).resolve().parent)

    with st.sidebar:
        st.markdown("## Review queue")
        st.caption("Triage cases and open one case for expert review.")

        use_custom_root = st.checkbox("Use manual project path", value=False)
        if use_custom_root:
            repo_root = st.text_input("Project root", value=auto_repo_root)
        else:
            repo_root = auto_repo_root
            st.text_input("Project root", value=auto_repo_root, disabled=True)

        with st.spinner("Building the review queue..."):
            ranking = build_case_ranking(repo_root, split)

        if ranking.empty:
            st.error("No test_open cases were found. Check the project root and dataset folders.")
            st.stop()

        review_log = read_expert_review_log(repo_root)
        queue = build_review_queue(ranking, review_log)
        max_observed_score = observed_max_priority(ranking)
        max_theoretical_score = theoretical_max_priority()

        min_priority = st.slider(
            "Minimum audit priority",
            min_value=0,
            max_value=max_observed_score,
            value=0,
            step=1,
        )
        status_filter = st.selectbox(
            "Review status",
            ["All cases", "Pending", "Reviewed or routed"],
            index=0,
        )

        filtered = queue[queue["priority_score"] >= min_priority].copy()
        if status_filter == "Pending":
            filtered = filtered[filtered["review_status"] == "Pending"]
        elif status_filter == "Reviewed or routed":
            filtered = filtered[filtered["review_status"] != "Pending"]

        if filtered.empty:
            st.warning("No cases match the selected filters.")
            st.stop()

        selected_cases = filtered["case_id"].astype(str).tolist()
        case_id = st.selectbox(
            "Case",
            selected_cases,
            format_func=lambda case: f"{case} · priority {int(queue.loc[queue['case_id'] == case, 'priority_score'].iloc[0])}",
        )

        total_cases = len(queue)
        reviewed_cases = int((queue["review_status"] != "Pending").sum())
        pending_cases = total_cases - reviewed_cases
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Pending", pending_cases)
        c2.metric("Reviewed", reviewed_cases)
        st.caption(f"Dataset fixed to {split}. Score range: 0–{max_observed_score} observed.")

    priority = compute_priority(repo_root, split, case_id)
    assets = panel_assets(repo_root, split, case_id)
    crop_pairs = crop_pairs_by_label(repo_root, split, case_id)
    gt_labels = gt_labels_from_crops(repo_root, split, case_id)
    vocabulary = load_label_vocabulary(repo_root)
    latest_review = latest_case_review(review_log, case_id)

    st.markdown('<div class="app-kicker">Human-in-the-loop inspection</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">TattooAudit</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Expert review workspace for prioritizing, inspecting, and routing semantically unstable tattoo cases.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="case-id">Selected case: {case_id}</div>', unsafe_allow_html=True)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Audit-priority score", f"{priority['score']} / {max_observed_score}")
    metric2.metric("Priority band", priority_band(priority["score"]))
    metric3.metric("Triggered indicators", len(priority["flags"]))
    metric4.metric("Review status", latest_review.get("review_status", "Pending") if latest_review is not None else "Pending")

    st.caption(
        f"The score is a relative triage index, not an error probability. "
        f"Observed maximum: {max_observed_score}; theoretical maximum under the current heuristics: {max_theoretical_score}."
    )

    review_tab, evidence_tab, history_tab, queue_tab = st.tabs(
        ["Case review", "Model evidence", "Decision history", "Review queue"]
    )

    with review_tab:
        left, right = st.columns([1.65, 1.0], gap="large")

        with left:
            st.markdown("## Case evidence")
            image_col, mask_col = st.columns(2, gap="medium")
            with image_col:
                show_image_card("Original image", assets["baseline"])
            with mask_col:
                show_image_card("Reference mask", assets["mask"])

            st.markdown("### Current expert reference")
            label_badges(gt_labels)
            st.caption("The reference annotation remains the primary authority. Model outputs provide supporting evidence for review.")

            if crop_pairs:
                with st.expander("Inspect GT-derived black and white crops", expanded=False):
                    for pair in crop_pairs:
                        st.markdown(f"#### Reference label: {pair['label']}")
                        black_col, white_col = st.columns(2)
                        with black_col:
                            show_image_card("Black-background crop", pair["black"])
                        with white_col:
                            show_image_card("White-background crop", pair["white"])
                        st.markdown("---")

            render_priority_reasons(priority)

        with right:
            st.markdown("## Expert decision")
            render_latest_review_card(latest_review)

            with st.form(f"expert_review_form_{case_id}", clear_on_submit=False):
                reviewer = st.text_input("Reviewer name")
                action = st.radio(
                    "Review action",
                    [
                        "Confirm current reference",
                        "Propose revised reference",
                        "Mark case as ambiguous",
                        "Send case to re-annotation",
                        "Exclude from high-confidence pool",
                    ],
                )
                reviewed_labels = st.multiselect(
                    "Reference labels after review",
                    options=vocabulary,
                    default=safe_multiselect_defaults(vocabulary, gt_labels),
                    help="Keep the current labels when confirming the reference, or edit them when proposing a revision.",
                )
                additional_label = st.text_input(
                    "Additional label not listed above (optional)",
                    placeholder="Type one label only when the vocabulary does not contain it",
                )
                confidence = st.selectbox(
                    "Reviewer confidence",
                    ["High", "Medium", "Low", "Not assessed"],
                    index=3,
                )
                operational_relevance = st.selectbox(
                    "Operational relevance",
                    ["Not assessed", "Low", "Medium", "High"],
                    index=0,
                )
                rationale = st.text_area(
                    "Decision rationale",
                    placeholder="Briefly explain the evidence that supports this decision.",
                    height=130,
                )
                submitted = st.form_submit_button("Save expert decision", type="primary")

            if submitted:
                final_labels = [label for label in reviewed_labels if label]
                if additional_label.strip():
                    final_labels.append(additional_label.strip())
                final_labels = sorted(dict.fromkeys(final_labels))

                validation_error = None
                if not reviewer.strip():
                    validation_error = "Enter the reviewer name before saving."
                elif action != "Confirm current reference" and not rationale.strip():
                    validation_error = "Add a short rationale for revision, ambiguity, routing, or exclusion decisions."
                elif action == "Propose revised reference" and not final_labels:
                    validation_error = "Select at least one proposed reference label."

                if validation_error:
                    st.error(validation_error)
                else:
                    payload = {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "case_id": case_id,
                        "split": split,
                        "reviewer": reviewer.strip(),
                        "review_action": action,
                        "review_status": review_status_for_action(action),
                        "original_reference_labels": ";".join(gt_labels),
                        "reviewed_reference_labels": ";".join(final_labels),
                        "reviewer_confidence": confidence,
                        "operational_relevance": operational_relevance,
                        "priority_score": priority["score"],
                        "priority_band": priority_band(priority["score"]),
                        "priority_flags": " | ".join(priority["flags"]),
                        "rationale": rationale.strip(),
                    }
                    append_expert_review(repo_root, payload)
                    st.success(f"Expert decision saved to {expert_review_log_path(repo_root)}")
                    st.rerun()

    with evidence_tab:
        st.markdown("## Model evidence")
        st.info("The VLM outputs support expert inspection; they do not replace the reference annotation or the expert's final judgment.")

        overview = model_evidence_frame(priority)
        st.dataframe(
            overview,
            width="stretch",
            hide_index=True,
            column_config={
                "FP": st.column_config.NumberColumn(format="%d"),
                "FN": st.column_config.NumberColumn(format="%d"),
                "Unknown": st.column_config.CheckboxColumn(),
            },
        )

        for model_name, variants in priority["predictions"].items():
            with st.expander(
                f"{model_name} — {variant_agreement_status(variants)}",
                expanded=False,
            ):
                model_cols = st.columns(3, gap="medium")
                for index, variant_key in enumerate(["baseline", "crops_black", "crops_white"]):
                    payload = variants[variant_key]
                    with model_cols[index]:
                        st.markdown(f"### {VARIANTS[variant_key]}")
                        st.markdown("**Predicted labels**")
                        label_badges(payload.get("labels", []), muted=False)

                        m1, m2 = st.columns(2)
                        m1.metric("FP", payload.get("fp", "—"))
                        m2.metric("FN", payload.get("fn", "—"))
                        st.caption(f"Unknown: {'Yes' if bool_value(payload.get('unknown')) else 'No'}")

                        with st.expander("Technical metrics", expanded=False):
                            details = build_row_details(payload, case_id, split).copy()
                            details["field"] = details["field"].astype(str)
                            details["value"] = details["value"].astype(str)
                            st.dataframe(details, width="stretch", hide_index=True)

    with history_tab:
        st.markdown("## Decision history")
        if review_log.empty or "case_id" not in review_log.columns:
            st.info("No expert decisions have been recorded yet.")
        else:
            case_history = review_log[review_log["case_id"].astype(str) == str(case_id)].copy()
            if case_history.empty:
                st.info("This case has no recorded decisions.")
            else:
                preferred = [
                    "timestamp", "reviewer", "review_status", "review_action",
                    "original_reference_labels", "reviewed_reference_labels",
                    "reviewer_confidence", "operational_relevance", "rationale",
                ]
                columns = [column for column in preferred if column in case_history.columns]
                st.dataframe(case_history[columns].iloc[::-1], width="stretch", hide_index=True)

        log_path = expert_review_log_path(repo_root)
        if log_path.exists():
            st.download_button(
                "Download expert review log",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="text/csv",
            )

    with queue_tab:
        st.markdown("## Review queue")
        st.caption("The queue preserves the original audit-priority heuristic and adds the latest human-review status.")

        queue_view = queue.copy()
        queue_view["Audit-priority"] = queue_view["priority_score"].apply(
            lambda value: f"{int(value)} / {max_observed_score}"
        )
        queue_view = queue_view.rename(columns={
            "case_id": "Case",
            "band": "Band",
            "n_flags": "Indicators",
            "review_status": "Review status",
            "review_action": "Latest action",
            "reviewer": "Reviewer",
            "timestamp": "Last reviewed",
        })
        display_columns = [
            "Case", "Audit-priority", "Band", "Indicators",
            "Review status", "Latest action", "Reviewer", "Last reviewed",
        ]
        st.dataframe(
            queue_view[display_columns],
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download review queue",
            data=queue_view[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="tattoo_audit_review_queue.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
