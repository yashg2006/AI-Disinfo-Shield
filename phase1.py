import os
import re
import json
import unicodedata
import warnings
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("phase1")

base_dir = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    # RU22FACT — local csv files
    "ru22fact_train":   os.path.join(base_dir, r"ru22fact-main\ru22fact-main\data\ru22fact\ru22fact_train.csv"),
    "ru22fact_dev":     os.path.join(base_dir, r"ru22fact-main\ru22fact-main\data\ru22fact\ru22fact_validate.csv"),
    "ru22fact_test":    os.path.join(base_dir, r"ru22fact-main\ru22fact-main\data\ru22fact\ru22fact_test.csv"),

    "fakeclaim":        os.path.join(base_dir, r"FakeClaim-main\fakeclaim.csv"),
    "multifake":        os.path.join(base_dir, r"Multi-Fake-DetectiVE\dataset.csv"),
    "multifake_images": os.path.join(base_dir, r"Multi-Fake-DetectiVE\images"),
    "nela_labels":      os.path.join(base_dir, r"NELA-GT-2022\labels.csv"),
    "nela_articles":    os.path.join(base_dir, r"NELA-GT-2022\articles.csv"),
    "output":           os.path.join(base_dir, r"phase1_output"),
}

Path(PATHS["output"]).mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    "supported":          "REAL",
    "refuted":            "FAKE",
    "not enough info":    "SUSPICIOUS",
    "nei":                "SUSPICIOUS",
    "true":               "REAL",
    "mostly true":        "REAL",
    "false":              "FAKE",
    "mostly false":       "FAKE",
    "pants on fire":      "FAKE",
    "misleading":         "SUSPICIOUS",
    "unverified":         "SUSPICIOUS",
    "disputed":           "SUSPICIOUS",
    "mixture":            "SUSPICIOUS",
    "half true":          "SUSPICIOUS",
    "barely true":        "SUSPICIOUS",
    "needs context":      "SUSPICIOUS",
    "real":               "REAL",
    "fake":               "FAKE",
    "1":                  "REAL",
    "0":                  "FAKE",
    1:                    "REAL",
    0:                    "FAKE",
}

def unify_label(raw_label):
    if pd.isna(raw_label):
        return None
    key = str(raw_label).strip().lower()
    return LABEL_MAP.get(key, None)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)           
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)   
    text = re.sub(r"@\w+", "[USER]", text)          
    text = re.sub(r"#(\w+)", r"\1", text)           
    text = re.sub(r"\s+", " ", text).strip()
    return text

def token_count(text: str) -> int:
    return len(text.split())

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    log.warning("langdetect not installed — skipping language detection column.")

def detect_language(text: str) -> str:
    if not LANGDETECT_AVAILABLE or len(text) < 20:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"

def load_ru22fact() -> pd.DataFrame:
    log.info("Loading RU22FACT...")
    frames = []
    for split_name, path_key in [
        ("train", "ru22fact_train"),
        ("dev",   "ru22fact_dev"),
        ("test",  "ru22fact_test"),
    ]:
        path = PATHS[path_key]
        if not os.path.exists(path):
            log.warning(f"  RU22FACT {split_name} not found at {path} — skipping.")
            continue

        df = pd.read_csv(path, low_memory=False)
        df["original_split"] = split_name
        df["source_dataset"] = "RU22FACT"
        frames.append(df)

    if not frames:
        log.error("No RU22FACT files loaded. Check your paths.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    col_remap = {
        "claim_text": "text", "claim": "text",
        "verdict": "raw_label", "label": "raw_label",
        "gold_label": "raw_label",
    }
    df.rename(columns={k: v for k, v in col_remap.items() if k in df.columns}, inplace=True)

    if "evidence" not in df.columns:
        df["evidence"] = ""
    if "explanation" not in df.columns:
        df["explanation"] = ""

    df["image_path"]      = None
    df["source_domain"]   = "ru22fact"
    df["timestamp"]       = None

    log.info(f"  RU22FACT loaded: {len(df)} rows")
    return df[["text", "raw_label", "evidence", "source_domain",
               "timestamp", "image_path", "source_dataset", "original_split"]]

def load_fakeclaim() -> pd.DataFrame:
    log.info("Loading FakeClaim...")
    path = PATHS["fakeclaim"]
    if not os.path.exists(path):
        log.warning(f"  FakeClaim not found at {path} — skipping.")
        return pd.DataFrame()
    return pd.DataFrame()

def load_multifake() -> pd.DataFrame:
    log.info("Loading Multi-Fake-DetectiVE...")
    path = PATHS["multifake"]
    if not os.path.exists(path):
        log.warning(f"  MultiFake not found at {path} — skipping.")
        return pd.DataFrame()
    return pd.DataFrame()

def load_nela() -> pd.DataFrame:
    log.info("Loading NELA-GT-2022 (SourceRank data)...")
    labels_path   = PATHS["nela_labels"]
    articles_path = PATHS["nela_articles"]
    if not os.path.exists(labels_path):
        log.warning(f"  NELA labels not found at {labels_path} — skipping.")
        return pd.DataFrame()
    return pd.DataFrame()

def build_unified_dataset() -> pd.DataFrame:
    log.info("\n=== Building unified claim dataset ===")
    frames = []
    for loader in [load_ru22fact, load_fakeclaim, load_multifake]:
        df = loader()
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No datasets loaded. Check your PATHS configuration.")

    unified = pd.concat(frames, ignore_index=True)
    log.info(f"Raw merged rows: {len(unified)}")

    log.info("Cleaning text...")
    # tqdm.pandas(desc="  Cleaning")
    unified["text"] = unified["text"].fillna("").astype(str)
    unified["text"] = unified["text"].apply(clean_text)

    log.info("Unifying labels...")
    unified["label"] = unified["raw_label"].apply(unify_label)

    before = len(unified)
    unified = unified[unified["text"].str.len() > 20]     
    unified = unified.dropna(subset=["label"])             
    log.info(f"Dropped {before - len(unified)} rows (empty text or unknown label)")

    if LANGDETECT_AVAILABLE:
        log.info("Detecting language (this takes a few minutes)...")
        sample_size = min(5000, len(unified))
        sample_idx  = unified.sample(sample_size, random_state=42).index
        unified.loc[sample_idx, "language"] = unified.loc[sample_idx, "text"].apply(
            detect_language
        )
        unified["language"] = unified["language"].fillna("en")
    else:
        unified["language"] = "unknown"

    unified["token_count"] = unified["text"].apply(token_count)

    def truncate_to_1024(text):
        tokens = text.split()
        if len(tokens) > 1024:
            return " ".join(tokens[:1024])
        return text

    long_count = (unified["token_count"] > 1024).sum()
    if long_count > 0:
        log.info(f"Truncating {long_count} texts to 1024 tokens...")
        unified["text"] = unified["text"].apply(truncate_to_1024)
        unified["token_count"] = unified["text"].apply(token_count)

    before = len(unified)
    unified = unified.drop_duplicates(subset=["text"], keep="first")
    log.info(f"Deduplication removed {before - len(unified)} duplicate texts")

    unified = unified.reset_index(drop=True)
    unified["id"] = unified.index

    log.info(f"\nFinal unified dataset: {len(unified)} rows")
    return unified

def create_splits(df: pd.DataFrame) -> dict:
    log.info("\n=== Creating stratified splits (60/20/20) ===")
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["label"]
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for name, sdf in splits.items():
        log.info(f"  {name:5s}: {len(sdf):5d} rows | "
                 + " | ".join(f"{l}: {n}" for l, n in sdf["label"].value_counts().items()))

    indices = {k: v.index.tolist() for k, v in splits.items()}
    idx_path = os.path.join(PATHS["output"], "split_indices.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(indices, f)
    log.info(f"  Split indices saved to {idx_path}")

    return splits

def analyse_imbalance(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts()
    total  = len(df)
    log.info("\n=== Class Distribution ===")
    for label, count in counts.items():
        log.info(f"  {label:12s}: {count:5d} ({count/total*100:.1f}%)")
    majority = counts.max()
    ratios   = {lbl: majority / cnt for lbl, cnt in counts.items()}
    log.info("\n  Inverse class weights for focal loss / XGBoost:")
    for lbl, weight in ratios.items():
        log.info(f"  {lbl:12s}: {weight:.3f}")
    min_ratio = min(counts.values) / max(counts.values)
    gamma_rec = 2 if min_ratio < 0.4 else 1
    log.info(f"\n  Recommended focal loss gamma: {gamma_rec}")
    log.info(f"  Min/max class ratio: {min_ratio:.3f}")
    return {"counts": counts.to_dict(), "weights": ratios, "focal_gamma": gamma_rec}

def generate_audit_report(unified: pd.DataFrame, splits: dict, nela: pd.DataFrame, imbalance: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("AI DISINFO SHIELD — PHASE 1 AUDIT REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("\n[1] UNIFIED DATASET SUMMARY")
    lines.append(f"  Total rows        : {len(unified)}")
    lines.append(f"  Unique sources    : {unified['source_dataset'].nunique()}")
    for src, cnt in unified["source_dataset"].value_counts().items():
        lines.append(f"    {src:30s}: {cnt}")
    lines.append("\n[2] LABEL DISTRIBUTION")
    for lbl, cnt in imbalance["counts"].items():
        lines.append(f"  {lbl:12s}: {cnt:5d}  (weight={imbalance['weights'][lbl]:.3f})")
    lines.append(f"  Recommended focal gamma: {imbalance['focal_gamma']}")
    lines.append("\n[3] TEXT STATISTICS")
    lines.append(f"  Avg token count   : {unified['token_count'].mean():.1f}")
    lines.append(f"  Median token count: {unified['token_count'].median():.1f}")
    lines.append(f"  Max token count   : {unified['token_count'].max()}")
    lines.append(f"  Texts > 512 tok   : {(unified['token_count'] > 512).sum()}")
    lines.append(f"  Texts > 1024 tok  : {(unified['token_count'] > 1024).sum()}")
    lines.append("\n[4] MULTIMODAL COVERAGE")
    has_image = unified["image_path"].notna().sum()
    lines.append(f"  Rows with image   : {has_image} ({has_image/len(unified)*100:.1f}%)")
    lines.append("\n[5] SPLIT SIZES")
    for name, sdf in splits.items():
        lines.append(f"  {name:5s}: {len(sdf)}")
    lines.append("\n[6] NULL CHECK (unified dataset)")
    for col in ["text", "label", "source_domain", "source_dataset"]:
        n = unified[col].isna().sum()
        lines.append(f"  {col:20s}: {n} nulls")
    lines.append("\n[7] SOURCERANK DATASET (NELA)")
    if not nela.empty:
        lines.append(f"  Total sources     : {len(nela)}")
        if "credibility_score" in nela.columns:
            for score, cnt in nela["credibility_score"].value_counts().sort_index().items():
                label = {0: "Low", 1: "Mixed", 2: "High"}.get(score, str(score))
                lines.append(f"  {label:8s} (score={score}): {cnt}")
    else:
        lines.append("  NELA not loaded.")
    lines.append("\n[8] RECOMMENDED NEXT STEPS (Phase 2)")
    lines.append("  - Use unified_dataset.csv for Stages 2, 4, 5 training")
    lines.append("  - Use nela_sourcerank.csv for Stage 3 (XGBoost) training")
    lines.append("  - image_path column → Phase 4 (multimodal)")
    lines.append("  - focal_gamma from above → Phase 3 training config")
    lines.append("  - split_indices.json → must stay fixed for all experiments")
    report = "\n".join(lines)
    report_path = os.path.join(PATHS["output"], "audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"\nAudit report saved to {report_path}")
    return report

def save_outputs(unified: pd.DataFrame, splits: dict, nela: pd.DataFrame):
    out = PATHS["output"]
    unified_path = os.path.join(out, "unified_dataset.csv")
    unified.to_csv(unified_path, index=False)
    log.info(f"Saved: {unified_path}")
    for name, sdf in splits.items():
        path = os.path.join(out, f"claims_{name}.csv")
        sdf.to_csv(path, index=False)
        log.info(f"Saved: {path}")
    if not nela.empty:
        nela_path = os.path.join(out, "nela_sourcerank.csv")
        nela.to_csv(nela_path, index=False)
        log.info(f"Saved: {nela_path}")
    log.info(f"\n✅ All Phase 1 outputs saved to: {out}")

def sanity_check(splits: dict):
    log.info("\n=== SANITY CHECK — 3 random examples per split ===")
    for name, sdf in splits.items():
        log.info(f"\n── {name.upper()} ──")
        sample = sdf.sample(min(3, len(sdf)), random_state=0)
        for _, row in sample.iterrows():
            log.info(f"  [{row['label']}] {row['text'][:120]}...")
            log.info(f"  Source: {row['source_dataset']} | Tokens: {row['token_count']}")

def main():
    log.info("╔══════════════════════════════════════════╗")
    log.info("║  AI DISINFO SHIELD — PHASE 1 PIPELINE   ║")
    log.info("╚══════════════════════════════════════════╝\n")
    unified = build_unified_dataset()
    imbalance = analyse_imbalance(unified)
    splits = create_splits(unified)
    nela = load_nela()
    save_outputs(unified, splits, nela)
    sanity_check(splits)
    report = generate_audit_report(unified, splits, nela, imbalance)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\n" + report.encode('ascii', errors='replace').decode('ascii'))
    log.info("\n✅ Phase 1 complete. Check audit_report.txt for next steps.")
    return unified, splits, nela, imbalance

if __name__ == "__main__":
    main()
