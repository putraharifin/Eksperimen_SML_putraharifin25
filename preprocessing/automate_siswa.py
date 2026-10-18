#!/usr/bin/env python3
"""Otomatisasi preprocessing AI4I 2020 Predictive Maintenance (konversi dari Eksperimen_siswa.ipynb).

Langkah: drop UDI & Product ID -> drop kolom leakage (TWF, HDF, PWF, OSF, RNF)
-> OrdinalEncoder 'Type' (L=0, M=1, H=2) -> StandardScaler pada 5 sensor numerik.

Output di --output-dir: ai4i2020_preprocessing.csv dan scaler.joblib

Contoh:
    python preprocessing/automate_siswa.py \
        --input ai4i2020_raw/ai4i2020.csv \
        --output-dir preprocessing/ai4i2020_preprocessing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

LOGGER = logging.getLogger("automate_siswa")

TARGET_COL = "Machine failure"
ID_COLS = ["UDI", "Product ID"]
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_COL = "Type"
TYPE_ORDER = ["L", "M", "H"]  # L=0, M=1, H=2
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_ORDER = [CATEGORICAL_COL, *NUMERIC_FEATURES]
REQUIRED_COLUMNS = [*ID_COLS, CATEGORICAL_COL, *NUMERIC_FEATURES, TARGET_COL, *LEAKAGE_COLS]

OUTPUT_CSV_NAME = "ai4i2020_preprocessing.csv"
OUTPUT_SCALER_NAME = "scaler.joblib"


def load_data(path: Path) -> pd.DataFrame:
    """Baca CSV mentah dan validasi kolom wajib."""
    if not path.is_file():
        raise FileNotFoundError(f"File data mentah tidak ditemukan: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {missing}")
    LOGGER.info("Data dimuat: %d baris, %d kolom", *df.shape)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop UDI, Product ID, dan kolom leakage; buang baris dengan nilai kosong."""
    df = df.drop(columns=[*ID_COLS, *LEAKAGE_COLS])
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) != n_before:
        LOGGER.warning("%d baris dengan nilai kosong dibuang", n_before - len(df))
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """OrdinalEncoder untuk 'Type' dengan urutan eksplisit L < M < H."""
    encoder = OrdinalEncoder(categories=[TYPE_ORDER])
    df = df.copy()
    df[CATEGORICAL_COL] = encoder.fit_transform(df[[CATEGORICAL_COL]]).astype(int).ravel()
    return df


def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScaler pada seluruh sensor numerik."""
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERIC_FEATURES] = scaler.fit_transform(df[NUMERIC_FEATURES])
    return df, scaler


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Pipeline lengkap: data mentah -> data siap latih + scaler."""
    df = drop_unused_columns(df)
    df = encode_type(df)
    df, scaler = scale_features(df)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df[[*FEATURE_ORDER, TARGET_COL]], scaler


def run(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / OUTPUT_CSV_NAME
    out_scaler = output_dir / OUTPUT_SCALER_NAME

    processed, scaler = preprocess(load_data(input_path))
    processed.to_csv(out_csv, index=False)
    joblib.dump(scaler, out_scaler)
    LOGGER.info("Dataset siap latih : %s (%d baris, %d kolom)", out_csv, *processed.shape)
    LOGGER.info("Scaler disimpan    : %s", out_scaler)
    return out_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocessing otomatis AI4I 2020 Predictive Maintenance.")
    p.add_argument("--input", "-i", type=Path, required=True, help="Path CSV data mentah.")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("preprocessing/ai4i2020_preprocessing"),
                   help="Folder output (ai4i2020_preprocessing.csv + scaler.joblib).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        run(args.input, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
