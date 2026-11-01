"""Load the Tabrizi et al. (2020) table tennis IMU dataset from its raw files.

The dataset ships one CSV per stroke (`Raw/Sensory-Data/<id>.csv`), a scores CSV,
and the authors' cleaned set (`Processed/`, the same rows zero-padded to 70).
`load_dataset` reads every raw file once, converts units, attaches labels, scores
and level, and marks each sample as included or excluded with a reason.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_SENSOR_COLUMNS = {
    "Roll": "roll_deg", "Pitch": "pitch_deg", "Yaw": "yaw_deg",
    "acc(x)": "acc_x", "acc(y)": "acc_y", "acc(z)": "acc_z",
    "gyro(x)": "gyro_x", "gyro(y)": "gyro_y", "gyro(z)": "gyro_z",
    "mag(x)": "mag_x", "mag(y)": "mag_y", "mag(z)": "mag_z",
}  # fmt: skip
ACC = ["acc_x", "acc_y", "acc_z"]  # m/s^2, sensor frame
GYRO = ["gyro_x", "gyro_y", "gyro_z"]  # rad/s, sensor frame
EULER = ["roll_deg", "pitch_deg", "yaw_deg"]  # degrees, as reported by the BNO055
MAG = ["mag_x", "mag_y", "mag_z"]  # microtesla
CHANNELS = ACC + GYRO + EULER + MAG
CRITERIA = ["C1", "C2", "C3", "C4", "C5"]
STROKES = ("BASIC", "PUSH", "TOPSPIN")
PROCESSED_FOLDERS = {"Basic": "BASIC", "PUSH": "PUSH", "TOPSPIN": "TOPSPIN"}

REASON_REMOVED_BY_AUTHORS = "removed by the authors (not in their cleaned set)"
REASON_CLEAN_CSV_ONLY = (
    "listed in the authors' clean scores CSV, but has no processed file and is "
    "absent from the clean spreadsheet, so not part of their cleaned set"
)
REASON_EMPTY = "raw file has no data rows"
REASON_ACC_GYRO_ZERO = "accelerometer and gyroscope are zero in every row"

_STROKE_ALIASES = {"BASIC": "BASIC", "PUSH": "PUSH", "TOP": "TOPSPIN", "TOPSPIN": "TOPSPIN"}
_TIMESTAMP = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2})时(\d{1,2})分(?:(\d{1,2})秒)?")


@dataclass(frozen=True)
class DatasetFiles:
    """Paths inside the unzipped dataset folder (`data/raw/tabrizi2020/`)."""

    root: Path

    @property
    def sensor_dir(self) -> Path:
        return self.root / "Raw" / "Sensory-Data"

    @property
    def raw_scores(self) -> Path:
        return self.root / "Raw" / "Samples-Scores" / "Samples-Scores.csv"

    @property
    def clean_scores(self) -> Path:
        return self.root / "Processed" / "Clean-Samples-Scores" / "Clean-Samples-Scores.csv"

    @property
    def processed_dir(self) -> Path:
        return self.root / "Processed"

    def stroke_file(self, sample_id: int) -> Path:
        return self.sensor_dir / f"{sample_id}.csv"


@dataclass
class Dataset:
    """One row per raw sample id in `index`; converted sensor rows per id in `strokes`."""

    index: pd.DataFrame
    strokes: dict[int, pd.DataFrame]

    def included(self) -> pd.DataFrame:
        return self.index[self.index["included"]]


def normalize_stroke(label: str) -> str:
    """Map the 8 spellings in the scores CSV ("BASIC ", "Top", "push", ...) to 3 labels."""
    key = label.strip().upper()
    if key not in _STROKE_ALIASES:
        raise ValueError(f"Unknown stroke label: {label!r}")
    return _STROKE_ALIASES[key]


def normalize_player(code: str) -> str:
    """Map "NO1" (CSV, letter O) and "#N01 " (spreadsheet) to "N01"."""
    text = code.strip().lstrip("#").upper()
    group, number = text[:1], text[1:].replace("O", "0")
    if group not in ("N", "P") or not number.isdigit():
        raise ValueError(f"Unknown player code: {code!r}")
    return f"{group}{int(number):02d}"


def level_of(player: str) -> str:
    return "novice" if player.startswith("N") else "pro"


def decode_timestamp(raw: str) -> tuple[datetime | None, bool]:
    """Decode a `Time/Date` cell; return (time, whether it includes seconds).

    Cells look like "b'2019\\xc3\\x84\\xc3\\xaa9...'": the Python repr of bytes that are
    GBK-encoded Chinese date text, mis-decoded as Latin-1 and re-encoded as UTF-8.
    Decoding gives e.g. "2019年9月17日10时43分20秒". Some cells stop at minutes.
    """
    try:
        text = ast.literal_eval(raw).decode("utf-8").encode("latin-1").decode("gbk")
    except (ValueError, SyntaxError, UnicodeError, AttributeError):
        return None, False
    match = _TIMESTAMP.fullmatch(text.strip())
    if match is None:
        return None, False
    year, month, day, hour, minute, second = match.groups()
    parts = [int(year), int(month), int(day), int(hour), int(minute), int(second or 0)]
    return datetime(*parts), second is not None


def read_raw_table(path: Path) -> pd.DataFrame:
    """Read one raw stroke CSV exactly as stored (raw integers, undecoded timestamps)."""
    return pd.read_csv(path)


def to_physical(raw: pd.DataFrame, units: dict[str, float]) -> pd.DataFrame:
    """Convert a raw stroke table to physical units and decode its timestamps.

    acc -> m/s^2, gyro -> rad/s, mag -> microtesla, Euler angles stay in degrees.
    """
    out = raw.rename(columns=RAW_SENSOR_COLUMNS)[CHANNELS].astype(float)
    out[ACC] /= units["acc_lsb_per_m_s2"]
    out[GYRO] = np.deg2rad(out[GYRO] / units["gyro_lsb_per_dps"])
    out[MAG] /= units["mag_lsb_per_ut"]
    decoded = {s: decode_timestamp(s) for s in raw["Time/Date"].unique()}
    out.insert(0, "time", pd.to_datetime(raw["Time/Date"].map(lambda s: decoded[s][0])))
    out.insert(1, "time_has_seconds", raw["Time/Date"].map(lambda s: decoded[s][1]).astype(bool))
    return out.reset_index(drop=True)


def stroke_quality(stroke: pd.DataFrame) -> dict[str, Any]:
    """Row count, emptiness, all-zero acc+gyro, and the rate of exact repeated rows."""
    n = len(stroke)
    values = stroke[CHANNELS].to_numpy()
    zero = bool(n) and not np.any(values[:, : len(ACC) + len(GYRO)])
    dup_rate = float(np.mean(np.all(values[1:] == values[:-1], axis=1))) if n > 1 else np.nan
    return {"n_rows": n, "empty": n == 0, "acc_gyro_zero": zero, "duplicate_rate": dup_rate}


def read_scores(path: Path, player_overrides: dict[int, str] | None = None) -> pd.DataFrame:
    """Read a scores CSV into tidy columns, applying documented player overrides."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"sample_id": df["#"].astype(int)})
    out["player"] = df["Player"].map(normalize_player)
    overrides = {int(k): normalize_player(v) for k, v in (player_overrides or {}).items()}
    corrected = [overrides.get(s, p) for s, p in zip(out["sample_id"], out["player"], strict=True)]
    out["player_overridden"] = out["player"] != corrected
    out["player"] = corrected
    out["level"] = out["player"].map(level_of)
    out["stroke"] = df["Stroke"].map(normalize_stroke)
    out[CRITERIA] = df[CRITERIA].astype(float)
    out["score_mean"] = out[CRITERIA].mean(axis=1)
    return out


def authors_clean_ids(files: DatasetFiles) -> pd.Series:
    """Sample id -> stroke label from the folder it sits in, for the authors' cleaned set."""
    labels = {
        int(p.stem): label
        for folder, label in PROCESSED_FOLDERS.items()
        for p in (files.processed_dir / folder).glob("*.csv")
    }
    return pd.Series(labels, name="folder_stroke").sort_index()


def exclusion_reason(row: pd.Series) -> str | None:
    """The first rule that excludes a sample, in priority order, or None if it is kept."""
    if not row["in_authors_set"]:
        return REASON_CLEAN_CSV_ONLY if row["in_clean_scores_csv"] else REASON_REMOVED_BY_AUTHORS
    if row["empty"]:
        return REASON_EMPTY
    if row["acc_gyro_zero"]:
        return REASON_ACC_GYRO_ZERO
    return None


def load_dataset(root: str | Path, dataset_cfg: dict[str, Any]) -> Dataset:
    """Read every raw stroke, attach labels and scores, and mark inclusion with reasons."""
    files = DatasetFiles(Path(root))
    if not files.sensor_dir.is_dir():
        raise FileNotFoundError(
            f"No raw sensor folder at {files.sensor_dir}. Download the dataset and unzip "
            "Raw.zip and Processed.zip into the dataset root (see README)."
        )
    index = read_scores(files.raw_scores, dataset_cfg.get("player_overrides"))
    index["in_clean_scores_csv"] = index["sample_id"].isin(
        read_scores(files.clean_scores)["sample_id"]
    )
    folder = authors_clean_ids(files)
    index["in_authors_set"] = index["sample_id"].isin(folder.index)
    index["folder_stroke"] = index["sample_id"].map(folder)

    strokes, quality = {}, []
    for sample_id in index["sample_id"]:
        stroke = to_physical(read_raw_table(files.stroke_file(sample_id)), dataset_cfg["units"])
        strokes[int(sample_id)] = stroke
        quality.append(stroke_quality(stroke))
    index = pd.concat([index, pd.DataFrame(quality, index=index.index)], axis=1)
    index["exclusion_reason"] = index.apply(exclusion_reason, axis=1)
    index["included"] = index["exclusion_reason"].isna()
    return Dataset(index=index.sort_values("sample_id").reset_index(drop=True), strokes=strokes)
