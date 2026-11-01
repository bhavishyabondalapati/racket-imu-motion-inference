"""Build docs/DATASET.md from the loaded dataset. Every number comes from the files."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from racketimu import io

CITATION = (
    "Sahar S. Tabrizi, Vajiheh Javani, Saeid Pashazadeh. *Data Acquired by a Single Object "
    "Sensor (IMU) of Table Tennis Forehand Strokes.* Mendeley Data, Version 3, 2020. "
    "DOI: [10.17632/b7bc9y232m.3](https://doi.org/10.17632/b7bc9y232m.3). "
    "Licence: CC BY 4.0."
)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub Markdown table (no extra dependency)."""
    frame = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.to_numpy()]
    return "\n".join([head, rule, *body])


def id_ranges(ids: Iterable[int]) -> str:
    """Compress ids into ranges: [1, 2, 3, 7] -> "1-3, 7"."""
    ids = sorted(int(i) for i in ids)
    if not ids:
        return "none"
    runs, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def rows_per_whole_second(strokes: Iterable[pd.DataFrame]) -> np.ndarray:
    """Rows logged within each whole second, skipping each file's first and last second.

    Timestamps have 1-second resolution, so this is the finest rate measure available.
    Files whose timestamps stop at minutes are skipped.
    """
    counts: list[int] = []
    for stroke in strokes:
        if len(stroke) == 0 or not stroke["time_has_seconds"].all():
            continue
        per_second = stroke["time"].value_counts(sort=False).sort_index()
        counts.extend(per_second.iloc[1:-1].tolist())
    return np.array(counts, dtype=int)


def processed_matches_raw(files: io.DatasetFiles, index: pd.DataFrame) -> dict[str, Any]:
    """Check that each processed file is the raw rows, zero-padded at the front."""
    raw_cols = list(io.RAW_SENSOR_COLUMNS)
    result = {"checked": 0, "equal": 0, "padded_to_70": 0, "longer_than_70": 0, "mismatch": []}
    result["replaced"] = []  # raw acc+gyro all zero, processed has different, non-zero rows
    imu_cols = [c for c, name in io.RAW_SENSOR_COLUMNS.items() if name in io.ACC + io.GYRO]
    authors = index[index["in_authors_set"]]
    for sample_id, folder_stroke in zip(
        authors["sample_id"], authors["folder_stroke"], strict=True
    ):
        folder = next(f for f, s in io.PROCESSED_FOLDERS.items() if s == folder_stroke)
        processed = pd.read_csv(files.processed_dir / folder / f"{sample_id}.csv")
        raw = io.read_raw_table(files.stroke_file(sample_id))
        nonzero = processed[(processed[raw_cols] != 0).any(axis=1)]
        leading_pad = len(processed) - len(nonzero)
        same = len(nonzero) == len(raw) and np.allclose(
            nonzero[raw_cols].to_numpy(float), raw[raw_cols].to_numpy(float)
        )
        front_only = (processed[raw_cols].iloc[:leading_pad] == 0).all().all()
        result["checked"] += 1
        result["equal"] += bool(same and front_only)
        result["padded_to_70"] += len(processed) == 70
        result["longer_than_70"] += len(processed) > 70
        if not (same and front_only):
            result["mismatch"].append(int(sample_id))
            if not raw[imu_cols].to_numpy().any() and nonzero[imu_cols].to_numpy().any():
                result["replaced"].append(int(sample_id))
    return result


def _counts_by_player(index: pd.DataFrame, excluded_test: list[str]) -> pd.DataFrame:
    inc = index[index["included"]]
    table = pd.crosstab(inc["player"], inc["stroke"]).reindex(
        columns=list(io.STROKES), fill_value=0
    )
    players = sorted(index.loc[index["in_authors_set"], "player"].unique())
    table = table.reindex(players, fill_value=0)
    table["total used"] = table.sum(axis=1)
    authors = index[index["in_authors_set"]].groupby("player").size()
    table.insert(0, "level", [io.level_of(p) for p in table.index])
    table.insert(1, "in authors' set", authors.reindex(table.index).fillna(0).astype(int))
    table["classes used"] = [
        ", ".join(s for s in io.STROKES if table.loc[p, s] > 0) or "none" for p in table.index
    ]
    table["LOPO test player (acc/gyro)"] = [
        "no" if p in excluded_test or table.loc[p, "total used"] == 0 else "yes"
        for p in table.index
    ]
    return table.rename_axis("player")


def _mean_std(s: pd.Series) -> str:
    return "n/a (no samples)" if s.empty else f"{s.mean():.1f} ± {s.std():.1f}"


def _score_summary(inc: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in [*io.CRITERIA, "score_mean"]:
        s = inc[col]
        rows[col] = {
            "mean ± std": _mean_std(s),
            "median": f"{s.median():.1f}",
            "min-max": f"{s.min():.0f}-{s.max():.0f}",
            "novice mean ± std": _mean_std(s[inc["level"] == "novice"]),
            "pro mean ± std": _mean_std(s[inc["level"] == "pro"]),
        }
    return pd.DataFrame(rows).T.rename_axis("score")


def _reconciliation(index: pd.DataFrame) -> pd.DataFrame:
    reasons = index["exclusion_reason"].value_counts()
    n_clean_csv = int(index["in_clean_scores_csv"].sum())
    n_authors = int(index["in_authors_set"].sum())
    steps = [
        ("Raw stroke files (= rows in Samples-Scores.csv)", len(index)),
        ("removed by the authors", -int(reasons.get(io.REASON_REMOVED_BY_AUTHORS, 0))),
        ("Rows in the authors' Clean-Samples-Scores.csv", n_clean_csv),
        ("in the clean CSV only (no processed file, not in clean spreadsheet)",
         -int(reasons.get(io.REASON_CLEAN_CSV_ONLY, 0))),
        ("**Authors' cleaned set** (processed files; the description says 1570)", n_authors),
        ("raw file has no data rows", -int(reasons.get(io.REASON_EMPTY, 0))),
        ("accelerometer and gyroscope zero in every row",
         -int(reasons.get(io.REASON_ACC_GYRO_ZERO, 0))),
        ("**Used in this project**", int(index["included"].sum())),
    ]  # fmt: skip
    return pd.DataFrame(steps, columns=["step", "samples"])


def _exclusion_section(index: pd.DataFrame) -> list[str]:
    lines = []
    excluded = index[~index["included"]]
    order = [io.REASON_REMOVED_BY_AUTHORS, io.REASON_CLEAN_CSV_ONLY, io.REASON_EMPTY,
             io.REASON_ACC_GYRO_ZERO]  # fmt: skip
    for reason in order:
        group = excluded[excluded["exclusion_reason"] == reason]
        if group.empty:
            continue
        breakdown = group.groupby(["player", "stroke"]).size().rename("samples")
        lines += [
            f"### {reason[0].upper()}{reason[1:]} ({len(group)} sample{'s' * (len(group) != 1)})",
            "",
            f"Sample ids: {id_ranges(group['sample_id'])}",
            "",
            md_table(breakdown.reset_index(), index=False),
            "",
        ]
    return lines


def _sampling_section(
    inc: pd.DataFrame, strokes: dict[int, pd.DataFrame], stated: dict
) -> list[str]:
    used = [strokes[i] for i in inc["sample_id"]]
    rps = rows_per_whole_second(used)
    n_rows = inc["n_rows"]
    pooled_dups = sum(
        int(np.all(s[io.CHANNELS].to_numpy()[1:] == s[io.CHANNELS].to_numpy()[:-1], axis=1).sum())
        for s in used
    )
    pooled_pairs = int((n_rows - 1).clip(lower=0).sum())
    minute_only = sum(not s["time_has_seconds"].all() for s in used)
    stated_text = ", ".join(f"{hz} Hz ({name})" for name, hz in stated.items())
    observed = (
        "not computable (no whole seconds with second-resolution timestamps)"
        if len(rps) == 0
        else f"mean {rps.mean():.2f}, std {rps.std():.2f}, min {rps.min()}, max {rps.max()} "
        f"(n = {len(rps)} whole seconds)"
    )
    return [
        f"- **Stated by the authors:** {stated_text}.",
        f"- **Observed rows per whole second:** {observed}. Timestamps have "
        "1-second resolution, so this is the finest timing available; each file's first and "
        f"last (partial) second are skipped, and {minute_only} files whose timestamps stop at "
        "minutes are skipped.",
        f"- **Rows per stroke:** median {n_rows.median():.0f}, 5th-95th percentile "
        f"{n_rows.quantile(0.05):.0f}-{n_rows.quantile(0.95):.0f}, min {n_rows.min()}, "
        f"max {n_rows.max()}.",
        f"- **Exact duplicate rows** (a row identical to the previous one in every channel): "
        f"{pooled_dups} of {pooled_pairs} consecutive pairs ({pooled_dups / pooled_pairs:.1%}). "
        f"Per stroke: median {inc['duplicate_rate'].median():.1%}. Duplicates are kept.",
    ]


def _signal_section(inc: pd.DataFrame, strokes: dict[int, pd.DataFrame]) -> list[str]:
    """Peak rotation rate, acceleration magnitude, and peak-on-first-row share (used set)."""
    used = [strokes[i] for i in inc["sample_id"]]
    gyro = [np.linalg.norm(s[io.GYRO].to_numpy(), axis=1) for s in used]
    peaks = np.array([g.max() for g in gyro])
    first = np.array([g.argmax() == 0 for g in gyro])
    acc = np.concatenate([np.linalg.norm(s[io.ACC].to_numpy(), axis=1) for s in used])
    p1, p99 = np.percentile(acc, [1, 99])
    return [
        f"- **Peak rotation rate per stroke:** median {np.median(peaks):.2f} rad/s "
        f"({np.degrees(np.median(peaks)):.0f} deg/s), 95th percentile "
        f"{np.percentile(peaks, 95):.2f} rad/s, max {peaks.max():.2f} rad/s.",
        f"- **Acceleration magnitude, all rows:** median {np.median(acc):.2f} m/s², "
        f"1st-99th percentile {p1:.2f}-{p99:.2f} m/s², max {acc.max():.2f} m/s². "
        "Gravity alone is 9.81 m/s².",
        f"- **Peak on the first row:** {first.sum()} of {len(first)} strokes "
        f"({first.mean():.1%}) have their largest rotation rate on the first logged row.",
        "",
        "**Interpretation (supported by these numbers, not certain):** a modest peak rotation "
        "rate and an acceleration magnitude that stays close to gravity, at about 8 rows per "
        "second, suggest that the logging rate undersamples the fast phase of the stroke. The "
        "signal then mostly reflects the racket's orientation rather than swing dynamics. "
        "Strokes whose peak is on the first row are a likely sign that recordings sometimes "
        "start mid-swing.",
    ]


def _checks_section(index: pd.DataFrame, processed: dict[str, Any], dates: list[str]) -> list[str]:
    authors = index[index["in_authors_set"]]
    label_mismatch = int((authors["folder_stroke"] != authors["stroke"]).sum())
    overridden = index[index["player_overridden"]]
    return [
        f"- Stroke labels: the scores CSV uses several spellings (e.g. `BASIC `, `Basic `, "
        f"`Top`, `push`). After normalizing to {', '.join(io.STROKES)}, "
        f"{label_mismatch} of {len(authors)} labels disagree with the processed folder.",
        f"- Processed files are the raw rows zero-padded at the front: {processed['equal']} of "
        f"{processed['checked']} match ({processed['padded_to_70']} are 70 rows long, "
        f"{processed['longer_than_70']} are longer and not padded). Mismatches: "
        f"{id_ranges(processed['mismatch'])}.",
        f"- For {len(processed['replaced'])} of those ids ({id_ranges(processed['replaced'])}) "
        "the raw file has accelerometer and gyroscope zero in every row, while the processed "
        "file holds different rows with non-zero accelerometer and gyroscope data and a "
        "different first timestamp. Following the approved rule (rows come from the raw files), "
        "these samples are excluded as zero-sensor files; the processed versions are not used.",
        "- Player overrides: "
        + ", ".join(f"id {r.sample_id} -> {r.player}" for r in overridden.itertuples())
        + " (scores CSV says P06; the spreadsheet range 651-720 says P05).",
        "- Player codes: the CSV writes `NO1`/`PO1` (letter O); the spreadsheets write "
        "`#N01`/`#P01`. Both are normalized to `N01`/`P01`.",
        f"- Recording dates (from the timestamps): {', '.join(dates)}.",
    ]


def build_dataset_card(ds: io.Dataset, files: io.DatasetFiles, cfg: dict[str, Any]) -> str:
    """Return the full Markdown text of docs/DATASET.md."""
    index, inc = ds.index, ds.included()
    excluded_test = list(cfg["lopo"]["excluded_test_players"])
    by_level = pd.crosstab(inc["level"], inc["stroke"], margins=True, margins_name="total")
    corr = inc[io.CRITERIA].corr().map(lambda v: f"{v:.2f}")
    dates = sorted(
        {str(t.date()) for i in inc["sample_id"] for t in ds.strokes[i]["time"].dropna()[:1]}
    )
    n_players = index.loc[index["in_authors_set"], "player"].nunique()
    n_used_players = inc["player"].nunique()
    lines = [
        "# Dataset card: Tabrizi et al. (2020)",
        "",
        "> Generated by `make inspect` (`scripts/inspect_dataset.py`) from the files in "
        f"`{cfg['dataset']['root']}/`. Do not edit by hand.",
        "",
        "## Source", "", CITATION, "",
        "- One BNO055 9-axis IMU on the racket face; 16 players (8 novice, 8 professional); "
        "Basic, Topspin and Push forehands.",
        "- Strokes are shadow-play, without a ball (from the authors' papers).",
        "- C1-C5 are five evaluation criteria. Each value is the average of 3 coaches; individual "
        "coach scores are not published.",
        "",
        "## Counts and reconciliation", "", md_table(_reconciliation(index), index=False), "",
        f"- The authors' cleaned set has {index['in_authors_set'].sum()} samples from "
        f"{n_players} players, matching the description (1570 samples, 16 players).",
        "- The 2021 Computational Intelligence and Neuroscience paper's tables list 1525 samples "
        "per criterion. That differs from the cleaned set by "
        f"{index['in_authors_set'].sum() - 1525} samples; the files do not say which samples "
        "that paper left out, so the difference cannot be reconciled here.",
        f"- After exclusions, {n_used_players} players have usable samples. Models that use "
        f"accelerometer or gyroscope data run {n_used_players} LOPO folds "
        f"(not used as test players: {', '.join(excluded_test)}).",
        "",
        "## Samples per player (used set)", "",
        md_table(_counts_by_player(index, excluded_test)), "",
        "## Samples per level and class (used set)", "",
        md_table(by_level.rename_axis("level")), "",
        "## Score distributions (used set)", "",
        "Scores are 0-100. `score_mean` is the mean of C1-C5.", "",
        md_table(_score_summary(inc)), "",
        "Correlation between criteria (Pearson, used set). These are correlations between "
        "criteria, not between coaches:", "",
        md_table(corr.rename_axis("")), "",
        "**Agreement between coaches cannot be computed**: only the average of the 3 coaches is "
        "published for each criterion, so there is no coach-agreement ceiling for quality "
        "prediction.", "",
        "## Sampling rate", "",
        *_sampling_section(inc, ds.strokes, cfg["dataset"]["stated_rates_hz"]), "",
        "## Signal characteristics (used set)", "", *_signal_section(inc, ds.strokes), "",
        "## Channels and units", "",
        "| channels | unit | conversion from the raw integers |", "|---|---|---|",
        "| acc x, y, z | m/s² | raw / 100 (BNO055 default; median magnitude at rest ≈ 9.81) |",
        "| gyro x, y, z | rad/s | raw / 16 = deg/s, then to rad/s |",
        "| mag x, y, z | µT | raw / 16 |",
        "| roll, pitch, yaw | degrees | as stored; the Euler angle order is not documented |",
        "",
        "## Data checks", "", *_checks_section(index, processed_matches_raw(files, index), dates),
        "",
        "## Exclusions", "",
        f"{(~index['included']).sum()} of {len(index)} raw samples are not used. Each is listed "
        "under the first rule that excludes it.", "",
        *_exclusion_section(index),
    ]  # fmt: skip
    return "\n".join(lines).rstrip() + "\n"
