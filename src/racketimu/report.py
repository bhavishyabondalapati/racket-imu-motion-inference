"""D9: build results/SUMMARY.md from the saved result files only (latest run of each milestone)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from racketimu.classify import CLASSES
from racketimu.classify_report import MODEL_LABELS
from racketimu.datacard import md_table
from racketimu.errors import confusion_pairs, per_player, seed_averaged


def latest(results: Path, prefix: str) -> Path | None:
    runs = sorted(results.glob(f"{prefix}_*"))
    return runs[-1] if runs else None


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _pm(s: pd.Series, fmt: str = ".3f") -> str:
    return f"{s.mean():{fmt}}" if len(s) == 1 else f"{s.mean():{fmt}} ± {s.std():{fmt}}"


def runs_section(runs: dict[str, Path | None]) -> list[str]:
    rows = []
    for name, run in runs.items():
        if run is None:
            rows.append({"milestone": name, "run": "not run", "commit": "-", "clean": "-"})
            continue
        info = _json(run / "run.json")
        rows.append(
            {
                "milestone": name,
                "run": f"`{run.name}`",
                "commit": f"`{info['commit'][:10]}`",
                "clean": "yes" if not info["dirty"] else "no",
            }
        )
    return ["## Runs used", "", md_table(pd.DataFrame(rows), index=False), ""]


def dataset_section(d3: Path) -> list[str]:
    oof = pd.read_csv(d3 / "oof_predictions.csv.gz")
    s = oof[oof["model"] == "majority"].assign(
        stroke=lambda d: d["y_true"].map(dict(enumerate(CLASSES)))
    )
    table = pd.crosstab(s["level"], s["stroke"], margins=True, margins_name="total")
    return [
        "## Dataset",
        "",
        f"Tabrizi et al. (2020), table tennis forehands from one racket-mounted IMU. Used: "
        f"{len(s)} strokes from {s['test_player'].nunique()} players (of the authors' 1570 "
        "cleaned samples from 16 players; every exclusion and its reason is in "
        "`docs/DATASET.md`). Observed logging rate about 8 rows per second.",
        "",
        md_table(table.rename_axis("level")),
        "",
    ]


def d3_section(d3: Path) -> list[str]:
    m = _json(d3 / "metrics.json")
    ps = pd.DataFrame(m["per_seed"])
    g = ps.groupby("model", sort=False)
    table = pd.DataFrame(
        {
            "macro-F1": g["macro_f1"].apply(_pm),
            "accuracy": g["accuracy"].apply(_pm),
            "ECE (uncalibrated)": g["ece_raw"].apply(_pm),
            "ECE after T": g["ece_cal"].apply(_pm),
        }
    )
    table.index = [MODEL_LABELS.get(i, i) for i in table.index]
    c = m["checks"]
    return [
        "## Stroke type (D3)",
        "",
        "15-fold leave-one-player-out, pooled out-of-fold, mean ± std over 3 seeds.",
        "",
        md_table(table.rename_axis("model")),
        "",
        f"- Heading check: {c['heading_gain_mean']:+.3f} macro-F1 from yaw + magnetometer "
        f"({'flagged' if c['heading_flagged'] else 'not flagged'}, "
        f"rule ≥ {c['thresholds']['heading_gain_flag']}).",
        f"- Leak check: channel means alone are {c['mean_only_margin']:+.3f} above majority "
        f"({'**flagged**' if c['mean_only_flagged'] else 'not flagged'}, "
        f"rule ≥ {c['thresholds']['leak_margin_flag']}); "
        f"{c['mean_only_vs_full_rf']:+.3f} vs the full RF. Motion-only RF: "
        f"{c['motion_only_macro_f1']:.3f} ({c['motion_only_vs_full_rf']:+.3f} vs the full RF).",
        f"- Length check: {c['length_only_margin']:+.3f} above majority "
        f"({'flagged' if c['length_only_flagged'] else 'not flagged'}).",
        "- Calibration: temperature fitted on 2 inner validation players made ECE worse for every "
        "model, so uncalibrated probabilities are the primary result.",
        "",
    ]


def d6_section(d6: Path) -> list[str]:
    s = pd.DataFrame(_json(d6 / "metrics.json"))
    lines = ["## Sparsity ablation (D6)", ""]
    for model in sorted(s["model"].unique(), key=lambda m: m != "rf"):
        t = s[s["model"] == model].assign(label=lambda d: d["label"].str.replace("2th", "2nd"))
        lines += [
            f"**{'Random forest' if model == 'rf' else model.upper()}**",
            "",
            md_table(
                pd.DataFrame(
                    {
                        "condition": t["label"],
                        "macro-F1": [
                            f"{a:.3f} ± {b:.3f}"
                            for a, b in zip(t["f1_mean"], t["f1_std"], strict=True)
                        ],
                        "Δ vs full": [
                            f"{a:+.3f}" if c != "full" else "-"
                            for a, c in zip(t["delta_mean"], t["condition"], strict=True)
                        ],
                    }
                ),
                index=False,
            ),
            "",
        ]
    s = s.assign(label=s["label"].str.replace("2th", "2nd"))
    rf = s[(s["model"] == "rf") & (s["condition"] != "full")].sort_values("delta_mean")
    hurt = rf[rf["delta_mean"] < -2 * rf["delta_std"].clip(lower=0.005)]
    rest = rf.drop(hurt.index)
    lines += [
        "Reductions that hurt (RF, Δ below −2 × its seed std): "
        + ("; ".join(f"{r.label} ({r.delta_mean:+.3f})" for r in hurt.itertuples()) or "none")
        + f". All other reductions stay within {rest['delta_mean'].abs().max():.3f} of full data.",
        "",
    ]
    return lines


D4_NAMES = {
    "global_mean": "Global training mean",
    "class_mean": "Per-class training mean",
    "level_mean": "Level-only baseline",
    "rf": "RF regressor",
    "gru": "GRU",
    "cnn": "1D CNN",
}


def d4_section(d4: Path) -> list[str]:
    m = _json(d4 / "metrics.json")
    sc = pd.DataFrame(m["scores"])
    lines = ["## Stroke quality (D4)", ""]
    for subset, title in [
        ("all", "All test samples"),
        ("novice", "Novices only"),
        ("pro", "Pros only"),
    ]:
        t = sc[(sc["subset"] == subset) & (sc["target"] == "score_mean")]
        g = t.groupby("method", sort=False)
        lines += [
            f"**{title}, mean score (score_mean)**",
            "",
            md_table(
                pd.DataFrame(
                    {
                        "MAE": g["mae"].apply(lambda s: _pm(s, ".2f")),
                        "Spearman": g["spearman"].apply(
                            lambda s: "n/a" if s.isna().all() else _pm(s)
                        ),
                    }
                )
                .rename(index=D4_NAMES)
                .rename_axis("method")
            ),
            "",
        ]
    for model, verdict in m["beats_level_baseline"].items():
        yes = [t for t, ok in verdict.items() if ok]
        lines.append(
            f"- **{model.upper()}** beats the level-only baseline (pooled MAE lower in every "
            f"seed) on {len(yes)} of {len(verdict)} targets"
            + (f": {', '.join(yes)}." if yes else ".")
        )
    return [
        *lines,
        "- The negative Spearman of the mean baselines is a leave-one-out artifact (removing a "
        "low-scoring player raises the training mean predicted for that player); their MAE is "
        "the meaningful number.",
        "",
        "Full per-criterion tables: the D4 `report.md`.",
        "",
    ]


def _drops(cg: pd.DataFrame) -> list[str]:
    out = []
    for level, other in (("novice", "pro"), ("pro", "novice")):
        w = cg[(cg["test_level"] == level) & (cg["mode"] == "within-group LOPO")]
        c = cg[(cg["test_level"] == level) & (cg["mode"] == "cross-group")]
        f1 = c["type_macro_f1"].mean() - w["type_macro_f1"].mean()
        mae = c["mae_score_mean"].mean() - w["mae_score_mean"].mean()
        out.append(
            f"- Testing on {level}s, training on {other}s instead of other {level}s: "
            f"stroke-type macro-F1 {f1:+.3f}, score_mean MAE {mae:+.2f} points."
        )
    return out


def d5_section(d5: Path) -> list[str]:
    m = _json(d5 / "metrics.json")
    names = {"majority": "Majority", "rf": "RF", "gru": "GRU", "cnn": "1D CNN"}
    lv = pd.DataFrame(m["level"]).replace({"model": names})
    g = lv.groupby("model", sort=False)
    cg = pd.DataFrame(m["cross_group"])
    h = cg.groupby("setting", sort=False)
    return [
        "## Skill level and cross-group transfer (D5)",
        "",
        md_table(
            pd.DataFrame(
                {
                    "accuracy": g["accuracy"].apply(_pm),
                    "balanced accuracy": g["balanced_accuracy"].apply(_pm),
                    "macro-F1": g["macro_f1"].apply(_pm),
                }
            ).rename_axis("novice vs pro")
        ),
        "",
        md_table(
            pd.DataFrame(
                {
                    "mode": h["mode"].first(),
                    "stroke-type macro-F1": h["type_macro_f1"].apply(_pm),
                    "score_mean MAE": h["mae_score_mean"].apply(lambda s: _pm(s, ".2f")),
                    "score_mean Spearman": h["spearman_score_mean"].apply(_pm),
                }
            ).rename_axis("setting")
        ),
        "",
        *_drops(cg),
        "",
        "- Caveat: level is fully confounded with recording day (all novices recorded on "
        "2019-09-25, all pros on other days), so the novice-vs-pro result may partly reflect "
        "session differences.",
        "",
    ]


def d7_section(d3: Path) -> list[str]:
    df = seed_averaged(pd.read_csv(d3 / "oof_predictions.csv.gz"), "rf_primary")
    pairs, players = confusion_pairs(df), per_player(df)
    bt = pairs[pairs["true"].isin(["BASIC", "TOPSPIN"]) & pairs["pred"].isin(["BASIC", "TOPSPIN"])]
    return [
        "## Error analysis (D7)",
        "",
        f"- {int((~df['correct']).sum())} of {len(df)} strokes misclassified by the RF "
        "(seed-averaged); "
        f"basic/topspin confusions are {bt['errors'].sum()} of them.",
        f"- Per-player accuracy ranges from {players['accuracy'].min():.3f} "
        f"({players.index[0]}) to "
        f"{players['accuracy'].max():.3f}.",
        "- Details, the most confident errors, and failure plots: `docs/ERRORS.md`.",
        "",
    ]


def d8_section(d8: Path) -> list[str]:
    lat = _json(d8 / "latency.json")
    return [
        "## Inference latency (D8)",
        "",
        f"`racketimu.infer.infer` on {lat['cpu']} (CPU only, single thread): median "
        f"{lat['median_ms']:.2f} ms, p95 {lat['p95_ms']:.2f} ms per stroke over "
        f"{lat['n_runs']} runs.",
        "",
    ]


LIMITATIONS = [
    "Table tennis rather than tennis, forehands only, and shadow strokes without a ball (confirmed "
    "by the authors' papers).",
    "No synchronized video or motion capture, so body motion cannot be inferred or validated.",
    "16 players (15 with usable accelerometer and gyroscope data), one sensor placement on the "
    "racket face.",
    "A very low observed logging rate (about 8 rows per second versus 100 Hz / 70 Hz stated); "
    "the signal probably undersamples the fast phase of the stroke and mostly reflects racket "
    "orientation.",
    "Classification relies on static racket angle: channel means alone match the full models; "
    "a recording-block artifact cannot be ruled out because each stroke type was recorded in its "
    "own block.",
    "Level is fully confounded with recording day: all novices were recorded on 2019-09-25, "
    "all pros on other days. Novice-vs-pro accuracy (D5) may partly reflect session "
    "differences such as sensor mounting.",
    "Coach scores are averages of 3 coaches per criterion, so coach agreement (a ceiling for "
    "quality prediction) cannot be computed. Scores depend strongly on level.",
    "Temperature scaling fitted on 2 inner validation players did not transfer to unseen players; "
    "confidences are uncalibrated.",
    "Choosing the 'best' model by its LOPO score (for D4-D6) is a small optimistic bias.",
]


def build_summary(results: Path) -> str:
    prefixes = {
        "D3 stroke type": "d3",
        "D6 ablation": "d6",
        "D4 quality": "d4",
        "D5 skill / cross-group": "d5",
        "D8 latency": "d8",
    }
    runs = {name: latest(results, p) for name, p in prefixes.items()}
    lines = [
        "# Results summary",
        "",
        "> Generated by `make report` (`scripts/report.py`) from the saved result files in "
        "`results/`. Do not edit by hand.",
        "",
        *runs_section(runs),
    ]
    d3 = runs["D3 stroke type"]
    if d3:
        lines += dataset_section(d3) + d3_section(d3)
    sections = [
        ("D6 ablation", d6_section),
        ("D4 quality", d4_section),
        ("D5 skill / cross-group", d5_section),
    ]
    for name, fn in sections:
        lines += fn(runs[name]) if runs[name] else [f"## {name}", "", "Not run yet.", ""]
    if d3:
        lines += d7_section(d3)
    lines += (
        d8_section(runs["D8 latency"])
        if runs["D8 latency"]
        else ["## D8 latency", "", "Not run yet.", ""]
    )
    lines += [
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
        "## Future work",
        "",
        "- Collect synchronized racket-IMU and video or motion-capture recordings, at a higher "
        "logging rate and with stroke types interleaved within recordings, to extend this from "
        "stroke recognition to body-motion inference and to rule out recording-block artifacts.",
    ]
    return "\n".join(lines) + "\n"
