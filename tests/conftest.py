"""Shared fixtures. Everything here is synthetic: it mimics the dataset's file layout only."""

from pathlib import Path

import pytest

HEADER = (
    "#,Time/Date,Roll,Pitch,Yaw,acc(x),acc(y),acc(z),gyro(x),gyro(y),gyro(z),mag(x),mag(y),mag(z)"
)
SCORES_HEADER = "C5,C4,C3,C2,C1,Stroke,Player,# "
UNITS = {"acc_lsb_per_m_s2": 100, "gyro_lsb_per_dps": 16, "mag_lsb_per_ut": 16}


def encode_timestamp(text: str) -> str:
    """Produce a cell the way the dataset stores it: repr of GBK bytes mis-read as Latin-1."""
    return str(text.encode("gbk").decode("latin-1").encode("utf-8"))


def _rows(n: int, *, zero_imu: bool = False, second: int = 20) -> list[str]:
    rows = []
    for k in range(n):
        ts = encode_timestamp(f"2019年9月17日10时43分{second + k // 3}秒")
        acc = "0,0,0" if zero_imu else f"981,{k},0"
        gyro = "0,0,0" if zero_imu else f"{16 * (k + 1)},0,0"
        rows.append(f"{k + 2},{ts},-10.0,20.0,{30 + k}.0,{acc},{gyro},640,0,0")
    return rows


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Six samples covering every inclusion rule.

    1: BASIC, normal (keeps a duplicated row)    4: BASIC, removed by the authors
    2: TOPSPIN, acc+gyro zero in every row       5: PUSH, player overridden P06 -> P05
    3: PUSH, empty raw file                      6: BASIC, in clean CSV but no processed file
    """
    root = tmp_path / "tabrizi2020"
    sensor = root / "Raw" / "Sensory-Data"
    sensor.mkdir(parents=True)
    r1 = _rows(6)
    r1.insert(3, r1[2])  # an exact repeated row
    files = {1: r1, 2: _rows(5, zero_imu=True), 3: [], 4: _rows(4), 5: _rows(7), 6: _rows(4)}
    for sid, rows in files.items():
        (sensor / f"{sid}.csv").write_text("\n".join([HEADER, *rows]) + "\n")

    scores = [
        "80,90,90,90,90,BASIC ,PO1,1",
        "70,80,60,70,70,Top,PO2,2",
        "60,80,50,60,60,push,NO1,3",
        "50,80,40,50,50,Basic ,NO2,4",
        "90,95,90,95,95,PUSH,PO6,5",
        "40,80,30,40,40,BASIC,NO3,6",
    ]
    (root / "Raw" / "Samples-Scores").mkdir(parents=True)
    (root / "Raw" / "Samples-Scores" / "Samples-Scores.csv").write_text(
        "\n".join([SCORES_HEADER, *scores]) + "\n"
    )
    clean_dir = root / "Processed" / "Clean-Samples-Scores"
    clean_dir.mkdir(parents=True)
    clean = [s for s in scores if not s.endswith(",4")]
    clean_csv = clean_dir / "Clean-Samples-Scores.csv"
    clean_csv.write_text("\n".join([SCORES_HEADER, *clean]) + "\n")
    for folder, ids in {"Basic": [1], "TOPSPIN": [2], "PUSH": [3, 5]}.items():
        (root / "Processed" / folder).mkdir(parents=True)
        for sid in ids:
            (root / "Processed" / folder / f"{sid}.csv").write_text(HEADER + "\n")
    return root


@pytest.fixture
def dataset_cfg() -> dict:
    return {"units": UNITS, "player_overrides": {5: "P05"}, "stated_rates_hz": {"paper": 100}}
