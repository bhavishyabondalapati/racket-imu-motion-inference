"""Tests for the summary builder: it must say plainly when results are missing."""

from racketimu.report import build_summary, latest


def test_summary_marks_missing_runs(tmp_path):
    text = build_summary(tmp_path)
    assert "| D3 stroke type | not run |" in text
    assert "Not run yet." in text
    assert "## Limitations" in text


def test_latest_picks_the_newest_run(tmp_path):
    for name in ["d3_20260101T000000Z", "d3_20261231T000000Z", "d4_20270101T000000Z"]:
        (tmp_path / name).mkdir()
    assert latest(tmp_path, "d3").name == "d3_20261231T000000Z"
    assert latest(tmp_path, "d6") is None
