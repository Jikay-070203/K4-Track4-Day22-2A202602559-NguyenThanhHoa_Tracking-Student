"""Tests for sweep helpers without GPU, lab images, or a network."""

import pytest

from run_all import SUBMISSION_CONFIG, merge_config
from sweep import format_table, grid, parse_summary, pick_best, track_stats

SUMMARY = "HOTA DetA MOTA IDF1 IDSW\n29.46 18.1 19.81 29.35 25\n"


def test_parse_summary_reads_columns() -> None:
    parsed = parse_summary(SUMMARY)
    assert parsed["HOTA"] == 29.46
    assert parsed["IDSW"] == 25.0


def test_parse_summary_rejects_mismatch() -> None:
    with pytest.raises(ValueError):
        parse_summary("A B C\n1 2\n")
    with pytest.raises(ValueError):
        parse_summary("A B C\n")


def test_track_stats_counts_rows_ids_and_length() -> None:
    lines = ["1,1,0,0,10,10,0.9,-1,-1,-1", "2,1,0,0,10,10,0.9,-1,-1,-1", "2,2,5,5,10,10,0.8,-1,-1,-1", ""]
    stats = track_stats(lines, n_frames=2)
    assert stats["rows"] == 3 and stats["ids"] == 2
    assert stats["boxes_per_frame"] == 1.5
    assert stats["mean_track_len"] == 1.5


def test_track_stats_empty() -> None:
    assert track_stats([], 10) == {"rows": 0, "ids": 0, "boxes_per_frame": 0.0, "mean_track_len": 0.0}


def test_grid_size() -> None:
    assert len(grid(["a", "b"], [0.1, 0.2, 0.3], 0.5)) == 6


def test_pick_best_by_key() -> None:
    rows = [{"HOTA": 10.0, "t": "a"}, {"HOTA": 30.0, "t": "b"}]
    assert pick_best(rows)["t"] == "b"
    with pytest.raises(ValueError):
        pick_best([])


def test_format_table_has_header_and_rows() -> None:
    table = format_table([{"tracker": "botsort", "HOTA": 29.456}], ["tracker", "HOTA"])
    lines = table.splitlines()
    assert lines[0] == "| tracker | HOTA |"
    assert lines[2] == "| botsort | 29.46 |"


def test_merge_config_overrides_without_mutating_base() -> None:
    merged = merge_config(SUBMISSION_CONFIG, {"video_1": ["ocsort", 0.15, 0.6]})
    assert merged["video_1"] == ("ocsort", 0.15, 0.6)
    assert SUBMISSION_CONFIG["video_1"] != merged["video_1"]
    assert merged["video_2"] == SUBMISSION_CONFIG["video_2"]


def test_merge_config_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        merge_config(SUBMISSION_CONFIG, {"video_9": ["a", 0.1, 0.5]})
    with pytest.raises(ValueError):
        merge_config(SUBMISSION_CONFIG, {"video_1": ["a", 0.1]})
