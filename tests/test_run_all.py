"""Tests for run_all helpers without GPU, lab images, or a network."""

from pathlib import Path

from run_all import SUBMISSION_CONFIG, VIDEOS, build_command, sweep_grid, sweep_out_dir


def test_build_command_full_run_has_no_max_frames() -> None:
    cmd = build_command(Path("lab"), "video_2", "botsort", 0.25, 0.5, Path("runs/nop_bai"), "cpu")
    assert "--max-frames" not in cmd
    assert "--save-video" in cmd
    assert cmd[cmd.index("--seq-name") + 1] == "video_2"
    assert cmd[cmd.index("--tracker") + 1] == "botsort"


def test_build_command_with_max_frames() -> None:
    cmd = build_command(Path("lab"), "video_1", "bytetrack", 0.3, 0.5, Path("o"), "cpu", max_frames=150)
    assert cmd[cmd.index("--max-frames") + 1] == "150"


def test_submission_config_covers_all_videos() -> None:
    assert set(SUBMISSION_CONFIG) == set(VIDEOS)


def test_sweep_grid_changes_one_parameter_and_has_no_duplicates() -> None:
    grid = sweep_grid(["bytetrack"], [0.15, 0.3, 0.5], [0.4, 0.5, 0.7])
    assert len(grid) == len(set(grid)) == 5
    assert all((c == 0.3 or i == 0.5) for _, c, i in grid)


def test_sweep_out_dir_name() -> None:
    assert sweep_out_dir(Path("runs"), "botsort", 0.3, 0.5) == Path("runs") / "botsort_c0.30_i0.5"
