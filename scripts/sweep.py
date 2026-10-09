#!/usr/bin/env python
"""Quét tracker / conf / iou để chọn cấu hình nộp.

Hai chế độ:
    video1: chạy đủ frame, chấm HOTA / MOTA / IDF1 bằng TrackEval (video_1 có nhãn),
            rồi quét thêm ``iou`` quanh tracker tốt nhất. Ghi ``best_config.json``.
    others: video_2..video_5 không có nhãn; chạy giới hạn frame và in số liệu thống kê
            (hộp/frame, số ID, độ dài track trung bình) để đối chiếu với video xem thử.

Detector, kích thước ảnh và Re-ID vẫn cố định; chỉ đổi ``--tracker``, ``--conf``, ``--iou``.

Ví dụ:
    python scripts/sweep.py --mode video1 --lab-data-root "$LAB_DATA" \\
        --trackeval-root TrackEval --lab-eval-root lab_eval --device cuda:0
    python scripts/sweep.py --mode others --lab-data-root "$LAB_DATA" --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TRACKERS = ["bytetrack", "ocsort", "botsort", "strongsort", "deepocsort"]
CONFS_VIDEO1 = [0.1, 0.2, 0.3]
CONFS_OTHERS = [0.15, 0.3]
IOUS_REFINE = [0.4, 0.7]
OTHER_VIDEOS = ["video_2", "video_3", "video_4", "video_5"]
SCORE_KEYS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def parse_summary(text: str) -> Dict[str, float]:
    """Đọc file ``pedestrian_summary.txt`` của TrackEval (dòng tên cột, dòng giá trị).

    Args:
        text: Nội dung file.

    Returns:
        Dict tên cột -> giá trị số.

    Raises:
        ValueError: Khi file có ít hơn hai dòng hoặc số cột không khớp số giá trị.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Summary cần ít nhất hai dòng (tên cột, giá trị).")
    keys, values = lines[0].split(), lines[1].split()
    if len(keys) != len(values):
        raise ValueError(f"Số cột {len(keys)} khác số giá trị {len(values)}.")
    return {k: float(v) for k, v in zip(keys, values)}


def track_stats(lines: Sequence[str], n_frames: int) -> Dict[str, float]:
    """Thống kê nhanh một file kết quả MOT, không cần nhãn.

    Args:
        lines: Các dòng ``frame,id,x,y,w,h,conf,...``.
        n_frames: Số frame đã chạy.

    Returns:
        Dict gồm ``rows``, ``ids``, ``boxes_per_frame`` và ``mean_track_len``
        (số hộp trung bình trên mỗi ID; càng ngắn càng dễ bị đứt ID).
    """
    ids = set()
    rows = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(",")
        ids.add(int(float(parts[1])))
        rows += 1
    return {
        "rows": rows,
        "ids": len(ids),
        "boxes_per_frame": rows / n_frames if n_frames else 0.0,
        "mean_track_len": rows / len(ids) if ids else 0.0,
    }


def grid(trackers: Sequence[str], confs: Sequence[float], iou: float) -> List[Tuple[str, float, float]]:
    """Liệt kê cấu hình quét: mỗi tracker thử từng ``conf`` với một ``iou`` cố định.

    Args:
        trackers: Danh sách tracker.
        confs: Danh sách ``conf``.
        iou: Giá trị ``iou`` dùng chung.

    Returns:
        Danh sách ``(tracker, conf, iou)``.
    """
    return [(t, c, iou) for t in trackers for c in confs]


def format_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    """Dựng bảng Markdown từ danh sách dict.

    Args:
        rows: Mỗi phần tử là một hàng.
        columns: Thứ tự cột.

    Returns:
        Chuỗi Markdown có dòng tiêu đề và dòng phân cách.
    """
    def cell(value: object) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)

    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        out.append("| " + " | ".join(cell(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def pick_best(rows: Sequence[Dict[str, object]], key: str = "HOTA") -> Dict[str, object]:
    """Chọn hàng có ``key`` lớn nhất.

    Args:
        rows: Các hàng kết quả đã chấm.
        key: Tên cột dùng để xếp hạng.

    Returns:
        Hàng tốt nhất.

    Raises:
        ValueError: Khi ``rows`` rỗng.
    """
    if not rows:
        raise ValueError("Không có kết quả nào để chọn.")
    return max(rows, key=lambda r: float(r[key]))


def run_tracker(lab_data: Path, video: str, tracker: str, conf: float, iou: float,
                out_dir: Path, device: str, max_frames: int) -> Path:
    """Gọi ``run_tracking.py`` cho một cấu hình và trả đường dẫn file kết quả.

    Args:
        lab_data: Thư mục lab_data.
        video: Tên video.
        tracker: Tên tracker.
        conf: Ngưỡng confidence của detector.
        iou: Ngưỡng IoU NMS của detector.
        out_dir: Thư mục xuất.
        device: ``cpu`` hoặc ``cuda:0``.
        max_frames: Giới hạn frame, 0 là đủ.

    Returns:
        Đường dẫn ``out_dir/<video>.txt``.

    Raises:
        RuntimeError: Khi ``run_tracking.py`` thoát với mã khác 0.
    """
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_tracking.py"),
        "--source", str(lab_data / video / "img1"), "--seq-name", video,
        "--tracker", tracker, "--conf", str(conf), "--iou", str(iou),
        "--out", str(out_dir), "--device", device,
    ]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{tracker} conf={conf} iou={iou} lỗi:\n{proc.stderr[-1500:]}")
    return out_dir / f"{video}.txt"


def score_video1(txt: Path, run_name: str, trackeval_root: Path, lab_eval_root: Path) -> Dict[str, float]:
    """Chấm video_1 bằng ``evaluate_practice.py`` rồi đọc bảng điểm.

    Args:
        txt: File ``video_1.txt`` cần chấm.
        run_name: Tên lần chấm (duy nhất cho mỗi cấu hình).
        trackeval_root: Thư mục TrackEval.
        lab_eval_root: Thư mục chứa ``video_1/gt``, ``seqinfo.ini``, ``eval_config.json``.

    Returns:
        Dict điểm TrackEval (HOTA, MOTA, IDF1, ...).

    Raises:
        RuntimeError: Khi chấm lỗi hoặc không tìm thấy file summary.
    """
    cmd = [
        sys.executable, str(SCRIPT_DIR / "evaluate_practice.py"),
        "--trackeval-root", str(trackeval_root), "--lab-data-root", str(lab_eval_root),
        "--submission", str(txt), "--run-name", run_name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"})
    found = list((trackeval_root / "data" / "trackers").rglob(f"{run_name}/pedestrian_summary.txt"))
    if proc.returncode != 0 or not found:
        raise RuntimeError(f"Chấm {run_name} lỗi:\n{(proc.stderr or proc.stdout)[-1500:]}")
    return parse_summary(found[0].read_text())


def write_outputs(rows: Sequence[Dict[str, object]], columns: Sequence[str], out_dir: Path, stem: str) -> None:
    """Ghi bảng kết quả ra CSV và Markdown.

    Args:
        rows: Các hàng kết quả.
        columns: Thứ tự cột.
        out_dir: Thư mục xuất.
        stem: Tên file không có đuôi.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{stem}.md").write_text(format_table(rows, columns) + "\n", encoding="utf-8")


def sweep_video1(args: argparse.Namespace) -> None:
    """Quét tracker x conf, rồi quét iou quanh tracker tốt nhất, chấm bằng HOTA.

    Args:
        args: Tham số dòng lệnh đã parse.
    """
    rows: List[Dict[str, object]] = []

    def evaluate(tracker: str, conf: float, iou: float) -> None:
        name = f"{tracker}_c{conf:.2f}_i{iou:g}"
        out_dir = args.work / "thu_nghiem" / name
        print(f"[video_1] {name} ...", flush=True)
        try:
            txt = run_tracker(args.lab_data_root, "video_1", tracker, conf, iou, out_dir, args.device, 0)
            score = score_video1(txt, name, args.trackeval_root, args.lab_eval_root)
        except RuntimeError as err:
            print(f"  BỎ QUA: {err}", flush=True)
            return
        row = {"tracker": tracker, "conf": conf, "iou": iou, **{k: score.get(k, 0.0) for k in SCORE_KEYS}}
        rows.append(row)
        print("  " + ", ".join(f"{k}={row[k]:.2f}" for k in SCORE_KEYS), flush=True)

    for tracker, conf, iou in grid(TRACKERS, CONFS_VIDEO1, 0.5):
        evaluate(tracker, conf, iou)
    if not rows:
        raise SystemExit("Không cấu hình nào chấm được.")
    best = pick_best(rows)
    for iou in IOUS_REFINE:
        evaluate(str(best["tracker"]), float(best["conf"]), iou)

    rows.sort(key=lambda r: -float(r["HOTA"]))
    columns = ["tracker", "conf", "iou"] + SCORE_KEYS
    write_outputs(rows, columns, args.ketqua, "sweep_video_1")
    best = pick_best(rows)
    (args.ketqua / "best_config.json").write_text(
        json.dumps({"video_1": [best["tracker"], best["conf"], best["iou"]]}), encoding="utf-8")
    print("\n" + format_table(rows, columns))
    print(f"\nTốt nhất theo HOTA: {best['tracker']} conf={best['conf']} iou={best['iou']}")


def sweep_others(args: argparse.Namespace) -> None:
    """Chạy giới hạn frame cho video_2..5 và thống kê, không cần nhãn.

    Args:
        args: Tham số dòng lệnh đã parse.
    """
    for video in args.videos:
        rows: List[Dict[str, object]] = []
        for tracker, conf, iou in grid(TRACKERS, CONFS_OTHERS, 0.5):
            name = f"{tracker}_c{conf:.2f}_i{iou:g}"
            print(f"[{video}] {name} ...", flush=True)
            try:
                txt = run_tracker(args.lab_data_root, video, tracker, conf, iou,
                                  args.work / "thu_nghiem" / video / name, args.device, args.max_frames)
            except RuntimeError as err:
                print(f"  BỎ QUA: {err}", flush=True)
                continue
            stats = track_stats(txt.read_text().splitlines(), args.max_frames)
            rows.append({"tracker": tracker, "conf": conf, "iou": iou, **stats})
        columns = ["tracker", "conf", "iou", "rows", "ids", "boxes_per_frame", "mean_track_len"]
        write_outputs(rows, columns, args.ketqua, f"sweep_{video}")
        print(f"\n### {video} ({args.max_frames} frame)\n" + format_table(rows, columns) + "\n")


def parse_args() -> argparse.Namespace:
    """Khai báo tham số dòng lệnh.

    Returns:
        Namespace chứa các tham số đã parse.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["video1", "others"])
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--trackeval-root", type=Path, default=Path("TrackEval"))
    parser.add_argument("--lab-eval-root", type=Path, default=Path("lab_eval"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work", type=Path, default=Path("runs"), help="Thư mục chứa kết quả từng lần thử")
    parser.add_argument("--ketqua", type=Path, default=Path("ketqua"), help="Thư mục ghi bảng tổng hợp")
    parser.add_argument("--videos", nargs="+", default=OTHER_VIDEOS, choices=OTHER_VIDEOS)
    parser.add_argument("--max-frames", type=int, default=150, help="Chỉ dùng cho chế độ others")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.ketqua.mkdir(parents=True, exist_ok=True)
    if arguments.mode == "video1":
        sweep_video1(arguments)
    else:
        sweep_others(arguments)
