#!/usr/bin/env python
"""Chạy tracker cho cả năm video (bản nộp) hoặc quét conf/iou (bản thử) trong một lệnh.

Script chỉ gọi lại ``run_tracking.py``; detector, kích thước ảnh và Re-ID vẫn cố định.
Cấu hình nộp nằm trong ``SUBMISSION_CONFIG`` — sửa sau khi đã xem video thử.

Ví dụ:
    # Bản nộp: đủ frame, đủ 5 video -> runs/nop_bai/video_N.txt
    python scripts/run_all.py --lab-data-root "$LAB_DATA" --device cuda:0

    # Bản thử: quét tracker x conf x iou, giới hạn frame, một video
    python scripts/run_all.py --lab-data-root "$LAB_DATA" --sweep \\
        --videos video_1 --max-frames 150
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS = ["video_1", "video_2", "video_3", "video_4", "video_5"]

# video -> (tracker, conf, iou): cấu hình của bản nộp lần 1 (sinh ra runs/nop_bai/).
#   video_1: HOTA cao nhất trong 17 cấu hình quét bằng sweep.py (có nhãn).
#   video_2..5: chọn theo đặc điểm cảnh trước khi quét; bảng quét 150 frame ở bang_quet/ cho thấy
#   còn cấu hình tốt hơn (xem mục 4 của báo cáo), sẽ cập nhật ở bản sau.
SUBMISSION_CONFIG: Dict[str, Tuple[str, float, float]] = {
    "video_1": ("botsort", 0.10, 0.5),     # tĩnh, ban ngày: HOTA 29.52, tốt nhất khi quét
    "video_2": ("deepocsort", 0.20, 0.6),  # đêm, rất đông, người nhỏ: conf thấp, Re-ID
    "video_3": ("botsort", 0.25, 0.5),     # camera di chuyển: botsort có bù chuyển động camera
    "video_4": ("botsort", 0.40, 0.5),     # trong nhà, kính phản chiếu: conf cao bớt hộp giả
    "video_5": ("strongsort", 0.30, 0.5),  # xe bus rung lắc, đông: Re-ID + làm mượt
}

SWEEP_TRACKERS = ["bytetrack", "botsort"]
SWEEP_CONFS = [0.15, 0.3, 0.5]
SWEEP_IOUS = [0.4, 0.5, 0.7]


def build_command(
    lab_data_root: Path,
    video: str,
    tracker: str,
    conf: float,
    iou: float,
    out: Path,
    device: str,
    max_frames: int = 0,
    save_video: bool = True,
) -> List[str]:
    """Dựng lệnh gọi ``run_tracking.py`` cho một video.

    Args:
        lab_data_root: Thư mục lab_data giảng viên phát.
        video: Tên video, ví dụ ``video_1``.
        tracker: Tên tracker.
        conf: Ngưỡng confidence của detector.
        iou: Ngưỡng IoU NMS của detector.
        out: Thư mục xuất kết quả.
        device: ``cpu``, ``cuda:0``, ...
        max_frames: Giới hạn số frame; 0 nghĩa là toàn bộ.
        save_video: Có xuất video xem thử không.

    Returns:
        Danh sách tham số dòng lệnh, dùng được với ``subprocess.run``.
    """
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_tracking.py"),
        "--source", str(lab_data_root / video / "img1"),
        "--seq-name", video,
        "--tracker", tracker,
        "--conf", str(conf),
        "--iou", str(iou),
        "--out", str(out),
        "--device", device,
    ]
    if save_video:
        cmd.append("--save-video")
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    return cmd


def merge_config(
    base: Dict[str, Tuple[str, float, float]], override: Dict[str, Sequence]
) -> Dict[str, Tuple[str, float, float]]:
    """Ghi đè cấu hình nộp bằng dict ``video -> [tracker, conf, iou]``.

    Args:
        base: Cấu hình gốc.
        override: Cấu hình mới, thường đọc từ file JSON (giá trị là list).

    Returns:
        Dict mới, không sửa ``base``.

    Raises:
        ValueError: Khi tên video không hợp lệ hoặc giá trị không đủ ba phần tử.
    """
    merged = dict(base)
    for video, value in override.items():
        if video not in base:
            raise ValueError(f"Video không hợp lệ: {video}")
        if len(value) != 3:
            raise ValueError(f"{video}: cần [tracker, conf, iou], nhận {value}")
        merged[video] = (str(value[0]), float(value[1]), float(value[2]))
    return merged


def sweep_out_dir(base: Path, tracker: str, conf: float, iou: float) -> Path:
    """Đặt tên thư mục kết quả cho một cấu hình quét.

    Args:
        base: Thư mục gốc chứa các lần thử.
        tracker: Tên tracker.
        conf: Ngưỡng confidence.
        iou: Ngưỡng IoU.

    Returns:
        Đường dẫn dạng ``base/<tracker>_c0.30_i0.5``.
    """
    return base / f"{tracker}_c{conf:.2f}_i{iou:g}"


def sweep_grid(
    trackers: Sequence[str], confs: Sequence[float], ious: Sequence[float]
) -> List[Tuple[str, float, float]]:
    """Liệt kê các cấu hình quét, mỗi lần chỉ đổi MỘT tham số quanh mốc (0.3, 0.5).

    Args:
        trackers: Các tracker cần thử.
        confs: Các giá trị ``conf``.
        ious: Các giá trị ``iou``.

    Returns:
        Danh sách ``(tracker, conf, iou)`` không trùng, giữ thứ tự.
    """
    base_conf, base_iou = 0.3, 0.5
    seen = []
    for tracker in trackers:
        for conf, iou in itertools.chain(
            ((c, base_iou) for c in confs), ((base_conf, i) for i in ious)
        ):
            item = (tracker, conf, iou)
            if item not in seen:
                seen.append(item)
    return seen


def main() -> None:
    """Chạy bản nộp cho các video, hoặc quét cấu hình khi có ``--sweep``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--videos", nargs="+", default=VIDEOS, choices=VIDEOS)
    parser.add_argument("--out", type=Path, default=None, help="Mặc định runs/nop_bai (nộp) hoặc runs/thu_nghiem (quét)")
    parser.add_argument("--config-json", type=Path, default=None, help="File JSON ghi đè SUBMISSION_CONFIG")
    parser.add_argument("--sweep", action="store_true", help="Quét tracker/conf/iou thay vì chạy bản nộp")
    parser.add_argument("--max-frames", type=int, default=0, help="Chỉ dùng khi quét; bản nộp luôn đủ frame")
    args = parser.parse_args()

    jobs: List[List[str]] = []
    if args.sweep:
        base = args.out or Path("runs/thu_nghiem")
        for video in args.videos:
            for tracker, conf, iou in sweep_grid(SWEEP_TRACKERS, SWEEP_CONFS, SWEEP_IOUS):
                out = sweep_out_dir(base, tracker, conf, iou)
                jobs.append(build_command(
                    args.lab_data_root, video, tracker, conf, iou, out,
                    args.device, args.max_frames or 150,
                ))
    else:
        out = args.out or Path("runs/nop_bai")
        config = SUBMISSION_CONFIG
        if args.config_json:
            config = merge_config(config, json.loads(args.config_json.read_text(encoding="utf-8")))
        for video in args.videos:
            tracker, conf, iou = config[video]
            print(f"[cấu hình] {video}: {tracker} conf={conf} iou={iou}")
            jobs.append(build_command(args.lab_data_root, video, tracker, conf, iou, out, args.device))

    for i, cmd in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
