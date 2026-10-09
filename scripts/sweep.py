#!/usr/bin/env python
"""Quét tracker / conf / iou để chọn cấu hình nộp.

Hai chế độ:
    video1: chạy đủ frame, chấm HOTA / MOTA / IDF1 bằng TrackEval (video_1 có nhãn).
            Ba giai đoạn, mỗi lần chỉ đổi một tham số: tracker x conf -> conf mịn hơn
            (0.15, 0.5) cho tracker tốt nhất -> iou (0.4, 0.7) cho cấu hình tốt nhất.
    others: video_2..video_5 không có nhãn nên không chấm HOTA được. Dùng điểm thay thế
            ``proxy`` tính từ file kết quả (độ phủ hộp/frame và số chỗ track bị đứt quãng),
            với hệ số đã khớp trên 17 cấu hình của video_1 (xem ``PROXY_*``). Cũng ba giai đoạn
            như trên, rồi ghi cấu hình có ``proxy`` cao nhất.

``best_config.json`` được ghi/gộp theo từng video để ``run_all.py --config-json`` đọc.
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
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
TRACKERS = ["bytetrack", "ocsort", "botsort", "strongsort", "deepocsort"]
CONFS_VIDEO1 = [0.1, 0.2, 0.3]
CONFS_OTHERS = [0.15, 0.3]
CONFS_REFINE_VIDEO1 = [0.15, 0.5]
CONFS_REFINE_OTHERS = [0.1, 0.2, 0.5]
IOUS_REFINE = [0.4, 0.7]
OTHER_VIDEOS = ["video_2", "video_3", "video_4", "video_5"]
SCORE_KEYS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
PROXY_FRAMES = 150

# Điểm thay thế cho video không nhãn: HOTA ước lượng = a + b * phủ_chuẩn_hoá + c * đứt_chuẩn_hoá.
# Hệ số khớp bằng bình phương tối thiểu trên 17 cấu hình của video_1 (150 frame đầu), trong đó
# phủ_chuẩn_hoá = hộp/frame chia cho giá trị lớn nhất cùng video, đứt_chuẩn_hoá = số chỗ track
# đứt quãng chia cho giá trị lớn nhất cùng video. Kiểm tra leave-one-out: tương quan 0,82 với HOTA
# thật, cấu hình được chọn đứng hạng 5/17 (HOTA kém cấu hình tốt nhất 0,8). Chỉ kiểm chứng trên một
# video có nhãn, nên chỉ dùng để xếp hạng, không xem là số HOTA.
PROXY_INTERCEPT = 24.53
PROXY_W_COVER = 13.35
PROXY_W_GAPS = -14.39


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

    Chỉ tính các dòng có ``frame <= n_frames``.

    Args:
        lines: Các dòng ``frame,id,x,y,w,h,conf,...``.
        n_frames: Số frame cần tính (và dùng làm mẫu số của hộp/frame).

    Returns:
        Dict gồm ``rows``, ``ids``, ``boxes_per_frame``, ``mean_track_len`` (số hộp trung bình
        trên mỗi ID) và ``gaps`` (số lần một ID bị đứt quãng: có frame trống ở giữa hai lần
        xuất hiện; đây là dấu hiệu mất rồi bắt lại hoặc đổi danh tính).
    """
    frames_by_id: Dict[int, List[int]] = {}
    rows = 0
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(",")
        frame = int(float(parts[0]))
        if n_frames and frame > n_frames:
            continue
        frames_by_id.setdefault(int(float(parts[1])), []).append(frame)
        rows += 1
    gaps = 0
    for frames in frames_by_id.values():
        frames.sort()
        gaps += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
    ids = len(frames_by_id)
    return {
        "rows": rows,
        "ids": ids,
        "boxes_per_frame": rows / n_frames if n_frames else 0.0,
        "mean_track_len": rows / ids if ids else 0.0,
        "gaps": gaps,
    }


def proxy_scores(rows: Sequence[Dict[str, object]]) -> List[float]:
    """Tính điểm thay thế cho từng cấu hình của MỘT video (chuẩn hoá theo cấu hình tối đa).

    Args:
        rows: Mỗi hàng có ``boxes_per_frame`` và ``gaps``.

    Returns:
        Điểm HOTA ước lượng, cùng thứ tự với ``rows``. Giá trị chỉ dùng để xếp hạng.
    """
    if not rows:
        return []
    max_cover = max(float(r["boxes_per_frame"]) for r in rows) or 1.0
    max_gaps = max(float(r["gaps"]) for r in rows) or 1.0
    return [
        PROXY_INTERCEPT
        + PROXY_W_COVER * float(r["boxes_per_frame"]) / max_cover
        + PROXY_W_GAPS * float(r["gaps"]) / max_gaps
        for r in rows
    ]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Hệ số tương quan Pearson.

    Args:
        xs: Dãy thứ nhất.
        ys: Dãy thứ hai, cùng độ dài.

    Returns:
        Hệ số trong [-1, 1]; ``0.0`` khi một dãy không đổi hoặc ít hơn hai phần tử.

    Raises:
        ValueError: Khi hai dãy khác độ dài.
    """
    if len(xs) != len(ys):
        raise ValueError("Hai dãy phải cùng độ dài.")
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


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


def merge_best_config(existing: Dict[str, list], video: str, row: Dict[str, object]) -> Dict[str, list]:
    """Gộp cấu hình tốt nhất của một video vào dict ``best_config``.

    Args:
        existing: Nội dung ``best_config.json`` hiện có (có thể rỗng).
        video: Tên video.
        row: Hàng kết quả có ``tracker``, ``conf``, ``iou``.

    Returns:
        Dict mới; các video khác giữ nguyên.
    """
    merged = dict(existing)
    merged[video] = [row["tracker"], row["conf"], row["iou"]]
    return merged


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


def update_best_config(ketqua: Path, video: str, row: Dict[str, object]) -> None:
    """Ghi cấu hình tốt nhất của ``video`` vào ``best_config.json`` (gộp với nội dung cũ).

    Args:
        ketqua: Thư mục kết quả.
        video: Tên video.
        row: Hàng kết quả tốt nhất.
    """
    path = ketqua / "best_config.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps(merge_best_config(existing, video, row)), encoding="utf-8")


def run_stage(configs: Sequence[Tuple[str, float, float]], seen: set, rows: List[Dict[str, object]],
              evaluate) -> None:
    """Chạy các cấu hình chưa thử trong ``configs`` bằng hàm ``evaluate``.

    Args:
        configs: Danh sách ``(tracker, conf, iou)``.
        seen: Tập cấu hình đã chạy; được cập nhật tại chỗ.
        rows: Danh sách kết quả; ``evaluate`` thêm hàng mới vào đây.
        evaluate: Hàm nhận ``(tracker, conf, iou)`` và thêm một hàng vào ``rows`` khi thành công.
    """
    for config in configs:
        key = (config[0], round(config[1], 3), round(config[2], 3))
        if key in seen:
            continue
        seen.add(key)
        evaluate(*config)


def sweep_video1(args: argparse.Namespace) -> None:
    """Quét ba giai đoạn cho video_1 và chấm HOTA bằng TrackEval.

    Args:
        args: Tham số dòng lệnh đã parse.
    """
    rows: List[Dict[str, object]] = []
    seen: set = set()

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
        stats = track_stats(txt.read_text().splitlines(), PROXY_FRAMES)
        row = {"tracker": tracker, "conf": conf, "iou": iou, **{k: score.get(k, 0.0) for k in SCORE_KEYS},
               "bpf": stats["boxes_per_frame"], "gaps": stats["gaps"]}
        rows.append(row)
        print("  " + ", ".join(f"{k}={row[k]:.2f}" for k in SCORE_KEYS), flush=True)

    run_stage(grid(args.trackers, CONFS_VIDEO1, 0.5), seen, rows, evaluate)
    if not rows:
        raise SystemExit("Không cấu hình nào chấm được.")
    best = pick_best(rows)
    run_stage(grid([str(best["tracker"])], CONFS_REFINE_VIDEO1, 0.5), seen, rows, evaluate)
    best = pick_best(rows)
    run_stage([(str(best["tracker"]), float(best["conf"]), iou) for iou in IOUS_REFINE], seen, rows, evaluate)

    for row, score in zip(rows, proxy_scores([{"boxes_per_frame": r["bpf"], "gaps": r["gaps"]} for r in rows])):
        row["proxy"] = score
    corr = pearson([float(r["proxy"]) for r in rows], [float(r["HOTA"]) for r in rows])
    rows.sort(key=lambda r: -float(r["HOTA"]))
    columns = ["tracker", "conf", "iou"] + SCORE_KEYS + ["bpf", "gaps", "proxy"]
    write_outputs(rows, columns, args.ketqua, "sweep_video_1")
    best = pick_best(rows)
    update_best_config(args.ketqua, "video_1", best)
    print("\n" + format_table(rows, columns))
    print(f"\nTương quan điểm thay thế (proxy) với HOTA thật trên {len(rows)} cấu hình: {corr:+.2f}")
    print(f"Tốt nhất theo HOTA: {best['tracker']} conf={best['conf']} iou={best['iou']}")


def sweep_others(args: argparse.Namespace) -> None:
    """Quét ba giai đoạn cho video_2..5 và xếp hạng bằng điểm thay thế ``proxy``.

    Args:
        args: Tham số dòng lệnh đã parse.
    """
    for video in args.videos:
        rows: List[Dict[str, object]] = []
        seen: set = set()

        def evaluate(tracker: str, conf: float, iou: float, video: str = video) -> None:
            name = f"{tracker}_c{conf:.2f}_i{iou:g}"
            print(f"[{video}] {name} ...", flush=True)
            try:
                txt = run_tracker(args.lab_data_root, video, tracker, conf, iou,
                                  args.work / "thu_nghiem" / video / name, args.device, args.max_frames)
            except RuntimeError as err:
                print(f"  BỎ QUA: {err}", flush=True)
                return
            stats = track_stats(txt.read_text().splitlines(), args.max_frames)
            rows.append({"tracker": tracker, "conf": conf, "iou": iou, **stats})

        def rank() -> Dict[str, object]:
            for row, score in zip(rows, proxy_scores(rows)):
                row["proxy"] = score
            return pick_best(rows, "proxy")

        run_stage(grid(args.trackers, CONFS_OTHERS, 0.5), seen, rows, evaluate)
        if not rows:
            print(f"[{video}] không cấu hình nào chạy được, bỏ qua.")
            continue
        best = rank()
        run_stage(grid([str(best["tracker"])], CONFS_REFINE_OTHERS, 0.5), seen, rows, evaluate)
        best = rank()
        run_stage([(str(best["tracker"]), float(best["conf"]), iou) for iou in IOUS_REFINE], seen, rows, evaluate)
        best = rank()

        rows.sort(key=lambda r: -float(r["proxy"]))
        columns = ["tracker", "conf", "iou", "rows", "ids", "boxes_per_frame", "mean_track_len", "gaps", "proxy"]
        write_outputs(rows, columns, args.ketqua, f"sweep_{video}")
        update_best_config(args.ketqua, video, best)
        print(f"\n### {video} ({args.max_frames} frame)\n" + format_table(rows, columns) + "\n")
        print(f"Tốt nhất theo proxy: {best['tracker']} conf={best['conf']} iou={best['iou']}\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Khai báo tham số dòng lệnh.

    Args:
        argv: Danh sách tham số; ``None`` nghĩa là đọc từ ``sys.argv``.

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
    parser.add_argument("--trackers", nargs="+", default=TRACKERS, choices=TRACKERS)
    parser.add_argument("--videos", nargs="+", default=OTHER_VIDEOS, choices=OTHER_VIDEOS)
    parser.add_argument("--max-frames", type=int, default=PROXY_FRAMES, help="Chỉ dùng cho chế độ others")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.ketqua.mkdir(parents=True, exist_ok=True)
    if arguments.mode == "video1":
        sweep_video1(arguments)
    else:
        sweep_others(arguments)
