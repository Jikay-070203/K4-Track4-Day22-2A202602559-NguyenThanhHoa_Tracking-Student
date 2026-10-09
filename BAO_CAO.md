# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** …………………………… **Thành viên:** Nguyễn Thanh Hòa (2A202602559)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

> Cách dùng file này: cấu hình bên dưới là điểm khởi đầu suy ra từ đặc điểm cảnh. Sau khi chạy
> `python scripts/run_all.py` và xem video, sửa cột "Quan sát" bằng điều bạn **thực sự thấy**,
> và nếu đổi cấu hình thì sửa cả `SUBMISSION_CONFIG` trong `scripts/run_all.py`.
> Các ô `[ĐIỀN: …]` chưa có số/quan sát thật nên cần bạn điền sau khi chạy.

## 1. Cấu hình đã chọn

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.30 | 0.5 | [ĐIỀN: ID có đổi màu khi hai người cắt nhau không? có hộp giả trên nền không?] | bytetrack 0.30/0.5 — [ĐIỀN: lý do, kèm HOTA nếu có chấm] |
| video_2 (phố đêm, tĩnh, rất đông) | deepocsort | 0.20 | 0.6 | [ĐIỀN: người nhỏ/tối có bị bỏ sót không? ID nhảy ở chỗ đông?] | bytetrack 0.30/0.5 — [ĐIỀN] |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.25 | 0.5 | [ĐIỀN: khi camera rung/xoay, ID có giữ được không?] | ocsort 0.25/0.5 — [ĐIỀN] |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.40 | 0.5 | [ĐIỀN: bóng phản chiếu trên kính có thành hộp giả không?] | bytetrack 0.25/0.5 — [ĐIỀN] |
| video_5 (trên xe bus, giao lộ đông) | strongsort | 0.30 | 0.5 | [ĐIỀN: rung lắc làm mất/đổi ID ở đâu?] | bytetrack 0.30/0.5 — [ĐIỀN] |

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

```
(dán output ở đây sau khi chạy evaluate_practice.py)
```

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

**video_2 (phố đêm, tĩnh, rất đông) — đánh giá bằng mắt.**
Camera đứng yên nên chuyển động của từng người dễ dự đoán, nhưng cảnh rất đông nên người hay che nhau
và đi sát nhau. Tracker chỉ dựa vào chuyển động (ByteTrack/OC-SORT) dễ gán nhầm khi hai người cắt nhau,
vì hai hộp chồng lên nhau có IoU gần như ngang nhau. Tracker có Re-ID (DeepOCSORT) thêm đặc trưng ngoại
hình để tách hai người, nên kỳ vọng ít đổi danh tính hơn. Ban đêm người nhỏ và tối, nên hạ `conf`
xuống 0.20 để không bỏ sót; cái giá là có thể thêm hộp giả. [ĐIỀN: xác nhận hoặc phản bác bằng điều
nhìn thấy trong `video_2_preview.mp4`, nêu ít nhất một đoạn cụ thể.]

**video_4 (trong nhà, camera tiến tới, kính phản chiếu) — đánh giá bằng mắt.**
Camera di chuyển làm vị trí hộp thay đổi cả khi người đứng yên, nên mô hình chuyển động thuần túy kém
tin cậy; BoT-SORT có bù chuyển động camera và Re-ID nên phù hợp hơn. Kính phản chiếu sinh ra "người ảo",
vì thế dùng `conf` cao hơn (0.40) để loại hộp điểm thấp. [ĐIỀN: xác nhận bằng video — hộp giả trên kính
còn không? có người thật nào bị mất vì `conf` cao?]

**video_1 (có số).** [ĐIỀN: so sánh HOTA / MOTA / IDF1 của cấu hình nộp với cấu hình đã loại; giải thích
chênh lệch thuộc về phát hiện (DetA) hay giữ danh tính (AssA).]

## 4. Nếu có thêm thời gian

Quét `conf` mịn hơn (0.2–0.4, bước 0.05) cho video_2 và video_4, xem lại các frame có ID nhảy để biết lỗi
do detector bỏ sót hay do tracker gán nhầm, và thử Re-ID khác như phần mở rộng (ghi rõ ngoài bài nộp chính).
