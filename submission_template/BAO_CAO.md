# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Cá nhân **Thành viên:** Nguyễn Thanh Hòa (2A202602559)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Đây là **bản nộp lần 1**. Quy trình: chạy 5 video đủ frame với cấu hình chọn theo đặc điểm cảnh, sau đó quét 5 tracker (`bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`) bằng `scripts/sweep.py` để kiểm tra lại.
- `video_1` (có nhãn): 5 tracker × `conf` {0.1, 0.2, 0.3} ở `iou` 0.5, đủ 600 frame, rồi quét `iou` {0.4, 0.7} quanh tracker tốt nhất; chấm HOTA / MOTA / IDF1 bằng TrackEval. Bảng 17 cấu hình: `bang_quet/sweep_video_1.md`. Cấu hình nộp của `video_1` lấy từ bảng này.
- `video_2`–`video_5` (không nhãn): 5 tracker × `conf` {0.15, 0.3}, `iou` 0.5, 150 frame đầu. Không có số HOTA, nên so sánh bằng ba số đếm từ file kết quả: **hộp/frame** (độ phủ), **số ID**, **độ dài track trung bình** (ít ID, track dài là ít đổi danh tính). Bảng: `bang_quet/sweep_video_N.md`. Ba số này chỉ là chỉ báo gián tiếp, không phân biệt hộp đúng với hộp giả.
- Cấu hình nộp của `video_2`–`video_5` được chọn **trước** khi quét. Kết quả quét cho thấy một số video còn cấu hình tốt hơn; điều đó được ghi trung thực ở cột cuối và ở mục 4, chưa thay vào bản này.

## 1. Cấu hình đã chọn

Trong cột quan sát: "toàn video" là thống kê trên file nộp đủ frame; "150 frame đầu" là cùng file nhưng chỉ lấy 150 frame đầu, để so sánh được với bảng quét.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại / đã quét |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.10 | 0.5 | Người ở gần được bao hộp ổn định; nhóm người đi sát nhau ở giữa ảnh vẫn giữ riêng từng hộp. Người ở xa và nhỏ (nhiều người ở nền phố bên trái) phần lớn không có hộp: tracker chỉ bắt được 24% số người trong nhãn, còn ID thì ít đổi (23 lần đổi ID trong 600 frame). | `bytetrack` 0.2: ít đổi ID nhất (12) nhưng DetA thấp nhất (15,4), HOTA 27,50. `strongsort` 0.1: đổi ID 197 lần, MOTA 15,6. `ocsort`/`deepocsort` 0.1: đổi ID ~300 lần, HOTA ~23. `botsort` 0.3 (HOTA 29,46) và `iou` 0.4 / 0.7 (HOTA 29,33 / 28,73): kém hơn một chút. |
| video_2 (phố đêm, tĩnh, rất đông) | deepocsort | 0.20 | 0.6 | Cảnh có đèn đường sáng chói nhưng người vẫn rõ; hộp phủ nhiều người đi bộ ở hai bên vỉa hè. Người ở rất xa (phía trên ảnh) và người bị cột đèn che nửa thân dễ không có hộp. Toàn video: 14,3 hộp/frame, 107 ID, 31% track ngắn hơn 15 frame. 150 frame đầu: 12,95 hộp/frame, 31 ID, track 62,7 frame. | Quét 150 frame, chưa thay: `strongsort` 0.15 tốt hơn trên cả ba số (14,25 hộp/frame, 30 ID, track 71,2). `bytetrack` 0.15: ít ID (15) nhưng chỉ 9,18 hộp/frame, bỏ sót khoảng một phần ba số người. `deepocsort` 0.15: 38 ID, track 55,9. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.25 | 0.5 | Ảnh 640×480, người đi rất gần camera nên hộp rất lớn, bị cắt ở mép khung và chồng lên nhau; camera di chuyển làm vị trí hộp thay đổi cả khi người đứng yên. Toàn video: 5,8 hộp/frame, **168 ID cho 837 frame, 54% track ngắn hơn 15 frame** (đứt ID nhiều). 150 frame đầu: 5,23 hộp/frame, 22 ID, track 35,7 frame. | Quét 150 frame, chưa thay: `strongsort` 0.15 tốt hơn (7,01 hộp/frame, 24 ID, track 43,8). `botsort` 0.15: 5,65 hộp/frame, 21 ID, track 40,4. `ocsort`/`deepocsort` 0.15: cùng độ phủ ~7,0 nhưng 36–37 ID, track ~28–29. `bytetrack` 0.15: chỉ 3,98 hộp/frame. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.40 | 0.5 | Trung tâm thương mại, sàn bóng và lan can kính phản chiếu ánh đèn. Người đi gần camera được bao hộp lớn, rõ; người ở nền bị người phía trước che. Toàn video: 6,7 hộp/frame, 66 ID, 26% track ngắn hơn 15 frame. 150 frame đầu: 5,85 hộp/frame, 16 ID, track 54,9 frame. Phản chiếu người trên kính cần kiểm tra thêm trên video xem thử (khung mẫu không thấy hộp trên kính). | Quét 150 frame, chưa thay: `botsort` 0.15 cùng 16 ID nhưng nhiều hộp hơn (6,91 hộp/frame) và track dài hơn (64,8); `botsort` 0.30: 6,39 hộp/frame, 18 ID, track 53,2. `bytetrack` 0.15: 13 ID, track 65 nhưng chỉ 5,63 hộp/frame. `strongsort`/`deepocsort`/`ocsort` 0.15: 31–35 ID, track 33–37. |
| video_5 (trên xe bus, giao lộ đông) | strongsort | 0.30 | 0.5 | Phần lớn khung hình là mặt đường và xe; người đi bộ rất nhỏ (khung mẫu chỉ có hai người được bao hộp) và camera rung. Toàn video: 4,2 hộp/frame, **104 ID cho 750 frame, 49% track ngắn hơn 15 frame**. 150 frame đầu: 6,33 hộp/frame, 30 ID, track 31,7 frame. | Quét 150 frame, chưa thay: `botsort` 0.15 tốt hơn (6,63 hộp/frame, 27 ID, track 36,8); `botsort` 0.30: 6,05 hộp/frame, 27 ID, track 33,6. `strongsort` 0.15: độ phủ ~10 hộp/frame nhưng 60 ID, track 24,9. `ocsort`/`deepocsort` 0.15: 51–57 ID. |

## 2. Số liệu video_1

Cấu hình nộp: `botsort`, `conf` 0.10, `iou` 0.5. Bảng do `scripts/evaluate_practice.py` in ra (toàn văn: `bang_quet/danh_gia_video_1.txt`).

```
HOTA: final_video1-pedestrian      HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            29.523    19.561    44.952    20.335    75.327    47.931    82.084    82.598    30.152    36.707    76.442    28.059

CLEAR: final_video1-pedestrian     MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
video_1                            21.005    80.163    21.129    24.062    89.135    12.903    20.968    66.129    16.232    4471      14110     545       23        8         13        41        64

Identity: final_video1-pedestrian  IDF1      IDR       IDP       IDTP      IDFN      IDFP
video_1                            30.3      19.24     71.272    3575      15006     1441

Count: final_video1-pedestrian     Dets      GT_Dets   IDs       GT_IDs
video_1                            5016      18581     56        62
```

Tóm tắt: **HOTA 29,52 · MOTA 21,01 · IDF1 30,30**. `video_2` đến `video_5` không có nhãn trong gói lab, nên không điền số cho các video đó.

## 3. Phân tích

**video_1: giới hạn nằm ở detector, không phải tracker.** Trong 17 cấu hình, DetA chỉ nằm trong khoảng 15–23 và recall tối đa ~24%, trong khi AssA dao động 24–49 và số lần đổi ID từ 12 đến ~300. Nghĩa là việc chọn tracker làm thay đổi mạnh phần giữ danh tính (AssA, IDSW) nhưng gần như không đẩy được phần phát hiện (DetA), vì detector `yolo26n` ở 640 px không bắt được nhiều người nhỏ/xa (nhãn có 18 581 hộp, tracker chỉ xuất 5 016). Hạ `conf` từ 0.3 xuống 0.1 làm DetA tăng 18,1 → 19,6 nhưng độ chính xác giảm (DetPr 78,9 → 75,3), nên HOTA gần như không đổi (29,46 → 29,52). Vì vậy chênh lệch giữa các cấu hình tốt nhất là nhỏ.

**video_2 (đêm, tĩnh, rất đông).** Camera đứng yên nên ít bị lệch vị trí, nhưng cảnh đông và người đi sát nhau khiến hộp hay chồng nhau. Chọn `deepocsort` (có Re-ID) vì khi hai người cắt nhau, chỉ dựa vào chuyển động rất dễ gán nhầm. Số liệu quét ở `conf` 0.15 cho thấy `bytetrack` (chủ yếu theo chuyển động) chỉ phủ 9 hộp/frame nên bỏ sót nhiều; trong nhóm độ phủ cao (~14 hộp/frame), `strongsort` có 30 ID và track dài 71 frame, còn `deepocsort` 38 ID, track 56 frame và `ocsort` 37 ID, track 57 frame. Tức là việc dùng Re-ID là đúng hướng, nhưng `strongsort` 0.15 có vẻ hợp hơn cấu hình đã nộp (`deepocsort` 0.2, `iou` 0.6). Toàn video có 107 ID với 31% track ngắn hơn 15 frame, phù hợp với cảnh rất đông.

**video_4 (trong nhà, camera tiến tới).** Camera di chuyển làm chuyển động của từng người không còn tuyến tính, nên tracker chỉ dựa vào chuyển động dễ mất người; `botsort` có bù chuyển động camera và Re-ID. Trong bảng quét, `botsort` cho 16–18 ID và track 53–65 frame, trong khi `strongsort`/`deepocsort`/`ocsort` (độ phủ nhỉnh hơn khoảng 10%) lại cho 31–35 ID và track chỉ 33–37 frame; đây là lý do giữ `botsort`. Cấu hình nộp dùng `conf` 0.4 để hạn chế hộp giả từ sàn và kính phản chiếu; bảng quét lại cho thấy `conf` 0.15 cùng số ID nhưng nhiều hộp hơn và track dài hơn. Số đếm không phân biệt được hộp giả với hộp thật, nên cần xem video xem thử để biết hạ `conf` có thực sự sinh hộp giả từ phản chiếu hay không.

**Giới hạn của kết luận.** Với `video_2`–`video_5` không có nhãn, so sánh dựa trên ba số đếm trong 150 frame đầu cộng một khung hình mẫu mỗi video, không phải HOTA. Số đếm có thể thưởng cho cấu hình sinh nhiều hộp giả (độ phủ cao) hoặc cho cấu hình bỏ sót nhưng giữ ID tốt (ít ID), nên phần so sánh trên là bằng chứng gián tiếp. Ngoài ra `video_3` và `video_5` có tỉ lệ track ngắn rất cao ở bản nộp (54% và 49% ngắn hơn 15 frame), cho thấy hai cảnh camera di chuyển/rung này là chỗ cấu hình hiện tại còn yếu nhất.

## 4. Cải tiến

Cập nhật dự kiến cho bản sau, rút ra từ bảng quét 150 frame (chưa chạy đủ frame, chưa xem video): `video_2` → `strongsort` 0.15; `video_3` → `strongsort` 0.15; `video_4` → `botsort` 0.15; `video_5` → `botsort` 0.15 (`iou` giữ 0.5). Cả bốn đều có độ phủ bằng hoặc cao hơn, và số ID bằng hoặc thấp hơn cấu hình đã nộp. Sau đó cần chạy đủ frame, xem video xem thử xem hạ `conf` có sinh hộp giả không, quét `iou` cho từng video (hiện chỉ quét ở `video_1`), và xem lại các frame mà ID đổi để biết lỗi do detector bỏ sót hay do tracker gán nhầm. Thử Re-ID khác chỉ nên làm như phần mở rộng (ghi rõ ngoài bài nộp chính). Việc cải thiện DetA cần detector hoặc kích thước ảnh tốt hơn, nằm ngoài luật chơi của lab này.
