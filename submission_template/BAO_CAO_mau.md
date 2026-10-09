# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Cá nhân **Thành viên:** Nguyễn Thanh Hòa (2A202602559)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Đây là **bản nộp lần 2**. Cấu hình lấy từ bước quét ba giai đoạn của `scripts/sweep.py`, mỗi lần chỉ đổi một tham số: (1) 5 tracker (`bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`) × `conf`; (2) `conf` mịn hơn cho tracker tốt nhất (`conf` 0.5 và các mức thấp); (3) `iou` {0.4, 0.7} cho cấu hình tốt nhất. Toàn bộ bảng nằm trong `bang_quet/`.
- `video_1` (có nhãn): chạy đủ 600 frame, 19 cấu hình, chấm HOTA / MOTA / IDF1 bằng TrackEval. Cấu hình nộp là cấu hình có HOTA cao nhất.
- `video_2`–`video_5` (không nhãn): 150 frame đầu, 15 cấu hình mỗi video. Không có HOTA, nên xếp hạng bằng **điểm thay thế** tính từ file kết quả: độ phủ hộp/frame và **số chỗ track bị đứt quãng** (một ID có frame trống ở giữa các lần xuất hiện). Điểm này được kiểm chứng trên `video_1`: số chỗ đứt tương quan −0,76 với HOTA, −0,93 với AssA và +0,99 với số lần đổi ID, còn độ phủ hộp/frame đơn thuần lại tương quan **âm** với HOTA (−0,52) vì `conf` thấp sinh nhiều hộp và nhiều lần đổi ID. Hệ số khớp trên 17 cấu hình; trên 19 cấu hình, điểm thay thế tương quan 0,91 với HOTA thật (khớp trên chính các cấu hình này) và 0,87 khi kiểm tra bỏ-một-ra. Điểm này chỉ xếp hạng, không phải số HOTA, và mới được kiểm chứng trên một video có nhãn.
- **Tái lập:** kết quả đã chạy trọn vẹn trên Kaggle (có GPU, Python 3.11 trong venv); các lần chạy lặp lại cho file giống hệt nhau, nên kết quả là tất định. Cả 5 file `video_N.txt` có track ở mọi frame (video_1: 600, video_2: 1050, video_3: 837, video_4: 900, video_5: 750).

## 1. Cấu hình đã chọn

Cột quan sát: số liệu "toàn video" tính trên file nộp đủ frame; "hộp chồng" là tỉ lệ hộp có hộp của ID khác che lên với IoU > 0,5 (dấu hiệu hộp trùng lặp cho cùng một người).

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.10 | 0.7 | Người ở gần được bao hộp ổn định; so với lần chạy `conf` 0.3, thêm một số người nhỏ ở nền phố bên trái được bao hộp, nhưng nhóm người đi sát nhau ở giữa ảnh có hộp chồng lên nhau (27% số hộp). Toàn video: 63 ID trong file (TrackEval đếm 61 sau khi lọc; nhãn có 62), 22% track ngắn hơn 15 frame. Tracker vẫn chỉ bắt được 25% số người trong nhãn. | `botsort` `conf` 0.5: HOTA 27,17 (DetA 14,3) dù chỉ đổi ID 10 lần. `bytetrack` 0.2: HOTA 27,50. `strongsort` 0.1: đổi ID 197 lần. `ocsort`/`deepocsort` 0.1–0.2: HOTA 22,9–25,8, đổi ID 99–303 lần. `botsort` `iou` 0.5: HOTA 29,52 (kém 0,02) nhưng đổi ID 23 lần thay vì 39, MOTA 21,0 thay vì 20,2 và độ chính xác 75,3 thay vì 71,3. |
| video_2 (phố đêm, tĩnh, rất đông) | botsort | 0.10 | 0.7 | Đèn đường sáng chói nhưng người vẫn rõ; hộp phủ nhiều người đi bộ ở hai bên vỉa hè, người ở rất xa (phía trên ảnh) và người bị cột đèn che nửa thân vẫn dễ không có hộp. Toàn video: 14,3 hộp/frame, 62 ID, track dài trung bình 242 frame, chỉ 2% track ngắn hơn 15 frame (bản lần 1 với `deepocsort`: 107 ID, 31%). Hộp chồng 10,7%. | Điểm thay thế (150 frame đầu), cấu hình nộp 35,12 hạng 1/15: `botsort` `conf` 0.5: 24,98 (7,5 hộp/frame, 32 chỗ đứt). `bytetrack` 0.15: 32,72 (ít hộp hơn khoảng 27%). `strongsort`/`ocsort`/`deepocsort` 0.15: 23,4–24,9 (63–70 chỗ đứt). `iou` 0.5: 34,15 với hộp chồng 0,3% thay vì 12,8%. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.10 | 0.7 | Ảnh 640×480, người đi rất gần camera nên hộp rất lớn, bị cắt ở mép khung và chồng lên nhau; camera di chuyển làm vị trí hộp thay đổi cả khi người đứng yên. Ở khung mẫu có hai hộp chồng lên cùng một người ở giữa ảnh. Toàn video: 6,9 hộp/frame, **168 ID cho 837 frame, 47% track ngắn hơn 15 frame**, hộp chồng 27,1%. Đây là video yếu nhất (bản lần 1 cũng 168 ID, 54% track ngắn). | Điểm thay thế, cấu hình nộp 34,14 hạng 1/15: `botsort` `conf` 0.5: 27,96. `bytetrack` 0.15: 30,61 (chỉ 3,98 hộp/frame). `strongsort`/`ocsort`/`deepocsort` 0.15: 23,5–26,2 (71–88 chỗ đứt). `iou` 0.5: 32,62 nhưng hộp chồng chỉ 3,0% thay vì 26,3%. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.10 | 0.5 | Trung tâm thương mại, sàn bóng và lan can kính phản chiếu ánh đèn. Người đi gần camera được bao hộp lớn, rõ; người ở nền bị người phía trước che. Toàn video: 7,5 hộp/frame, 71 ID, 18% track ngắn hơn 15 frame (bản lần 1 với `conf` 0.4: 66 ID, 26%), hộp chồng 6,4%. Phản chiếu người trên kính cần kiểm tra thêm trên video xem thử (khung mẫu không thấy hộp trên kính). | Điểm thay thế, cấu hình nộp 35,05 hạng 1/15: `iou` 0.4 / 0.7: 35,03 / 34,75 (gần như bằng nhau). `conf` 0.15: 33,89. `conf` 0.5: 27,00. `bytetrack` 0.15: 33,31. `strongsort`/`deepocsort`/`ocsort`: ≤ 28,75. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.10 | 0.7 | Phần lớn khung hình là mặt đường và xe; người đi bộ rất nhỏ (khung mẫu chỉ có hai người được bao hộp) và camera rung. Toàn video: 4,9 hộp/frame, 82 ID, 29% track ngắn hơn 15 frame (bản lần 1 với `strongsort`: 104 ID, 49%), hộp chồng 11,1%. | Điểm thay thế, cấu hình nộp 33,58 hạng 1/15: `iou` 0.5: 33,21 (không có hộp chồng). `botsort` `conf` 0.5: 22,88 (chỉ 3,5 hộp/frame, 62 chỗ đứt). `bytetrack` 0.15: 30,46. `strongsort`/`ocsort`/`deepocsort`: 23,5–27,3 (55–143 chỗ đứt). |

## 2. Số liệu video_1

Cấu hình nộp: `botsort`, `conf` 0.10, `iou` 0.7. Bảng do `scripts/evaluate_practice.py` in ra (toàn văn: `bang_quet/danh_gia_video_1.txt`).

```
HOTA: final_video1-pedestrian      HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            29.543    19.601    44.923    20.663    71.338    48.083    80.394    82.169    30.386    37.116    74.913    27.804

CLEAR: final_video1-pedestrian     MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
video_1                            20.198    79.651    20.408    24.687    85.229    14.516    19.355    66.129    15.175    4587      13994     795       39        9         12        41        70

Identity: final_video1-pedestrian  IDF1      IDR       IDP       IDTP      IDFN      IDFP
video_1                            29.921    19.294    66.611    3585      14996     1797

Count: final_video1-pedestrian     Dets      GT_Dets   IDs       GT_IDs
video_1                            5382      18581     61        62
```

Tóm tắt: **HOTA 29,54 · MOTA 20,20 · IDF1 29,92**. `video_2` đến `video_5` không có nhãn trong gói lab, nên không điền số cho các video đó.

Lưu ý cách chấm: gói dữ liệu không kèm `eval_config.json`, nên `video_1` được chấm bằng cấu hình benchmark mặc định của TrackEval cho định dạng MOTChallenge (nhánh `train`, lớp người đi bộ, bỏ vùng bị đánh dấu bỏ qua trong nhãn). Cách chấm này có thể khác một chút so với số của giảng viên, nhưng so sánh giữa các cấu hình trong báo cáo dùng cùng một cách chấm nên vẫn nhất quán.

## 3. Phân tích

**So với bản lần 1.** Đổi sang `botsort` `conf` 0.10 làm giảm mạnh đứt ID ở `video_2` (107 → 62 ID, track ngắn 31% → 2%) và `video_5` (104 → 82 ID, 49% → 29%), cải thiện `video_4` ở tỉ lệ track ngắn (26% → 18%) dù số ID nhỉnh hơn (66 → 71), còn `video_3` gần như không đổi (168 ID, 54% → 47%). HOTA `video_1` giữ ở 29,5.

**video_1: giới hạn nằm ở detector, tracker chỉ điều chỉnh phần giữ ID.** Trong 19 cấu hình, DetA chỉ 14–23 và recall tối đa ~25%, trong khi AssA dao động 24–52 và số lần đổi ID từ 10 đến ~300: chọn tracker làm đổi mạnh phần giữ danh tính (AssA, IDSW) nhưng gần như không đẩy được phần phát hiện, vì `yolo26n` ở 640 px không bắt được nhiều người nhỏ/xa (nhãn có 18 581 hộp, tracker chỉ xuất 5 382). `conf` điều khiển đúng sự đánh đổi này: ở `conf` 0.5, `botsort` có AssA cao nhất (51,7) và đổi ID ít nhất (10 lần) nhưng DetA tụt về 14,3 nên HOTA chỉ 27,2; hạ xuống 0.1 đưa DetA lên 19,6 và HOTA lên 29,5. `botsort` đứng đầu ở mọi mức `conf` từ 0.1 đến 0.3 (HOTA 29,2–29,5), còn `ocsort`/`deepocsort` tụt mạnh khi hạ `conf` (HOTA 22,9–25,8, đổi ID 99–303 lần). Mình chưa kiểm chứng nguyên nhân của khác biệt này.

**`iou` và hộp chồng.** `iou` là ngưỡng NMS của detector: nâng lên 0.7 cho phép giữ nhiều hộp chồng lên nhau hơn. Ở `video_1`, `iou` 0.7 và 0.5 hòa về HOTA (29,54 so với 29,52), nhưng 0.7 có độ chính xác thấp hơn (71,3 so với 75,3), đổi ID nhiều hơn (39 so với 23), MOTA và IDF1 thấp hơn, và số hộp chồng gấp đôi (20,8% so với 11,0% ở 150 frame đầu). Ở các video không nhãn, hiệu ứng rõ hơn: `video_3` 26,3% so với 3,0% và `video_2` 12,8% so với 0,3%. Điểm thay thế thưởng cho độ phủ nên cũng thưởng cả hộp trùng lặp; vì thế việc chọn `iou` 0.7 ở `video_1`, `video_2`, `video_3`, `video_5` có thể chỉ là hiệu ứng hộp trùng lặp. Đây là điểm yếu đã biết của bản lần 2 (xem mục 4).

**video_2 (đêm, tĩnh, rất đông).** Camera đứng yên nên ít bị lệch vị trí, nhưng cảnh đông và người đi sát nhau khiến hộp hay chồng nhau, rất dễ gán nhầm khi hai người cắt nhau. Bản lần 1 chọn `deepocsort` (có Re-ID) theo suy luận đó, nhưng số liệu cho thấy nhóm `ocsort`/`deepocsort`/`strongsort` có 63–70 chỗ đứt trong 150 frame so với 6–11 của `botsort` ở độ phủ tương đương (khoảng 11–14 hộp/frame). `conf` thấp đóng vai trò rõ: `botsort` 0.5 chỉ còn 7,5 hộp/frame và 32 chỗ đứt, vì ban đêm người ở xa có điểm tin cậy thấp và bị loại sớm.

**video_4 (trong nhà, camera tiến tới).** Camera di chuyển làm chuyển động của từng người không còn tuyến tính, nên tracker chỉ dựa vào chuyển động dễ mất người; `botsort` có bù chuyển động camera và Re-ID, và đứng đầu điểm thay thế. Bản lần 1 dùng `conf` 0.4 để hạn chế hộp giả từ sàn và kính phản chiếu, nhưng số liệu cho thấy `conf` cao làm track đứt nhiều hơn (38 chỗ đứt ở 0.4 so với 5 ở 0.1). Số đếm không phân biệt được hộp giả với hộp thật, nên chưa biết `conf` 0.1 có sinh hộp giả từ phản chiếu hay không; cần xem video xem thử. `iou` hầu như không ảnh hưởng ở video này (hộp chồng 0% ở cả 0.5 và 0.7).

**Giới hạn của kết luận.** Với `video_2`–`video_5`, so sánh dựa trên điểm thay thế trong 150 frame đầu cộng một khung hình mẫu mỗi video, không phải HOTA. Điểm này mới được kiểm chứng trên một video có nhãn, nên có thể lệch ở cảnh khác (ví dụ camera rung). `conf` 0.1 là mức thấp nhất đã thử và đứng đầu ở mọi video, nên mức tối ưu có thể còn thấp hơn. `video_3` vẫn là cảnh yếu nhất (168 ID, 47% track ngắn, 27% hộp chồng).

## 4. Nếu có thêm thời gian

1. Đổi `iou` về 0.5 cho `video_1`, `video_2`, `video_3`, `video_5` và chạy lại đủ frame: ở `video_1` đây là cấu hình HOTA ngang bằng nhưng ít đổi ID, ít hộp chồng hơn; ở `video_3` có thể giảm mạnh số hộp trùng lặp.
2. Thử `conf` dưới 0.1 (0.05) vì 0.1 là biên của lưới đã quét.
3. Xem video xem thử của `video_3` và `video_4` để kiểm tra hộp giả và hộp trùng lặp, và xem lại các frame mà ID đổi để biết lỗi do detector bỏ sót hay do tracker gán nhầm.
4. Việc cải thiện DetA cần detector hoặc kích thước ảnh tốt hơn, nằm ngoài luật chơi của lab này; thử Re-ID khác chỉ nên làm như phần mở rộng (ghi rõ ngoài bài nộp chính).
