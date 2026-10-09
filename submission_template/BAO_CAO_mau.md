# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Cá nhân **Thành viên:** Nguyễn Thanh Hòa (2A202602559)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Đây là **bản nộp lần 1**. Quy trình: chạy 5 video đủ frame với cấu hình chọn theo đặc điểm cảnh, sau đó quét 5 tracker (`bytetrack`, `ocsort`, `botsort`, `strongsort`, `deepocsort`) bằng `scripts/sweep.py` để kiểm tra lại.
- `video_1` (có nhãn): 5 tracker × `conf` {0.1, 0.2, 0.3} ở `iou` 0.5, đủ 600 frame, rồi quét `iou` {0.4, 0.7} quanh tracker tốt nhất; chấm HOTA / MOTA / IDF1 bằng TrackEval. Bảng 17 cấu hình: `bang_quet/sweep_video_1.md`. Cấu hình nộp của `video_1` lấy từ bảng này.
- `video_2`–`video_5` (không nhãn): 5 tracker × `conf` {0.15, 0.3} ở `iou` 0.5, 150 frame đầu. Không có HOTA, nên xếp hạng bằng **điểm thay thế** tính từ file kết quả. Điểm này được kiểm chứng trên `video_1` (có nhãn, 17 cấu hình): độ phủ hộp/frame cao **không** đi cùng HOTA cao (tương quan −0,52, vì `conf` thấp sinh nhiều hộp và nhiều lần đổi ID) và số ID cũng không dự báo tốt (−0,38). Chỉ báo tốt nhất là **số chỗ track bị đứt quãng** (một ID có frame trống ở giữa các lần xuất hiện): tương quan −0,76 với HOTA, −0,93 với AssA và +0,99 với số lần đổi ID. Khớp HOTA ≈ 24,5 + 13,4·phủ_chuẩn_hoá − 14,4·đứt_chuẩn_hoá trên 17 cấu hình, kiểm tra bỏ-một-ra (leave-one-out) cho tương quan 0,82 và cấu hình được chọn đứng hạng 5/17 (HOTA kém cấu hình tốt nhất 0,8). Điểm này chỉ dùng để xếp hạng, không phải số HOTA, và mới được kiểm chứng trên một video có nhãn. Bảng: `bang_quet/sweep_video_N.md`.
- Cấu hình nộp của `video_2`–`video_5` trong bản này được chọn **trước** khi quét, theo đặc điểm cảnh. Điểm thay thế cho thấy cả bốn video còn cấu hình tốt hơn (cột cuối và mục 4); chưa thay vào bản này.
- **Tái lập:** notebook `2A202602559-NguyenThanhHoa.ipynb` đã chạy trọn vẹn hai lần trên Kaggle (có GPU, Python 3.11 trong venv). Hai lần cho 5 file `video_N.txt` giống hệt nhau (so sánh md5), bảng quét và điểm `video_1` giống nhau, nên kết quả nộp là tất định. `video_5.txt` có track ở 746/750 frame: 4 frame không có hộp nào, nhưng cả 750 ảnh đều đã được xử lý.

## 1. Cấu hình đã chọn

Trong cột quan sát: "toàn video" là thống kê trên file nộp đủ frame; "150 frame đầu" là cùng file nhưng chỉ lấy 150 frame đầu, để so sánh được với bảng quét.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại / đã quét |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.10 | 0.5 | Người ở gần được bao hộp ổn định; nhóm người đi sát nhau ở giữa ảnh vẫn giữ riêng từng hộp. Người ở xa và nhỏ (nhiều người ở nền phố bên trái) phần lớn không có hộp: tracker chỉ bắt được 24% số người trong nhãn, còn ID thì ít đổi (23 lần đổi ID trong 600 frame). | `bytetrack` 0.2: ít đổi ID nhất (12) nhưng DetA thấp nhất (15,4), HOTA 27,50. `strongsort` 0.1: đổi ID 197 lần, MOTA 15,6. `ocsort`/`deepocsort` 0.1: đổi ID ~300 lần, HOTA ~23. `botsort` 0.3 (HOTA 29,46) và `iou` 0.4 / 0.7 (HOTA 29,33 / 28,73): kém hơn một chút. |
| video_2 (phố đêm, tĩnh, rất đông) | deepocsort | 0.20 | 0.6 | Cảnh có đèn đường sáng chói nhưng người vẫn rõ; hộp phủ nhiều người đi bộ ở hai bên vỉa hè. Người ở rất xa (phía trên ảnh) và người bị cột đèn che nửa thân dễ không có hộp. Toàn video: 14,3 hộp/frame, 107 ID, 31% track ngắn hơn 15 frame. 150 frame đầu: 12,95 hộp/frame, 31 ID, track 62,7 frame. | Điểm thay thế (xem đầu báo cáo) xếp cấu hình nộp hạng 9/11 (150 frame đầu: 12,95 hộp/frame, **63 chỗ đứt quãng**). Hạng 1: `botsort` 0.15 (11,47 hộp/frame, 10 chỗ đứt, 21 ID); hạng 2–3: `bytetrack` 0.15 / 0.3 (2–3 chỗ đứt nhưng chỉ 8,75–9,18 hộp/frame). `strongsort`/`ocsort`/`deepocsort` 0.15 (~14 hộp/frame): 63–70 chỗ đứt, 30–38 ID. Chưa thay vào bản này. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.25 | 0.5 | Ảnh 640×480, người đi rất gần camera nên hộp rất lớn, bị cắt ở mép khung và chồng lên nhau; camera di chuyển làm vị trí hộp thay đổi cả khi người đứng yên. Toàn video: 5,8 hộp/frame, **168 ID cho 837 frame, 54% track ngắn hơn 15 frame** (đứt ID nhiều). 150 frame đầu: 5,23 hộp/frame, 22 ID, track 35,7 frame. | Điểm thay thế xếp cấu hình nộp hạng 3/11 (150 frame đầu: 5,23 hộp/frame, 26 chỗ đứt, 22 ID), sát hạng 1: `botsort` 0.15 (5,65 hộp/frame, 20 chỗ đứt, 21 ID). `strongsort`/`ocsort`/`deepocsort` 0.15 phủ cao hơn (~7,0 hộp/frame) nhưng 71–88 chỗ đứt, 24–37 ID. `bytetrack` 0.15: chỉ 3,98 hộp/frame. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.40 | 0.5 | Trung tâm thương mại, sàn bóng và lan can kính phản chiếu ánh đèn. Người đi gần camera được bao hộp lớn, rõ; người ở nền bị người phía trước che. Toàn video: 6,7 hộp/frame, 66 ID, 26% track ngắn hơn 15 frame. 150 frame đầu: 5,85 hộp/frame, 16 ID, track 54,9 frame. Phản chiếu người trên kính cần kiểm tra thêm trên video xem thử (khung mẫu không thấy hộp trên kính). | Điểm thay thế xếp cấu hình nộp hạng 9/11: ở `conf` 0.4 có 38 chỗ đứt (5,85 hộp/frame); hạ xuống 0.3 còn 24 chỗ đứt, xuống 0.15 còn **9 chỗ đứt** và phủ 6,91 hộp/frame (hạng 1, 16 ID). `bytetrack` 0.15: 3 chỗ đứt nhưng chỉ 5,63 hộp/frame. `strongsort`/`deepocsort`/`ocsort` 0.15: 33–54 chỗ đứt, 31–35 ID. Chưa thay vào bản này. |
| video_5 (trên xe bus, giao lộ đông) | strongsort | 0.30 | 0.5 | Phần lớn khung hình là mặt đường và xe; người đi bộ rất nhỏ (khung mẫu chỉ có hai người được bao hộp) và camera rung. Toàn video: 4,2 hộp/frame, **104 ID cho 750 frame, 49% track ngắn hơn 15 frame**. 150 frame đầu: 6,33 hộp/frame, 30 ID, track 31,7 frame. | Điểm thay thế xếp cấu hình nộp hạng 5/10 (150 frame đầu: 6,33 hộp/frame, 55 chỗ đứt, 30 ID). Hạng 1: `botsort` 0.15 (6,63 hộp/frame, **10 chỗ đứt**, 27 ID); hạng 2: `bytetrack` 0.15 (1 chỗ đứt, 4,62 hộp/frame). `strongsort` 0.15: phủ ~10 hộp/frame nhưng 105 chỗ đứt, 60 ID; `ocsort`/`deepocsort` 0.15: 123–143 chỗ đứt. Chưa thay vào bản này. |

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

Lưu ý cách chấm: gói dữ liệu không kèm `eval_config.json`, nên `video_1` được chấm bằng cấu hình benchmark mặc định của TrackEval cho định dạng MOTChallenge (nhánh `train`, lớp người đi bộ, bỏ vùng bị đánh dấu bỏ qua trong nhãn). Cách chấm này có thể khác một chút so với số của giảng viên, nhưng so sánh giữa các cấu hình trong báo cáo dùng cùng một cách chấm nên vẫn nhất quán.

## 3. Phân tích

**video_1: giới hạn nằm ở detector, không phải tracker.** Trong 17 cấu hình, DetA chỉ nằm trong khoảng 15–23 và recall tối đa ~24%, trong khi AssA dao động 24–49 và số lần đổi ID từ 12 đến ~300. Nghĩa là việc chọn tracker làm thay đổi mạnh phần giữ danh tính (AssA, IDSW) nhưng gần như không đẩy được phần phát hiện (DetA), vì detector `yolo26n` ở 640 px không bắt được nhiều người nhỏ/xa (nhãn có 18 581 hộp, tracker chỉ xuất 5 016). Hạ `conf` từ 0.3 xuống 0.1 làm DetA tăng 18,1 → 19,6 nhưng độ chính xác giảm (DetPr 78,9 → 75,3), nên HOTA gần như không đổi (29,46 → 29,52). Vì vậy chênh lệch giữa các cấu hình tốt nhất là nhỏ.

**video_2 (đêm, tĩnh, rất đông).** Camera đứng yên nên ít bị lệch vị trí, nhưng cảnh đông và người đi sát nhau khiến hộp hay chồng nhau. Mình chọn `deepocsort` (có Re-ID) với ý nghĩ rằng khi hai người cắt nhau, chỉ dựa vào chuyển động rất dễ gán nhầm. Số liệu không ủng hộ lựa chọn đó: trong 150 frame đầu, `deepocsort` 0.2 có 63 chỗ đứt quãng so với 10 của `botsort` 0.15 ở độ phủ tương đương (12,95 so với 11,47 hộp/frame), và cả nhóm `ocsort`/`deepocsort`/`strongsort` ở 0.15 đều có 63–70 chỗ đứt. Điều này khớp với kết quả có nhãn ở `video_1`, nơi `botsort` đứng đầu ở cả ba mức `conf` (HOTA 29,2–29,5) còn `ocsort`/`deepocsort` tụt mạnh khi hạ `conf` (HOTA 22,9–25,8, đổi ID 99–303 lần). Mình chưa kiểm chứng nguyên nhân (ví dụ vì sao Re-ID của `deepocsort` không giúp ở cảnh đêm đông); đây là bằng chứng gián tiếp và cần xem video xem thử để xác nhận.

**video_4 (trong nhà, camera tiến tới).** Camera di chuyển làm chuyển động của từng người không còn tuyến tính, nên tracker chỉ dựa vào chuyển động dễ mất người; `botsort` có bù chuyển động camera và Re-ID, và đứng đầu điểm thay thế. Cấu hình nộp dùng `conf` 0.4 với ý định hạn chế hộp giả từ sàn và kính phản chiếu, nhưng số liệu cho thấy `conf` cao làm track đứt nhiều hơn: 38 chỗ đứt ở 0.4, 24 ở 0.3, 9 ở 0.15 (độ phủ 5,85 → 6,39 → 6,91 hộp/frame). Số đếm không phân biệt được hộp giả với hộp thật, nên chưa biết hạ `conf` có thực sự sinh hộp giả từ phản chiếu hay không; cần xem video xem thử.

**Giới hạn của kết luận.** Với `video_2`–`video_5` không có nhãn, so sánh dựa trên điểm thay thế trong 150 frame đầu cộng một khung hình mẫu mỗi video, không phải HOTA. Điểm thay thế chỉ được kiểm chứng trên `video_1`, nên có thể lệch ở cảnh khác (ví dụ cảnh camera rung). Ngoài ra `video_3` và `video_5` có tỉ lệ track ngắn rất cao ở bản nộp (54% và 49% ngắn hơn 15 frame), cho thấy hai cảnh camera di chuyển/rung này là chỗ cấu hình hiện tại yếu nhất.

## 4. Nếu có thêm thời gian

Theo điểm thay thế, `botsort` `conf` 0.15 là cấu hình đứng đầu ở cả bốn video không nhãn (bản nộp lần 1 chưa dùng). Bản cập nhật sẽ chạy lại notebook với bước quét đầy đủ hơn theo gợi ý của hướng dẫn: ba giai đoạn (tracker × `conf` → `conf` {0.1, 0.2, 0.5} → `iou` {0.4, 0.7}) cho từng video, rồi chạy đủ frame và xem video xem thử để kiểm tra hộp giả. Việc cải thiện DetA cần detector hoặc kích thước ảnh tốt hơn, nằm ngoài luật chơi của lab này. Thử Re-ID khác chỉ nên làm như phần mở rộng (ghi rõ ngoài bài nộp chính).
