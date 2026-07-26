# Detection benchmark

Ngày chạy: 2026-07-26 (detection và stereo), 2026-07-25 (depth monocular)

CPU: Intel Core i5-13400F

Dataset: Pothole-600 official testing split, 180 ảnh / 196 instances
Input: 512x512, batch 1, ONNX Runtime CPU

## Final ONNX (`end2end=False`)

Model production train trên tập gộp Pothole-600 + PothRGBD (150 epochs,
`copy_paste=0.3`). Val vẫn chạy trên Pothole-600 official test split để so
trực tiếp với baseline chỉ-Pothole-600.

| Metric | Value |
|---|---:|
| Box precision | 87.9% |
| Box recall | 81.4% |
| Box mAP@0.5 | 89.8% |
| Box mAP@0.5:0.95 | 56.3% |
| Mask precision | 87.4% |
| Mask recall | 81.5% |
| Mask mAP@0.5 | 87.1% |
| Mask mAP@0.5:0.95 | 52.5% |
| Preprocess | 0.4 ms |
| ONNX inference | 25.7 ms |
| Postprocess | 3.3 ms |
| Total | 29.4 ms / 34.0 FPS |

SHA256:

```text
3ab52bdc4b41cc59b4b845b090bcddc2d927870ce272fdcff2dbb473b3a598c5
```

## Đối chứng

| Configuration | Box mAP@0.5 | Mask mAP@0.5 |
|---|---:|---:|
| ONNX merged (production) | 89.8% | 87.1% |
| ONNX baseline Pothole-600 | 85.5% | 81.2% |
| PyTorch baseline, `end2end=False` | 86.9% | 81.8% |
| ONNX baseline, end-to-end top-300 | 74.1% | 71.9% |

ONNX raw head được chọn vì đạt KPI accuracy và vẫn vượt yêu cầu tốc độ của
module detection.

Ứng viên thứ ba — fine-tune giai đoạn 2 trên riêng Pothole-600 từ checkpoint
merged — đạt box mAP@0.5 cao nhất (90.2%) nhưng bị loại: mAP@0.5:0.95 thấp hơn
(box 49.3% / mask 47.6%) và trên stereo thật nó bỏ sót hẳn một ổ gà
(detection coverage 96.3%, end-to-end 94.7%). mAP@0.5 chỉ cần IoU 0.5 nên
không phản ánh độ khít biên mask, thứ mà phép đo diện tích phụ thuộc trực tiếp.
Baseline cũ được giữ tại `artifacts/final-baseline-pothole600/`.

## Depth Anything V2 Small

Dataset: PothRGBD, GT segmentation mask + RealSense raw depth

Split: deterministic 80% calibration / 20% test
Input: 196x196, ONNX Runtime CPU, 6 threads

| Metric | Raw global scale | Ground-plane normalized |
|---|---:|---:|
| Test instances evaluated | 194 | 193 |
| Coverage | 86,2% | 85,8% |
| Mean relative error | 188,8% | 116,8% |
| Median relative error | 72,6% | 56,6% |
| Within ±15% | 8,8% | 12,4% |
| Within ±8% | 6,2% | 6,7% |

Depth inference median là 62,6 ms, tương đương 16,0 FPS. Relative Depth
Anything V2 Small không đạt KPI depth error <=15% trên PothRGBD; tăng ngưỡng
độ sâu tối thiểu cũng không đưa median error về gần 15%. Kết quả dùng raw
sensor units vì dataset không cung cấp `depth_scale` hay camera intrinsics.

### Metric Outdoor Small

Model `depth_anything_v2_vits_outdoor_dynamic.onnx` được đánh giá bằng cùng
protocol. Full test ở 196px giảm ground-plane median error còn 46,9%, đạt
16,5% instances trong ±15%, coverage 96,9% và 15,9 FPS.

| Input | Phạm vi | Median error | Within ±15% | FPS |
|---|---:|---:|---:|---:|
| 196 | 967 ảnh | 46,9% | 16,5% | 15,9 |
| 392 | 200 ảnh smoke | 47,0% | 20,0% | 4,9 |
| 518 | 200 ảnh smoke | 41,5% | 18,2% | 3,0 |

Metric Outdoor cải thiện baseline nhưng vẫn không đạt A2. Tăng resolution
không đủ bù domain gap và làm mất KPI 15 FPS, nên cấu hình 196px được giữ làm
baseline monocular; hướng RGB-D/stereo cần được ưu tiên cho phép đo metric.

## YOLO + stereo metric

Dataset: Fan stereo pothole, 27 pair / 3 potholes vật lý

Protocol: `model1` calibration; `model2` và `model3` held-out (19 pair).
Stereo working scale 0,3125; 112 disparities; 4 OpenCV threads; mỗi pair đo
3 lần. Detector là ONNX production (merged).

Hằng số hiệu chuẩn, tất cả fit trên `model1`:
`metric_scale=0,8334711918061039`, `area_scale=1,3755604448201877` (đường
residual fallback). Đường mask YOLO dùng hệ số 1,0 (xem bên dưới).

| Metric held-out | Baseline cũ | Hiện tại |
|---|---:|---:|
| Detection/fusion coverage | 100% / 100% | 100% / 100% |
| Strong alignment | 66,7% | 89,5% |
| Fallback rate | 33,3% | 10,5% |
| Median depth error | 4,97% | **4,01%** |
| Mean depth error | — | 3,88% |
| Depth trong ±8% / ±15% | — / 100% | 100% / 100% |
| Median area error | 11,61% | 11,23% |
| Area trong ±8% / ±15% | — / 89,47% | 26,32% / **84,21%** |
| Area p95 / max | — | 18,3% / 18,7% |
| Depth + area cùng trong ±15% | 89,47% | 84,21% |

Chênh lệch depth đến từ hai thay đổi:

1. **Fit mặt đường hai lượt.** Lượt một fit trên toàn bộ pixel valid, kể cả
   pixel trong hố, nên mặt phẳng bị võng xuống và làm hụt cả depth lẫn area.
   Lượt hai fit lại sau khi loại pixel thuộc residual mask
   (`fit_road_disparity(..., exclude=)`). Đây là bản rút gọn của bước tách
   vùng đường lành bằng Otsu trong Fan et al. 2019. Chi phí ~2 ms geometry.
2. **Resize SGBM bằng `INTER_LINEAR`** (đổi từ `INTER_AREA` khi tối ưu FPS):
   nhanh hơn ~2,6 ms và tình cờ giảm depth median 4,42% → 4,01%.

Về diện tích: hệ số `yolo_area_scale=0,9096` (fit trên `model1` cho đường
mask YOLO) từng được thử vì model merged kéo fallback rate xuống thấp, khiến
hơn 90% phép đo diện tích chạy qua đường không hiệu chuẩn. Nó nén phần giữa
phân bố (median 9,48%, ±8% lên 42,11%) nhưng kéo dài đuôi lỗi — p95 18,3% →
19,1%, max 18,7% → 20,4%, ±15% tụt 84,21% → 78,95% — vì `model1`/`model2` cần
hệ số ~0,90–0,92 còn `model3` cần ~1,03. Nghiệm thu chấm theo ngưỡng ±8/±15
và hệ số này fit trên đúng một ổ gà vật lý, nên mặc định giữ 1,0; giá trị
0,9096 vẫn dùng được qua `--yolo-area-scale`.

### Các hướng đã thử và loại

| Hướng | Kết quả | Lý do loại |
|---|---:|---|
| Convex hull mask trước khi tính diện tích | area median 11,0% | Mask đang over-estimate 9%, hull thổi thêm |
| Bilinear thay nearest khi upsample mask | area median 10,45% | Kém nearest |
| Mặt đường bậc 2 thay mặt phẳng | — | Tín hiệu/sai-số-mô-hình đã 100:1 |
| Đổi thống kê depth p90 → p95 / span | p95 median 1,98% | Mất mốc 100% trong ±15% |
| Fit lại `metric_scale` = 0,8518 | median 3,11% | Trong ±8% tụt 100% → 74%, mean xấu hơn |
| `image_scale` 0,3125 → 0,625 | mean 3,80% | FPS 16,5 → 6,1 |
| `yolo_area_scale` = 0,9096 | area median 9,48% | Đuôi lỗi dài ra: p95 19,1%, ±15% tụt còn 78,95% |

(Các dòng area phía trên đo trong giai đoạn còn dùng `yolo_area_scale=0,9096`;
so sánh tương đối giữa các dòng vẫn giữ nguyên giá trị.)

Mặt đường phẳng trong không gian 3D chiếu sang không gian disparity là một mặt
**tuyến tính** đúng chính xác, nên fit tuyến tính hiện tại không phải xấp xỉ.
Đo residual RMS của fit trên pixel đường: 0,061–0,081 px, trong khi tín hiệu
residual trong hố là 4,4–8,1 px — tỷ lệ ~100:1. Bậc 2 chỉ giảm thêm ~0,01 px,
tương đương ~0,15% relative, không đáng đổi 6 tham số trên 3 ổ gà vật lý.

Quét `image_scale` cho thấy độ phân giải disparity không phải nút thắt: gấp 4
lần số pixel trong hố (2 067 → 8 106) chỉ giảm mean error 4,07% → 3,80%.

### Sàn sai số còn lại

Phân tích sai số **có dấu** theo từng bộ stereo:

| Bộ | n | GT depth | Sai số có dấu | Độ tán trong bộ |
|---|---:|---:|---:|---:|
| model1 | 8 | 41,6 mm | −1,37% | 1,99% |
| model2 | 14 | 40,3 mm | +1,58% | 4,07% |
| model3 | 5 | 23,7 mm | −5,11% | 0,54% |

`model3` lệch −5,11% với độ tán chỉ 0,54%: phép đo chính xác cao nhưng bị sai
một hệ số hằng. Đó là dấu hiệu sai hiệu chuẩn rig, không phải nhiễu thị giác.
Bias giữa các bộ trải −5,1% đến +1,6% trong khi độ chính xác nội bộ mỗi bộ là
0,5–4%, nên một `metric_scale` toàn cục không hấp thụ được.

Quét `metric_scale` trong dải 0,80–0,875 cho thấy `model1` (n=8, một ổ gà) đạt
100% trong ±8% trên toàn dải — tập hiệu chuẩn không đủ sức phân biệt. Chọn
hằng số theo held-out sẽ là rò rỉ test set, nên giá trị cũ được giữ nguyên.

Kết luận: phần sai số depth còn lại thuộc về hiệu chuẩn rig và GT proxy, không
phải thuật toán. Vì diện tích tỷ lệ với bình phương độ sâu, sai số area xấp xỉ
gấp đôi sai số depth (3,88% × 2 ≈ 7,8%, đo được mean 10,26%), nên area cũng
đang ở gần sàn của nó.

### Ghi chú throughput

Đối chứng CPU cho thấy cấu hình sequential dùng khoảng 72% tổng 16 logical
CPU. Pipeline hai stage dùng khoảng 91% nhưng chậm hơn do ONNX Runtime và
OpenCV tranh core; tăng utilization không đồng nghĩa tăng throughput. Chạy
detect và geometry song song **trong cùng frame** cũng chết cùng kiểu: mọi
stage chậm gấp đôi kể cả khi ghim ORT 6–10 intra-op threads (thread bị đẩy
sang E-core của i5-13400F). Scale 0,3125 tạo working width 200 px và 112
disparities, nhanh hơn cấu hình 0,325/128 đồng thời tăng tỷ lệ held-out đạt
cả depth và area trong ±15%.

Hai tối ưu latency 2026-07-26, cả hai không đổi (hoặc cải thiện) A2:

1. `fit_road_disparity` rewrite — subsample lưới 4×4 thay `nonzero` toàn ảnh,
   broadcast thay `np.mgrid`: 4,4 → 2,1 ms mỗi lượt fit (×2 lượt).
2. Resize input SGBM `INTER_AREA` → `INTER_LINEAR`: −2,6 ms, depth median
   giảm 4,42% → 4,01%.

INT8 static quantization (QDQ, loại head `model.23`) được thử và loại: ORT
session 22,2 → 17,4 ms nhưng end-to-end chỉ lời ~2 ms, đổi lại area ±15% tụt
78,9% → 68,4% (một số mask xê dịch biên). A1 của bản INT8 vẫn đạt (box mAP@0.5
89,6%, mask 88,2%) nên đây là phương án dự phòng nếu cần FPS trên máy yếu hơn.

FPS đo trên máy rảnh sau tối ưu 1 (chưa gồm tối ưu 2): **56,2 ms / 17,8 FPS,
100% pair ≥15 FPS** — baseline cùng phiên là 61,0 ms / 16,4 FPS / 85,2%.
Cần một lần đo chốt cuối trên máy rảnh để gộp cả tối ưu 2 (~2 ms).
