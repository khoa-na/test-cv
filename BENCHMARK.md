# Detection benchmark

Ngày chạy: 2026-07-25

CPU: Intel Core i5-13400F

Dataset: Pothole-600 official testing split, 180 ảnh / 196 instances
Input: 512x512, batch 1, ONNX Runtime CPU

## Final ONNX (`end2end=False`)

| Metric | Value |
|---|---:|
| Box precision | 85.6% |
| Box recall | 80.1% |
| Box mAP@0.5 | 85.5% |
| Box mAP@0.5:0.95 | 49.7% |
| Mask precision | 82.8% |
| Mask recall | 77.6% |
| Mask mAP@0.5 | 81.2% |
| Mask mAP@0.5:0.95 | 46.8% |
| Preprocess | 0.7 ms |
| ONNX inference | 25.9 ms |
| Postprocess | 1.7 ms |
| Total | 28.3 ms / 35.3 FPS |

SHA256:

```text
8adf84f81c854b74883e25e8280c0f4f33c633167fb906f028875a733e50cf0a
```

## Đối chứng

| Configuration | Box mAP@0.5 | Mask mAP@0.5 |
|---|---:|---:|
| PyTorch, `end2end=False` | 86.9% | 81.8% |
| ONNX, `end2end=False` | 85.5% | 81.2% |
| ONNX, end-to-end top-300 | 74.1% | 71.9% |

ONNX raw head được chọn vì đạt KPI accuracy và vẫn vượt yêu cầu tốc độ của
module detection. FPS toàn pipeline phải được đo lại sau khi thêm depth/area.

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
3 lần.

| Metric held-out | Kết quả |
|---|---:|
| Detection/fusion coverage | 100% / 100% |
| Median depth error | 4,97% |
| Depth trong ±15% | 100% |
| Median area error | 11,61% |
| Area trong ±15% | 89,47% |
| Depth + area cùng trong ±15% | 89,47% |
| Median latency / FPS end-to-end | 57,3 ms / 17,45 FPS |

Đối chứng CPU cho thấy cấu hình sequential dùng khoảng 72% tổng 16 logical
CPU. Pipeline hai stage dùng khoảng 91% nhưng chậm hơn do ONNX Runtime và
OpenCV tranh core; tăng utilization không đồng nghĩa tăng throughput. Scale
0,3125 tạo working width 200 px và 112 disparities, nhanh hơn cấu hình
0,325/128 đồng thời tăng tỷ lệ held-out đạt cả depth và area trong ±15%.
