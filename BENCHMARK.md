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
