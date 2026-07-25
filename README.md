# Pothole depth and area estimation

Pipeline CPU/ONNX phát hiện ổ gà, tạo segmentation mask và ghép depth để ước
lượng độ sâu, diện tích. Hiện repository đã hoàn thành model
`YOLO26n-seg`; module depth/area sẽ dùng chính mask này làm ROI.

## Model đã chốt

- Model: `models/pothole_yolo26n_seg.onnx`
- Input: `1x3x512x512`
- Output: raw detections `1x37x5376` và mask prototypes `1x32x128x128`
- Export: ONNX opset 18, `simplify=True`, `end2end=False`
- Runtime: ONNX Runtime `CPUExecutionProvider`

Benchmark trên Pothole-600 official test split (180 ảnh, 196 instances),
i5-13400F:

| Metric | Kết quả |
|---|---:|
| Box mAP@0.5 | 85.5% |
| Box mAP@0.5:0.95 | 49.7% |
| Mask mAP@0.5 | 81.2% |
| Mask mAP@0.5:0.95 | 46.8% |
| Preprocess + inference + postprocess | 28.3 ms |
| Model pipeline FPS | 35.3 |

Model đạt KPI detection `mAP@0.5 >= 80%` và tốc độ riêng model `>= 15 FPS`.
FPS end-to-end sẽ được đo lại sau khi ghép depth/area.

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Train lại

Tải và giải nén Pothole-600 vào `.cache/data/pothole600`, sau đó:

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/python train_pothole600.py \
  --dataset .cache/data/pothole600 \
  --model yolo26n-seg.pt \
  --output artifacts/yolo26n-seg \
  --device 0 --batch 16 --epochs 100 --patience 0
```

## Export ONNX

`end2end=False` là bắt buộc để dùng raw one-to-many head; cấu hình end-to-end
top-300 cho accuracy thấp hơn trên test split.

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/yolo export \
  model=artifacts/yolo26n-seg/train/weights/best.pt \
  format=onnx imgsz=512 batch=1 device=cpu \
  opset=18 simplify=True dynamic=False end2end=False
```

## Kiểm thử trainer

```bash
.venv/bin/python train_pothole600.py \
  --dataset .cache/data/pothole600 --prepare-only
```

## Giới hạn hiện tại

- Model có một class `pothole`; severity sẽ được suy ra từ depth và area.
- Monocular depth chỉ cho relative depth nếu chưa hiệu chuẩn camera/ground plane.
- Kết quả FPS hiện chưa bao gồm depth inference và visualization.

## Audit PothRGBD

PothRGBD được giữ ngoài Git tại `.cache/data/pothrgbd`. Kiểm tra pairing,
định dạng depth và metadata calibration bằng:

```bash
.venv/bin/python audit_pothrgbd.py \
  --dataset ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET" \
  --output artifacts/pothrgbd-audit.json
```

Không quy đổi depth sang mét hoặc area sang m² nếu báo cáo audit chưa xác nhận
được `depth_scale` và camera intrinsics.

Kết quả audit archive công khai:

| Hạng mục | Kết quả |
|---|---:|
| RGB / depth / label | 1.000 / 1.000 / 1.000 |
| Triplet ghép chắc chắn | 996 |
| Triplet sẵn sàng cho hình học | 981 |
| Depth shape và dtype | 480x640 `uint16` |
| Segmentation instances | 1.097 |
| Calibration/depth scale metadata | Không có |

15 RGB đã bị crop trong khi depth giữ nguyên 480x640 và 4 mẫu có timestamp
trùng nên bị loại khỏi benchmark hình học. Depth có dạng raw sensor nhưng đơn
vị metric chưa được xác minh từ metadata của dataset.
