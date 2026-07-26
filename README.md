# Pothole depth and area estimation

Pipeline CPU/ONNX phát hiện ổ gà, tạo segmentation mask và ghép depth để ước
lượng độ sâu, diện tích. Pipeline cuối dùng `YOLO26n-seg` để định vị và
StereoSGBM để đo hình học so với mặt đường — xem
[Stereo depth và area metric](#stereo-depth-và-area-metric).

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
| Box mAP@0.5 | 89.8% |
| Box mAP@0.5:0.95 | 56.3% |
| Mask mAP@0.5 | 87.1% |
| Mask mAP@0.5:0.95 | 52.5% |
| Preprocess + inference + postprocess | 29.4 ms |
| Model pipeline FPS | 34.0 |

Model đạt KPI detection `mAP@0.5 >= 80%` và tốc độ riêng model `>= 15 FPS`.
Weights train trên tập gộp Pothole-600 + PothRGBD; val chạy trên Pothole-600
official test split nên so trực tiếp được với baseline chỉ-Pothole-600
(85.5% box / 81.2% mask), bản cũ giữ tại `artifacts/final-baseline-pothole600/`.

## Cài đặt

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Train lại

Tải và giải nén Pothole-600 vào `.cache/data/pothole600`, sau đó:

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/python -m training.train_pothole600 \
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
.venv/bin/python -m training.train_pothole600 \
  --dataset .cache/data/pothole600 --prepare-only
```

## Giới hạn hiện tại

- Model có một class `pothole`; severity sẽ được suy ra từ depth và area.
- Monocular depth chỉ cho relative depth nếu chưa hiệu chuẩn camera/ground plane
  và không đạt KPI A2, nên nhánh stereo được chọn cho phép đo metric.
- Stereo benchmark chỉ có 3 ổ gà vật lý; `metric_scale` bị under-determined bởi
  tập hiệu chuẩn một ổ gà và sai số còn lại chủ yếu là bias hiệu chuẩn rig.
- Số FPS trong tài liệu đo trên máy rảnh; lần chạy 2026-07-26 bị nhiễu tải máy.

## Audit PothRGBD

PothRGBD được giữ ngoài Git tại `.cache/data/pothrgbd`. Kiểm tra pairing,
định dạng depth và metadata calibration bằng:

```bash
.venv/bin/python -m data_tools.audit_pothrgbd \
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

## Convert PothRGBD sang YOLO-seg

Label PothRGBD đã ở dạng polygon chuẩn hoá nên converter chỉ lọc triplet hợp
lệ (ảnh 480x640, pairing duy nhất, polygon hợp lệ) và dựng cấu trúc dataset
Ultralytics:

```bash
.venv/bin/python -m data_tools.convert_pothrgbd_yoloseg \
  --dataset ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET" \
  --output .cache/data/pothrgbd-yoloseg
```

Kết quả: 768 ảnh train / 213 ảnh val (848 / 229 instances), 17 mẫu bị loại
theo đúng báo cáo audit. Split val dùng lại hash split của
`benchmarks.benchmark_depth_pothrgbd` (seed 42, fraction 0.2) nên trùng với
test split của depth benchmark — ảnh depth-test không bao giờ lọt vào tập
train detection. Dataset YAML: `.cache/data/pothrgbd-yoloseg/pothrgbd-seg.yaml`
(một class `pothole`); ảnh được symlink về archive gốc nên không tốn dung lượng.

## Train hỗn hợp Pothole-600 + PothRGBD

Tạo YAML gộp (train = Pothole-600 train + PothRGBD train; val/test giữ nguyên
Pothole-600 để mAP so sánh được với baseline 85.5%):

```bash
.venv/bin/python -m training.train_pothole600 --dataset .cache/data/pothole600 \
  --prepare-only --output artifacts/yolo26n-seg
.venv/bin/python - <<'EOF'
from pathlib import Path
import yaml
root = Path.cwd()
Path(".cache/data/pothole600-pothrgbd-seg.yaml").write_text(yaml.safe_dump({
    "path": str(root),
    "train": ["artifacts/yolo26n-seg/dataset/images/train",
              ".cache/data/pothrgbd-yoloseg/images/train"],
    "val": "artifacts/yolo26n-seg/dataset/images/val",
    "test": "artifacts/yolo26n-seg/dataset/images/test",
    "names": {0: "pothole"},
}, sort_keys=False), encoding="utf-8")
EOF
```

Train (1.008 ảnh / 1.092 instances):

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/yolo segment train \
  data=.cache/data/pothole600-pothrgbd-seg.yaml \
  model=yolo26n-seg.pt imgsz=512 epochs=150 batch=16 device=0 \
  seed=42 deterministic=True patience=0 cache=True \
  cos_lr=True copy_paste=0.3 \
  project=artifacts/yolo26n-seg-merged name=train exist_ok=True
```

Đánh giá trên Pothole-600 official test split để so với baseline:

```bash
YOLO_CONFIG_DIR="$PWD/.cache/ultralytics" .venv/bin/yolo segment val \
  data=.cache/data/pothole600-pothrgbd-seg.yaml split=test device=cpu \
  model=artifacts/yolo26n-seg-merged/train/weights/best.pt
```

Checkpoint này là model production hiện tại. Export bằng đúng recipe ở mục
[Export ONNX](#export-onnx) rồi chép vào `models/pothole_yolo26n_seg.onnx` và
`artifacts/final/`.

Một fine-tune giai đoạn 2 trên riêng Pothole-600 (lr thấp, `copy_paste=0.0`)
nâng box mAP@0.5 lên 90.2% nhưng hạ mAP@0.5:0.95 xuống 49.3%/47.6% và bỏ sót
một ổ gà trên stereo benchmark, nên không được dùng.

## Depth Anything V2 ONNX

Đặt model dynamic tại `.cache/models/depth_anything_v2_vits_dynamic.onnx`.
Tải bản ONNX ViT-S dynamic:

```bash
mkdir -p .cache/models
curl -L \
  https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx \
  -o .cache/models/depth_anything_v2_vits_dynamic.onnx
```

Chạy relative depth và benchmark CPU:

```bash
.venv/bin/python -m pipelines.depth_inference \
  --model .cache/models/depth_anything_v2_vits_dynamic.onnx \
  --image ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET/images/IMAGE.jpg" \
  --size 224 --output artifacts/depth
```

Input 224px là mặc định cân bằng tốc độ; depth map float được resize về kích
thước ảnh gốc và vẫn là **relative depth**, chưa phải mét.

Benchmark ONNX Runtime CPU trên i5-13400F:

| Input | Depth inference | FPS |
|---:|---:|---:|
| 196 | 53,5 ms | 18,7 |
| 224 | 65,0 ms | 15,4 |
| 252 | 78,3 ms | 12,8 |
| 280 | 107,5 ms | 9,3 |
| 518 static | 335,4 ms | 3,0 |

Khi tính cả preprocessing và resize output, CLI 224px đạt khoảng 14,7 FPS.
Pipeline cuối cần chạy depth bất đồng bộ hoặc tái sử dụng depth map giữa các
frame.

## Ghép detection, depth và area

Chạy pipeline trên một ảnh:

```bash
.venv/bin/python -m pipelines.pothole_pipeline \
  --detector models/pothole_yolo26n_seg.onnx \
  --depth-model .cache/models/depth_anything_v2_vits_dynamic.onnx \
  --image path/to/image.jpg \
  --depth-size 196 --warmup 1 --output artifacts/pipeline
```

Pipeline fit mặt phẳng depth cục bộ từ vành đai quanh mỗi segmentation mask,
sau đó báo `relative_depth`, `relative_area` và severity heuristic. Các trường
`metric_calibrated` và `severity_calibrated` giữ `false` cho đến khi có camera
intrinsics/depth scale đáng tin cậy.

## Benchmark relative depth

Benchmark Depth Anything bằng GT mask và raw RealSense depth của PothRGBD:

```bash
.venv/bin/python -m benchmarks.benchmark_depth_pothrgbd \
  --dataset ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET" \
  --model .cache/models/depth_anything_v2_vits_dynamic.onnx \
  --size 196 --threads 6 --output artifacts/depth-benchmark
```

Script fit đúng một scale trên calibration split, tự chọn hướng dấu depth bằng
calibration data, sau đó khóa cả hai để tính relative error trên test split.
Đơn vị vẫn được ghi là raw sensor units vì archive không kèm `depth_scale`.

Kết quả trên 967 ảnh hợp lệ cho thấy ground-plane normalization vẫn có median
relative error 56,6% và chỉ 12,4% test instances nằm trong ±15%. Vì vậy
Depth Anything V2 Small relative 196px hiện **không đạt** KPI depth accuracy;
pipeline giữ model này làm baseline tốc độ, không coi là phương án depth cuối.

## ROI depth regression

Train MobileNetV3-Small từ RGB crop, segmentation mask và RealSense depth của
PothRGBD:

```bash
.venv/bin/python -m training.train_depth_regressor \
  --dataset ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET" \
  --output artifacts/depth-regressor-context
```

Chạy pipeline YOLO segmentation + ROI depth ONNX:

```bash
.venv/bin/python -m pipelines.pothole_pipeline \
  --detector artifacts/final/pothole_yolo26n_seg.onnx \
  --depth-model artifacts/depth-regressor-context/pothole_depth_regressor.onnx \
  --image path/to/image.jpg \
  --output artifacts/pipeline-roi
```

Benchmark end-to-end bằng predicted mask trên test split PothRGBD:

```bash
.venv/bin/python -m benchmarks.benchmark_roi_pipeline \
  --dataset ".cache/data/pothrgbd/PUBLIC POTHOLE DATASET" \
  --detector artifacts/final/pothole_yolo26n_seg.onnx \
  --depth-model artifacts/depth-regressor-context/pothole_depth_regressor.onnx
```

Benchmark hiện tại đạt 25,5 FPS nhưng chỉ match 57,3% GT instances trên
PothRGBD. Median depth error trên matched instances là 24,4% và median
relative-area error là 19,5%; cấu hình này đạt KPI tốc độ nhưng chưa đạt A2.

## Stereo depth và area metric

Pipeline cuối dùng YOLO mask để định vị và StereoSGBM để đo chênh lệch so với
mặt đường. Cấu hình CPU mặc định dùng stereo scale `0.3125` (200 px trên bộ
Fan), 112 disparities và 4 OpenCV threads:

```bash
.venv/bin/python -m benchmarks.benchmark_stereo_yolo \
  --dataset .cache/data/fan-stereo-pothole \
  --detector models/pothole_yolo26n_seg.onnx
```

Benchmark 27 stereo pair, trong đó `model1` dùng để calibration và 19 pair của
`model2/model3` được giữ làm held-out:

| KPI held-out | Kết quả |
|---|---:|
| Detection / fusion coverage | 100% / 100% |
| Median depth error | 4,01% |
| Depth trong ±8% / ±15% | 100% / 100% |
| Median area error | 11,23% |
| Area trong ±8% / ±15% | 26,32% / 84,21% |
| Area p95 / max | 18,3% / 18,7% |
| Depth + area cùng trong ±15% | 84,21% |
| Median end-to-end FPS CPU | 17,8 |

Mặt đường được fit hai lượt: lượt một trên toàn bộ pixel valid, lượt hai fit
lại sau khi loại pixel thuộc residual mask. Nếu không loại, mặt phẳng bị võng
vào lòng hố và làm hụt cả depth lẫn area. Diện tích chỉ hiệu chuẩn đường
residual fallback (`--area-scale`, fit trên `model1`); đường mask YOLO để hệ
số 1,0 — hệ số 0,91 từng fit trên `model1` nén median nhưng kéo dài đuôi lỗi
(p95 18,3% → 19,1%) vì `model3` cần hệ số ngược hướng, nên bị bỏ.

Thử chạy detector và stereo đồng thời làm mức dùng CPU tăng từ khoảng 72% lên
91%, nhưng throughput giảm vì tranh chấp core — kể cả khi ghim ORT 6–10
intra-op threads, mọi stage đều chậm gấp đôi (thread bị đẩy sang E-core). Vì
vậy runtime giữ luồng sequential, không dùng queue và giới hạn OpenCV ở 4
threads.

FPS đo 2026-07-26 trên máy rảnh sau hai tối ưu (fit mặt đường rewrite
4,4 → 2,1 ms/lượt; resize SGBM chuyển `INTER_LINEAR`, −2,6 ms và depth median
còn tốt hơn): 56,2 ms / 17,8 FPS, 100% pair ≥15 FPS. Số này đo trước phần
`INTER_LINEAR` nên còn dư địa ~2 ms — cần một lần đo chốt trên máy rảnh.
Phân tích sàn sai số và các hướng tối ưu đã thử và loại nằm ở
[BENCHMARK.md](BENCHMARK.md).
