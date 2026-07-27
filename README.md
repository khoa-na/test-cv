# Pothole depth/area estimation + GPS-degraded localization

> **📄 Báo cáo kỹ thuật:** [`REPORT.pdf`](REPORT.pdf) để nộp và
> [`REPORT.md`](REPORT.md) để review source — quyết định thiết kế, bảng KPI,
> failure analysis và các thí nghiệm phản chứng. Đọc report trước nếu bạn đang
> chấm bài; README này là hướng dẫn tái lập.

Hai phần dùng chung một camera stereo và một ngân sách CPU:

- **Phần A** — phát hiện ổ gà, đo độ sâu và diện tích. Pipeline CPU/ONNX dùng
  `YOLO26n-seg` để định vị và StereoSGBM để đo hình học so với mặt đường; xem
  [Stereo depth và area metric](#stereo-depth-và-area-metric).
- **Phần B** — giữ pose liên tục khi GPS suy giảm, bằng stereo VO, IMU,
  landmark database và EKF hai frame.

Tóm tắt nghiệm thu: A1 đạt in-domain (86,2% box mAP@0.5; **49,9%** trên bộ
độc lập). A2/A3 đạt headline median trên proxy, nhưng area chỉ đạt ngưỡng ở
17/19 mẫu và tốc độ ở 26/27 pair. Phần B đạt B5/B6/B7 trong phạm vi replay;
B1 đạt 12/13 cửa sổ, B3 chỉ đạt garage proxy, còn B2/B4/B8 chưa đạt. Tất cả
failure và phạm vi claim được phân tích trong `REPORT.md`.

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
| Box mAP@0.5 | 86.2% |
| Box mAP@0.5:0.95 | 53.4% |
| Mask mAP@0.5 | 84.6% |
| Mask mAP@0.5:0.95 | 50.8% |
| Preprocess + inference + postprocess | 29.4 ms |
| Model pipeline FPS | 34.0 |

Số trên do `benchmarks/benchmark_a1_receipt.py` sinh; receipt đầy đủ kèm model
SHA, split và phiên bản thư viện ở `artifacts/verify-final/a1/benchmark.json`.
Bản nháp trước của tài liệu ghi 89.8% / 87.1% lấy từ dòng "ONNX merged" trong
`BENCHMARK.md`; con số đó không tái lập được bằng đường nào còn chạy được, xem
mục detector trong `REPORT.md`.

Model đạt KPI detection `mAP@0.5 >= 80%` và tốc độ riêng model `>= 15 FPS`.
Weights train trên tập gộp Pothole-600 + PothRGBD; val chạy trên Pothole-600
official test split nên so trực tiếp được với baseline chỉ-Pothole-600
(85.5% box / 81.2% mask), bản cũ giữ tại `artifacts/final-baseline-pothole600/`.

## Cài đặt

Cần **Python ≥ 3.10** (code dùng cú pháp `X | None`). Phát triển và đo trên
3.14.4, Linux x86-64.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` mặc định cài PyTorch CPU. Chỉ khi train trên NVIDIA CUDA
13.0 mới dùng:

```bash
.venv/bin/pip install -r requirements-train-cuda.txt
```

Mọi số deployment trong báo cáo đo bằng ONNX Runtime CPU.

`REPORT.pdf` render từ `REPORT.md` bằng:

```bash
.venv/bin/pip install -r requirements-report.txt
.venv/bin/python tools/render_report.py
```

WeasyPrint cần thư viện hệ thống Pango/Cairo (trên Debian/Ubuntu:
`libpango-1.0-0 libpangocairo-1.0-0`).

## Chạy test

```bash
.venv/bin/python -m pytest -q -s
```

117 test, không cần dataset và không cần GPU. Flag `-s` tránh lỗi capture
temporary-file của pytest đã quan sát trên WSL/Python 3.14; nó không thay đổi
test logic.

## Dữ liệu

Không dataset nào nằm trong repo. Tất cả tải về `.cache/data/`, đã gitignore.
Chỉ cần bộ tương ứng với phần muốn tái lập — không cần tải hết 24 GB.

| Bộ | Dùng cho | Dung lượng | Giấy phép | Nguồn |
|---|---|---:|---|---|
| Pothole-600 | Train và test detector (A1) | 218 MB | nghiên cứu | [sites.google.com/view/pothole-600](https://sites.google.com/view/pothole-600/dataset) |
| PothRGBD | Bổ sung mask, train depth regressor | 668 MB | công khai | `PUBLIC POTHOLE DATASET` (RGB-D) |
| Fan stereo pothole | Depth và area so ground truth (A2/A3) | 878 MB | **MIT** | [github.com/ruirangerfan/stereo_pothole_datasets](https://github.com/ruirangerfan/stereo_pothole_datasets) |
| Mendeley pothole video | Test cross-domain (49,9% mAP) | 1,9 GB | **CC BY 4.0** | [data.mendeley.com/datasets/5bwfg4v4cd](https://data.mendeley.com/datasets/5bwfg4v4cd/2) |
| 4Seasons | Toàn bộ Phần B | 19 GB | **CC BY-NC-SA 4.0** | [cvg.cit.tum.de/data/datasets/4seasons-dataset](https://cvg.cit.tum.de/data/datasets/4seasons-dataset) |

4Seasons bắt đăng ký trước khi tải vì GDPR, và phi thương mại. Lệnh tải bốn
recording cụ thể nằm ở [mục Phần B](#tải-dataset).

Bộ Mendeley tải trực tiếp không cần đăng nhập:

```bash
mkdir -p .cache/data/pothole-video-id && cd .cache/data/pothole-video-id
URL=$(curl -s "https://data.mendeley.com/public-api/datasets/5bwfg4v4cd/files?folder_id=root&version=2" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['content_details']['download_url'])")
curl -L "$URL" -o pothole_video.zip && unzip -q pothole_video.zip
```

## Video demo

Ba video 203 MB, quá nặng cho Git nên host riêng:

[Hugging Face — pothole-gps-localization-demos](https://huggingface.co/datasets/khoa-na/pothole-gps-localization-demos)

Cả ba render từ dataset công khai và gắn banner nguồn trên khung hình. **Không
video nào là footage tự quay tại TP.HCM.** Script render nằm ở `demo/`, chạy
lại được từ dataset gốc.

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

- Model có một class `pothole`; pipeline stereo suy ra
  `minor/moderate/severe` từ metric depth và area: `severe` nếu depth ≥50 mm
  hoặc area ≥100.000 mm²; `moderate` nếu depth ≥25 mm hoặc area ≥40.000 mm²;
  còn lại là `minor`. Đây là heuristic triage, nên output giữ
  `severity_calibrated=false` cho tới khi có field calibration.
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

---

# Phần B — Localization khi GPS suy giảm

Toàn bộ Phần B chạy trên 4Seasons replay: stereo VO, IMU yaw, NMEA thật, EKF
hai frame và landmark database. Kết quả và phân tích nằm ở
[REPORT.md](REPORT.md); mục này chỉ là hướng dẫn tái lập.

## Tải dataset

Bốn recording từ [4Seasons](https://www.4seasons-dataset.com/), giải nén vào
`.cache/data/4seasons/`:

```bash
mkdir -p .cache/data/4seasons && cd .cache/data/4seasons
BASE=https://vision.cs.tum.edu/webshare/g/4seasons-dataset
for rec in recording_2020-03-24_17-36-22 recording_2020-12-22_11-54-24 \
           recording_2021-02-25_13-39-06 recording_2021-05-10_19-15-19; do
  for part in imu_gnss reference_poses stereo_images_undistorted; do
    curl -sSL --retry 3 -C - -O "$BASE/dataset/$rec/${rec}_${part}.zip"
  done
done
curl -sSL -O "$BASE/calibration/calibration.zip"
for z in *.zip; do unzip -q -o "$z"; done
```

Tên gọi trong report: `office_loop_1`, `neighborhood_4`, `garage_2`,
`garage_3` — theo đúng thứ tự trên. Reference trajectory chỉ dùng để chấm
offline, không bao giờ đi vào VO, EKF hay landmark association.

## Chạy benchmark

Tổng khoảng 34 phút tuần tự trên i5-13400F. B7 phải chạy riêng lúc máy rảnh:
script tự ghi load average và cảnh báo nếu vượt 1,0.

```bash
# B1 — VO drift trên cửa sổ 500 m (4 sequence, ~11 phút)
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_vo_drift.py \
  --no-render --output artifacts/vo-drift-final

# B2 — Landmark re-identification, cả hai chiều (~6,5 phút)
.venv/bin/python benchmarks/benchmark_landmark_reid.py --reverse

# B3 — U-turn detection latency (~3 phút)
.venv/bin/python benchmarks/benchmark_uturn.py

# B5 + B8 — Localization trong hầm, ablation A/B/C và re-lock (~8 phút)
.venv/bin/python benchmarks/benchmark_garage_localization.py

# B6 + B8 — GPS handover và re-lock trên NMEA thật (~2 phút)
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_gps_fusion.py

# B7 — End-to-end throughput, chạy khi máy rảnh (~3 phút)
.venv/bin/python benchmarks/benchmark_system_fps.py
```

Hai cờ chẩn đoán, **không thuộc cấu hình nộp bài**, dùng để dựng các phản chứng
trong report:

```bash
# Ép mở cổng hiệu chỉnh gyro tĩnh
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_vo_drift.py \
  --sequence garage_3 --gyro-gate-override 0.10 --no-render \
  --output artifacts/vo-drift-gyro-counterfactual

# Quét ngưỡng consensus của recovery policy
.venv/bin/python benchmarks/benchmark_garage_localization.py \
  --consensus-sweep --output artifacts/garage-localization-sweep
```

## Audit dữ liệu

```bash
# Có cửa sổ đứng yên nào để hiệu chỉnh bias gyro không
PYTHONPATH=. .venv/bin/python data_tools/audit_gyro_calibration.py

# Chất lượng frame nhãn dùng để chấm B2
.venv/bin/python -m data_tools.audit_label_frame
```

## Lớp ROS 2

`FusionBridge` là Python thuần, không import `rclpy`; lớp node chỉ bóc message
rồi gọi bridge. GPS nhận raw NMEA GGA qua `/gps/gga` (`std_msgs/String`) để
giữ đúng `fix_quality`, số vệ tinh và HDOP; `sensor_msgs/NavSatFix` một mình
không chứa đủ metadata cho integrity gate. Nhờ vậy bridge kiểm chứng được mà
không cần cài ROS:

```bash
.venv/bin/python ros2/localization_node.py --self-check
```

## Video demo

```bash
.venv/bin/python demo/render_part_a.py        # stereo depth/area so GT
.venv/bin/python demo/render_pothole_video.py # video liên tục, bộ độc lập
.venv/bin/python demo/render_part_b.py        # quỹ đạo + trạng thái GPS
```

Cả ba đều gắn banner nguồn trên khung hình. Không video nào là footage tự quay
tại TP.HCM, và report không trình bày chúng như vậy.

---

## Giấy phép

Mã nguồn trong repo này theo **MIT** — xem [LICENSE](LICENSE).

Dataset không được redistribute ở đây; mỗi bộ giữ giấy phép riêng, liệt kê ở
[mục Dữ liệu](#dữ-liệu) và trong LICENSE. Một số hình bằng chứng dưới
`artifacts/` dẫn xuất từ dataset gốc nên chịu điều khoản của bộ đó chứ không
phải MIT — đáng chú ý là `artifacts/vo-drift-final/office_loop_1.png` vẽ từ
4Seasons (CC BY-NC-SA 4.0, phi thương mại).

Ba video demo phát hành riêng dưới CC BY-NC-SA 4.0 trên
[Hugging Face](https://huggingface.co/datasets/khoa-na/pothole-gps-localization-demos).
