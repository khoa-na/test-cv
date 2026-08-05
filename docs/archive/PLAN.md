# Kế hoạch Phần B — GPS + Visual Fallback

> **Tài liệu lịch sử.** Đây là bản kế hoạch viết *trước* khi chạy benchmark, giữ
> lại để thấy quá trình. Mọi con số và kết luận trong file này đã bị
> [`REPORT.md`](REPORT.md) thay thế — trong đó có một giả thuyết về cổng hiệu
> chỉnh gyro sau này bị thực nghiệm bác bỏ. Đọc `REPORT.md` để lấy kết quả cuối.

Trạng thái: Phần A xong (xem [README](../../README.md), [BENCHMARK.md](../../BENCHMARK.md)).
Phần B là khối việc lớn còn lại. Không còn thời gian tự thu data — dùng dataset open.
Đánh giá paper nền tại [PAPERS.md](PAPERS.md).

## Quyết định chốt

**Hướng: 4Seasons một bộ, stereo VO, GPS simulate từ reference pose.**

- Dataset: [4Seasons (TUM)](https://cvg.cit.tum.de/data/datasets/4seasons-dataset)
  — stereo + IMU + RTK reference pose cm-level, đa traversal cùng route,
  có parking garage 3 tầng (đúng kịch bản B5/S1) và tunnel.
- VO: nhánh stereo trong PAPERS.md (LIBVISO2-style). Stereo giết bài toán
  scale — nhánh rủi ro nhất của mono biến mất.
- GPS: **audit Bước 0 phát hiện dataset kèm `septentrio.nmea` — NMEA thật
  từ receiver** (fix quality, số vệ tinh, HDOP; garage_2 có 526 message
  mất fix + gap receiver câm 517 s). Đường chính cho B6/B8 là replay NMEA
  thật; simulator từ reference pose hạ xuống vai trò kịch bản kiểm soát
  bổ sung + unit test.

> **Ghi chú về tính hợp lệ:** reference pose chỉ được dùng để (1) sinh
> `GPS_sim` trước khi chạy và (2) đánh giá output. Stereo VO và EKF không
> được đọc reference pose trong lúc inference. GPS simulated chứng minh
> software handover/re-lock, không thay thế benchmark receiver GNSS thật.

### Lý do (tóm tắt)

| Tiêu chí | Đánh giá |
|---|---|
| Điểm phủ | B1 (10%) + B2 (12%) + B5 (8%) = 30% trên một bộ; B3/B4/B7 (16%) cùng data; B6/B8 (9%) simulate |
| Thời gian | Một parser, một calibration format, một vòng debug |
| Rủi ro | Không scale ambiguity; kịch bản GPS kiểm soát được |
| Cộng hưởng Phần A | Tái dùng stereo calibration, RANSAC và detector ONNX; chỉ dùng ổ gà trên 4Seasons nếu audit sample xác nhận có true positive lặp giữa các traversal |

### Dataset đã cân nhắc và loại (giữ cho failure analysis)

| Bộ | Lý do loại |
|---|---|
| UrbanNav HK | Real GNSS RINEX + tunnel thật; không làm trong deadline vì thêm format và camera chưa chắc stereo. Chỉ ghi làm future work |
| KITTI odometry | GT pose chuẩn nhưng không tunnel/GPS loss, không multi-traversal |
| comma2k19 | Mono + CAN, highway — không hầm, không lặp route, mất B2/B5 |
| BDD100K | Không GT pose — không chấm định lượng được |

## Các bước

### Bước 0 — Verify dataset — **ĐÃ XONG 2026-07-26**

Kết quả: loader `data_tools/fourseasons.py` + `tests/test_fourseasons.py`
(5 pass). Data tại `.cache/data/4seasons/`, 4 sequence:

| Sequence | Frames | Vai trò |
|---|---|---|
| parking_garage_2_train | 5.488 | B2/B5/B6 traversal 1, GPS loss thật |
| parking_garage_3_train | 5.253 | B2 traversal 2 (khác mùa/ánh sáng) |
| neighborhood_4_train | 9.771 | B1 (2.2 km) |
| office_loop_1_train | 15.177 | B1 bổ sung + B4 + landmark biển báo |

Audit: ≥500 m ✅ (2.2 km + garage 852/788 m); GPS loss thật ✅; 2 traversal
garage ✅; U-turn phố ❌ (chỉ ramp garage); vạch tim đường ❌ (phố Đức không
sơn — B4 chuyển sang road-half qua biên đường); detector chạy trên
monochrome ✅ nhưng conf 0,74 trên garage nghi FP — pothole-landmark nghiêng
Fan adapter. Test set 4Seasons không có reference poses nên toàn bộ dùng
train sequences. Domain `4seasons-dataset.com` đã bị chiếm (spam casino) —
chỉ dùng `cvg.cit.tum.de`. License CC BY-NC-SA 4.0, citation vào báo cáo.

Checklist gốc (đã chạy):

- Kiểm license/form đăng ký 4Seasons, cách tải.
- Tải 1 sequence **parking garage** + 1 sequence **urban/neighborhood**
  (2 traversal cùng route cho B2). Không tải cả 350 km.
- Đọc format: calibration stereo, reference pose, timestamp, IMU.
- **Kill switch 90 phút:** nếu chưa tải và parse được sample thì chuyển ngay
  sang KITTI Odometry; không thử thêm dataset thứ ba.
- Audit 100–200 frame sample:
  - Có đủ đoạn ≥ 500 m để chấm B1 không?
  - Có traversal reference/query thực sự overlap cho B2 không?
  - Có U-turn thật (heading đổi ≥ 150°) không?
  - Có lane marking đủ rõ để gán nhãn B4 không?
  - Detector Phần A có true-positive pothole không, và cùng pothole có xuất
    hiện trong hai traversal không? **Lưu ý domain gap: detector train trên
    RGB, 4Seasons là monochrome** — nếu detector rớt trên grayscale, chuyển
    hẳn nhánh pothole-landmark sang Fan adapter ngay tại đây, không chờ
    Bước 3.
- Kiểm tra ROS 2 ngay: `ros2`, `rclpy`, `nav_msgs`, `geometry_msgs`,
  `tf2_ros`. Nếu thiếu, core vẫn chạy nhưng ROS wrapper phải được ghi là
  chưa runtime-verified.
- Deliverable: `data_tools/` loader + manifest sequence + smoke test đọc được
  stereo frame, calibration, timestamp, reference pose và IMU.

> **Lý do sửa:** dataset access, nội dung sequence và ROS là các blocker có
> thể phát hiện trong giờ đầu. Không để đến cuối mới biết sequence không có
> lane/U-turn hoặc môi trường không chạy được ROS 2.

### Bước 1 — GPS (NMEA thật + simulate) + EKF + state machine bằng odometry proxy

- **Đường chính: replay `septentrio.nmea`** — parser GGA đã có trong
  `data_tools/fourseasons.py`. Fix quality/HDOP/số vệ tinh thật; garage cho
  degrade → lost → re-lock thật.
- Bổ sung `GPS_sim` offline từ reference pose (kịch bản kiểm soát + unit test):
  - `good`: 1–5 Hz, Gaussian noise + bias nhỏ.
  - `degraded`: noise/bias lớn dần, HDOP tăng, số vệ tinh giảm và có outlier.
  - `lost`: không phát measurement.
  - `recovering`: 3–5 fix liên tiếp trước khi cho phép trở lại `good`.
- Tạo `ODOM_proxy` từ relative reference motion rồi chèn noise/drift có seed
  cố định. Đây chỉ là test double để hoàn thiện fusion trước khi stereo VO có.
- EKF local `(x, y, θ, v, yaw_rate)` liên tục từ odometry; global correction
  từ GPS/landmark. Giữ `odom → base_link` liên tục, correction đi qua
  `map → odom`.
- Gate GPS: quality (HDOP ≤ 5, vệ tinh ≥ 4) **và** NIS χ² ≤ 5.991.
  Re-lock dùng `R_gps` lớn rồi giảm dần trong 0,5–2 s.
- Với 4Seasons, state machine latch tọa độ local ENU tin cậy cuối cùng.
  Phần lý thuyết giải thích WGS84 được đổi sang ENU tại datum; không tuyên bố
  dataset cung cấp WGS84/HDOP/SNR nếu file thực tế không có.
- Benchmark ngay B6/B8 (và B3 trên synthetic) bằng GT giữ riêng:
  - Handover latency.
  - Maximum local pose jump.
  - Drift trong dropout.
  - Re-lock error và thời gian hội tụ.
- Deliverable: `pipelines/localization_ekf.py`,
  `data_tools/simulate_gps.py`, unit tests cho transition/NIS/re-lock và
  benchmark fusion với proxy odometry.

> **Lý do sửa:** EKF/state machine là phần dễ kiểm soát và ăn điểm B3/B6/B8.
> Làm trước bảo đảm có pipeline end-to-end ngay cả khi stereo VO mất nhiều
> thời gian hoặc chưa đạt drift KPI.

### Bước 2 — Stereo VO (B1, 10%)

**ĐÃ IMPLEMENT 2026-07-26:** stereo ORB + symmetric PnP, optional IMU-yaw có
low-angular-rate bias gate, benchmark relative-pose 500 m và fusion
`--odom vo`. Kết quả cuối là 12/13 cửa sổ trên `artifacts/vo-drift-final/`; con
số 11/12 ở bản kế hoạch này lấy từ một lần chạy chưa đủ sequence.

- ORB detect/match trái-phải tại frame `t` → epipolar/disparity gate →
  triangulate → temporal match sang left `t+1` → `solvePnPRansac` + refine.
- Reject feature động/outlier bằng RANSAC; yêu cầu feature phân bố qua nhiều
  ô lưới ảnh, không chỉ tập trung một vùng.
- Confidence từ inlier count/ratio, reprojection RMS và grid coverage.
  Forward-backward consistency chạy ở keyframe rate nếu làm mỗi frame khiến
  FPS giảm; confidence scale `R_vo` hoặc reject update.
- Thay `ODOM_proxy` trong Bước 1 bằng stereo VO output; không đổi API EKF.
- Protocol B1:
  - Chia trajectory theo reference path length thành cửa sổ 500 m không
    overlap; join VO/reference bằng timestamp, không zip theo index.
  - Trong từng cửa sổ, so relative pose start→end:
    `E = inv(ΔT_ref) @ ΔT_vo`.
  - `drift = norm(translation(E)) / reference_path_length`; không
    scale-align hoặc rigid-fit bằng toàn cửa sổ.
  - Báo median, p95 và tỷ lệ cửa sổ ≤ 5%; ATE/RPE là metric phụ.
- Deliverable: `pipelines/stereo_vo.py`, unit tests geometry và
  `benchmarks/benchmark_vo_drift.py`.

> **Lý do sửa:** metric B1 phải ngăn scale alignment che lỗi calibration.
> Forward-backward check có ích nhưng không được phép làm B7 trượt chỉ vì
> chạy gấp đôi pose estimation trên mọi frame.

### Bước 3 — Landmark DB + correction (B2 12%, B5 8%)

- DB schema:
  `{id, class, p_3D_or_ENU, descriptor, t_first, t_last, n_obs, covariance}`.
- **MVP bắt buộc:** visual keyframe/landmark match → lấy ENU coordinate đã
  biết của reference → global EKF position update với covariance.
- Association gate:
  spatial prior → class/geometry nếu có → descriptor top-K → sequence
  consistency 5–10 keyframe → ORB/RANSAC geometric verification.
- Ground truth B2:
  - Query và database thuộc hai traversal khác nhau.
  - Positive nếu reference pose nằm trong localization radius cố định.
  - Báo Recall@1, Recall@5, false-positive rate và coverage.
- Pothole:
  - Chỉ dùng detection thật nếu audit Bước 0 xác nhận.
  - Nếu 4Seasons không có pothole lặp, benchmark pothole adapter riêng trên
    Fan stereo (3 physical IDs) và ghi rõ không phải end-to-end 4Seasons.
  - Không tạo pothole giả rồi tính Recall.
- **Bonus sau khi MVP đạt:** ghost projection semantic bbox/bearing residual
  theo Qu IV 2015 và Jacobian EKF. Không để phần này chặn B2/B5.
- B5: chạy pipeline trên garage với GPS cắt; báo drift, localization
  coverage và recovery khi ra. Chỉ gọi đây là “parking-garage GPS-denied”;
  không claim `<10 lux`/IR nếu dataset không có lux metadata.
- Deliverable: `pipelines/landmark_db.py`, unit tests association và
  `benchmarks/benchmark_landmark_reid.py`.

> **Lý do sửa:** kế hoạch cũ gộp database, semantic detection, sequence VPR,
> ghost projection và EKF Jacobian vào một bước. Position update từ matched
> reference đã đủ làm MVP; ghost projection là bonus của đề, không bắt buộc.

### Bước 4 — Lane + U-turn (B3 5%, B4 6%)

- **Audit xác nhận không có vạch tim đường** (phố Đức nhỏ): định nghĩa thực
  thi là **road-half** — xe ở nửa nào của lòng đường, xác định bằng IPM +
  biên đường (lề/vỉa hè), intensity/gradient, không dựa màu. Vạch
  dừng/mũi tên ở ngã tư office_loop dùng làm sanity check cục bộ. Ghi rõ
  định nghĩa vào báo cáo; `city_loop`/`highway` là phương án nếu B4 fail.
- Có trạng thái `unknown`; tự gán nhãn tối thiểu 100–200 frame.
- Báo đồng thời:
  - Accuracy trên toàn bộ eligible frame (`unknown` không tự động bị loại).
  - Coverage.
  - Accuracy khi confident.
  - Unknown rate.
- U-turn: tích lũy heading unwrapped, phát hiện `|Δθ| ≥ 150°` trong cửa sổ
  trượt. Chỉ claim KPI thực nếu audit tìm thấy U-turn thật; nếu không, cung
  cấp unit test + synthetic trajectory và ghi là validation mô phỏng.
  **Số B3 nộp báo cáo:** từ bước này nếu có U-turn thật; nếu không, dùng số
  synthetic của Bước 1 kèm nhãn "validation mô phỏng" — không trộn hai nguồn.
- UFLD ResNet-18 ONNX là bonus chỉ khi baseline không đạt và có thể
  export/benchmark trong tối đa 2 giờ.
- Deliverable: `pipelines/lane_position.py`,
  `pipelines/motion_events.py`, annotation nhỏ và benchmark B3/B4.

> **Lý do sửa:** garage không bảo đảm có lane chuẩn hoặc U-turn. Chỉ báo
> accuracy trên frame tự chọn/confident sẽ làm metric đẹp giả tạo; coverage
> và `unknown` phải được công khai.

### Bước 5 — Tích hợp + ROS 2 + FPS (B7 5%)

- Ghép core theo API thống nhất; ưu tiên sequential để giảm rủi ro tranh core,
  nhưng không coi kết quả concurrency của Phần A là bằng chứng mọi pipeline
  đều phải sequential.
- Multi-rate schedule:
  - VO: mỗi frame.
  - EKF: mỗi measurement.
  - Lane: 5–10 Hz.
  - Pothole detector: 2–5 Hz.
  - Descriptor/re-id: keyframe hoặc khi gần landmark dự đoán.
- ROS 2: node `rclpy` mỏng publish `nav_msgs/Odometry`; core pipeline giữ
  ngoài ROS để test được. Publish thêm pose/state diagnostics nếu message
  dependency đã có.
- Benchmark B7 phải bật các module được tuyên bố trong demo, kể cả detector
  Phần A ở tần số thưa. Báo FPS video, latency từng stage và update rate thật
  của từng module.
- Deliverable: node + `benchmarks/benchmark_partb_fps.py`.

### Bước 6 — Báo cáo + demo

- Báo cáo: kiến trúc, benchmark từng KPI, failure analysis (dataset loại,
  paper loại, hướng thử-và-bỏ — cùng format BENCHMARK.md).
- Video demo ≥ 60 s: overlay quỹ đạo + trạng thái GPS + landmark ghost
  trên sequence garage.
- Nêu rõ giới hạn: GPS simulated, không phải NMEA thật; UrbanNav là bước
  future work, không phải phần triển khai hiện tại.
- Nêu reference pose 4Seasons được tạo từ stereo VIO + RTK-GNSS nên không
  hoàn toàn độc lập với visual pipeline, và độ chính xác global trong vùng
  GNSS xấu có giới hạn theo công bố dataset.

## Thứ tự và phụ thuộc

```text
0 ──> 1 (proxy odom + EKF)
      ├──> 2 (stereo VO thay proxy) ──> 3 ──> 5 ──> 6
      └──> 4 ──────────────────────────┘
```

Bước 4 dùng heading từ Bước 1 nên có thể làm song song với VO/landmark.

### Lịch 2 ngày và cut line

| Thời gian | Mốc bắt buộc |
|---|---|
| Ngày 1, 0–2 h | Dataset/ROS audit; fallback KITTI nếu quá 90 phút |
| Ngày 1, 2–6 h | GPS simulator + EKF + state machine + unit tests |
| Ngày 1, 6–12 h | Stereo VO + drift benchmark |
| Ngày 2, 0–5 h | Landmark MVP + re-id/correction benchmark |
| Ngày 2, 5–8 h | Lane/U-turn baseline |
| Ngày 2, 8–12 h | Integration, ROS wrapper, FPS, report và video |

Khi cháy deadline, cắt theo thứ tự: UrbanNav (đã loại) → UFLD → ghost
projection/Jacobian → forward-backward mỗi frame → tightly-coupled IMU.
Không cắt GPS simulator, EKF/state machine, NIS gate, một dạng landmark
correction, FPS benchmark và failure analysis.

## Rủi ro chính

| Rủi ro | Ứng phó |
|---|---|
| 4Seasons cần đăng ký/duyệt chậm | Kill switch 90 phút sang KITTI; không chuyển UrbanNav |
| Sequence không có pothole/lane/U-turn | Dùng Fan cho pothole adapter; outdoor sequence cho lane; synthetic U-turn chỉ claim validation mô phỏng |
| Detector Phần A (train RGB) rớt trên ảnh monochrome 4Seasons | Phát hiện ở audit Bước 0; chuyển pothole-landmark sang Fan adapter, landmark 4Seasons dùng keyframe/ORB thuần |
| VO drift > 5% trên garage (ít texture, ánh sáng xấu) | Dùng gyro IMU cho heading nếu timestamp/extrinsics đã verify; vẫn báo số đo thật và failure analysis, không “giảm KPI claim” để né kết quả |
| Landmark MVP không đạt Recall 85% | Spatial gate + sequence matching; báo Recall@1/@5, FPR và coverage; không dùng GT pose để chọn đúng match |
| ~~Nghiệm thu bắt GPS thật~~ — đã hoá giải: NMEA receiver thật có sẵn trong 4Seasons | B6/B8 chạy trên NMEA thật; simulator chỉ còn là kịch bản kiểm soát bổ sung |
| FPS < 15 khi ghép đủ module | Landmark/lane chạy keyframe rate (2–5 Hz), chỉ VO + EKF mỗi frame |
| ROS 2 không có trong môi trường | Phát hiện ở Bước 0; giữ core độc lập, cung cấp wrapper/config và ghi rõ mức runtime verification |
