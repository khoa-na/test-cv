# STEP 2 — Stereo Visual Odometry (B1, 10%)

Mục tiêu: thay `create_odometry_proxy` bằng stereo VO thật, **giữ nguyên API
`OdometryMeasurement`** (dt/dx/dy/dtheta + std), đo B1 drift ≤ 5% / cửa sổ
500 m, rồi chạy lại benchmark fusion (vòng 4) để lấy số B8 NMEA thật đầu tiên
không bị proxy drift trói.

## 0. Dữ kiện đã xác minh

- Ảnh `undistorted_images/cam{0,1}`, 800×400 mono, ~30 Hz (dt ≈ 33 ms).
- Stereo **đã rectify hoàn toàn**: `T_cam1_cam0` rotation = 0°, translation
  `[-0.3005, 0, 0]` → epipolar line = cùng hàng ảnh. Depth = `fx·b/disparity`.
- Intrinsics từ `undistorted_calib_{0,1}.txt` (loader đã có,
  `data_tools/fourseasons.py:load_calibration`).
- garage_2 = 5488 frame; office_loop_1 = 15177 frame (dài nhất, nhiều cửa sổ
  500 m nhất → sequence chính cho B1).

## 1. Pipeline (`pipelines/stereo_vo.py`)

Một class, một method chính:

```
class StereoVO:
    def __init__(self, calib, config): ...
    def process(self, img_left, img_right, timestamp) -> OdometryMeasurement | None
```

`None` chỉ có nghĩa là **không có VO update**, không có nghĩa bỏ qua frame.
Caller vẫn phải chuyển `timestamp/dt` vào fusion qua đường `predict_only` mô
tả ở mục 3 để EKF, integrity timeout và global correction cùng tiến thời gian.

Mỗi frame:

1. **Detect** ORB trên left (grid 4×2 ô, bucket max N/ô — ép phân bố đều,
   yêu cầu PLAN dòng 132-133).
2. **Stereo match** left↔right: BFMatcher Hamming KNN + Lowe ratio, kiểm tra
   mutual left↔right, rồi gate epipolar `|v_left − v_right| ≤ 1 px` và
   disparity ∈ [d_min, d_max]. Tính ngưỡng từ calibration, không hard-code:
   `d_min = fx·b/Z_max`, `d_max = fx·b/Z_min`; calibration hiện tại cho
   khoảng 2.5–150 px với `Z ∈ [1, 60] m`.
3. **Depth**: `Z = fx·b/d`; back-project ra điểm 3D frame `t`.
4. **Temporal match** left `t` → left `t+1`: KNN ratio + mutual check, sau đó
   gate bán kính pixel theo chuyển động frame trước. Frame đầu hoặc frame ngay
   sau dropout dùng bán kính rộng có chặn trên; không để một lần fail khóa
   vĩnh viễn các frame sau.
5. **`solvePnPRansac`** (3D tại `t`, 2D tại `t+1`) + `solvePnPRefineLM` trên
   inlier. OpenCV trả transform của **điểm tĩnh** từ camera `t` sang camera
   `t+1`: `X_t+1 = R_pnp X_t + t_pnp`; đây là nghịch đảo ego-motion. Phải
   invert trước: `R_motion = R_pnp.T`,
   `t_motion = -R_pnp.T @ t_pnp`.
6. **Ego SE(3) → SE(2)** sau khi invert:
   - Trục camera: +Z tiến, +X phải, +Y xuống.
   - `dx = t_motion,z`, `dy = −t_motion,x` (chiếu lên mặt phẳng vehicle,
     bỏ `t_motion,y`).
   - `dtheta` lấy riêng rotation quanh trục Y camera từ `R_motion`; dấu được
     khóa bằng test projection synthetic, không suy đoán từ `decompose R`.
     Bỏ pitch/roll để pitch trên ramp không rò vào heading 2D.
7. **Confidence → std**: từ inlier_count, inlier_ratio, reprojection RMS,
   grid coverage (số ô có inlier). Dùng mapping đơn giản nhưng đủ biến:
   `quality = clip(inlier_ratio, qmin, 1) · clip(coverage/8, cmin, 1)` và
   `std = clip(base · (1 + rms/rms_ref) / sqrt(inlier_count·quality),
   std_min, std_max)`; rotation dùng base/clamp riêng. Hằng số chốt trước khi
   chạy tập chấm, không tune theo từng sequence.
8. **Reject / dropout**: inlier < ngưỡng (ví dụ 15) hoặc RANSAC fail →
   return `None`; caller gọi predict-only, không giả `dx=0` vì measurement
   giả sẽ kéo velocity EKF về 0.

**Forward-backward check + symmetric yaw**: dùng stereo depth tại `t+1` chạy
PnP ngược `t+1 → t`, so composed transform với identity. Lấy trung bình góc
trên manifold giữa yaw thuận và nghịch-đã-invert để giảm bias PnP ở motion
30 Hz rất nhỏ. Bản implement chạy mỗi frame vì benchmark đo còn 44,26 FPS,
vẫn dư B7; nếu phần cứng đích chậm hơn mới hạ xuống keyframe-rate K≈5. Nếu
lệch quá ngưỡng thì inflate std/reject từ frame kiểm tra trở đi, không hồi tố
các update đã fusion.

Không dùng: optical flow pyramid tự viết, bundle adjustment, loop closure.
Gyro-yaw là fallback nhỏ có kiểm soát sau khi full benchmark chứng minh
stereo-only fail do yaw drift: bias chỉ calibrate nếu 2 giây đầu pass
stationary gate (`gyro norm p95 ≤ 0.02 rad/s`); nếu fail thì tự giữ visual
yaw. CLI giữ `--yaw-source visual` để chạy ablation.

## 2. Benchmark B1 (`benchmarks/benchmark_vo_drift.py`)

Protocol chốt trong PLAN (dòng 138-142), nhắc lại vì đây là chỗ dễ tự lừa:

- Chia trajectory theo **reference path length** thành cửa sổ 500 m không
  overlap. Join ảnh/VO/reference theo timestamp và interpolate reference;
  không zip theo index vì số frame ảnh, `times.txt` và `result.txt` có thể
  lệch nhau.
- Mỗi cửa sổ so relative pose, không dùng absolute endpoint đã chứa drift từ
  các cửa sổ trước:
  `ΔT_vo = inv(T_vo_start) @ T_vo_end`,
  `ΔT_ref = inv(T_ref_start) @ T_ref_end`,
  `E = inv(ΔT_ref) @ ΔT_vo`.
- `translation_drift = ‖translation(E)‖ / path_length_ref`; B1 pass khi
  `≤ 5%`. Báo thêm rotation drift (deg/100 m).
- Việc đưa hai relative pose về origin/heading đầu cửa sổ chỉ bỏ gauge
  SE(2), **không fit scale** và không rigid-fit bằng toàn cửa sổ — scale sai
  phải lộ ra.
- Báo thêm cumulative drift từ đầu sequence như metric chẩn đoán, nhưng
  không dùng thay B1 500 m.
- Frame VO dropout được dead-reckon bằng velocity/yaw-rate hợp lệ gần nhất,
  cùng semantics `predict_only` của fusion. Báo riêng raw valid-update scale,
  dropout rate và quãng reference không có update để không che nguyên nhân.
- Reference là `result.txt × GNSS scale`, chỉ dùng trong evaluator, không đi
  vào VO/fusion inference.
- Báo: median, p95, tỷ lệ cửa sổ ≤ 5%, tổng số cửa sổ. ATE/RPE phụ.
- Sequences: office_loop_1 (chính), neighborhood_4, garage_2 (khó nhất:
  low-texture + ramp 3D — kỳ vọng xấu, báo đúng số xấu).
- Kèm: FPS trung bình VO-only (tham chiếu B7), tỷ lệ frame dropout, plot
  trajectory overlay per sequence.
- Artifact: `artifacts/vo-drift/` — thư mục riêng, JSON + PNG script sinh,
  không số phân tích tay.

## 3. Fusion vòng 4 (VO thay proxy)

- `benchmark_gps_fusion.py` thêm nguồn odom `--odom vo` (mặc định giữ
  `proxy` để tái lập vòng 3). VO output đi qua đúng đường
  `OdometryMeasurement` — **không đổi phương trình/motion model EKF, state
  machine hoặc ngưỡng nào** ("không hạ gate sau khi nhìn thấy KPI" vẫn hiệu
  lực). Chỉ bổ sung orchestration API cho frame không có VO update.
- Thêm đường thời gian rõ ràng cho VO dropout:
  `LocalizationFusion.predict_only(timestamp, dt,
  translation_process_std, rotation_process_std)`. Hai tham số `*_process_std`
  là độ bất định chuyển động dùng để inflate process noise trong bước predict,
  **không phải measurement standard deviation** và không tạo measurement
  update. Đường này chỉ predict local EKF bằng velocity/omega hiện tại, tăng
  covariance, chạy global correction, integrity `tick`, recovery timeout và
  ghi pose; tuyệt đối không update odometry bằng `dx=0`. Docstring
  implementation phải giữ rõ phân biệt này.
- `OdometryMeasurement.dy` hiện chỉ làm tăng measurement covariance trong
  `LocalOdometryEKF`, không trực tiếp cập nhật lateral velocity. Đây là giới
  hạn của motion model hiện tại; benchmark VO phải tích phân đầy đủ SE(2),
  còn fusion vòng 4 giữ model này để so sánh công bằng với vòng 3.
- Artifact: `artifacts/gps-fusion-round4-vo/`, vòng 3 giữ nguyên làm baseline.
- Câu hỏi vòng 4 trả lời: episode relock cuối garage_2 (quality 4/5 có mặt)
  có xuống dưới sàn ~7 m không. Nếu vẫn ≥ 5 m, chưa kết luận ngay là
  alignment: chạy ablation cùng VO với simulated GPS và real NMEA, báo riêng
  VO drift/dropout, GPS quality/bias và alignment residual. Chỉ gọi Fix 3 là
  blocker nếu simulated GPS pass, VO đạt/đủ ổn, nhưng real NMEA còn fail và
  error tương quan với alignment residual.

## 4. Tests (`tests/test_stereo_vo.py`)

Ít, đúng chỗ gãy:

1. Depth từ disparity: điểm synthetic biết trước Z, sai số < 1%; kiểm tra
   `d_min/d_max` được suy ra từ calibration.
2. PnP trên scene 3D tĩnh được project trước/sau chuyển động camera thuần
   tiến 1 m: sau khi invert PnP, `dx ≈ +1`, `dy ≈ dtheta ≈ 0`.
3. Pure rotation synthetic bằng projection: `dtheta` đúng dấu + độ lớn,
   translation ≈ 0.
4. SE(3)→SE(2) trên ramp synthetic (pitch 10°): dtheta không nhiễm pitch,
   translation +Z trong vehicle plane vẫn cho đúng dx; thành phần camera Y
   không đi vào state SE(2).
5. Ít feature (ảnh trơn) → `process` trả `None`, không throw.
6. Determinism: cùng 2 frame chạy 2 lần → cùng output (seed RANSAC cố định
   qua `cv2.setRNGSeed`).
7. Temporal recovery: một frame fail rồi frame sau có feature lại không bị
   radius gate khóa vĩnh viễn.
8. Fusion dropout contract: `predict_only` làm timestamp/pose/covariance,
   integrity timeout và correction cùng tiến, nhưng không update velocity
   bằng measurement 0.

## 5. Thứ tự làm + gate dừng

| # | Việc | Gate qua bước sau |
|---|------|-------------------|
| 1 | `StereoVO` + tests synthetic + fusion predict-only | 8/8 test mới pass và toàn bộ test cũ pass |
| 2 | Chạy đoạn office_loop liên tục có ≥100–200 m chuyển động, plot vs reference | đúng dấu forward/yaw; path-length ratio 0.9–1.1; có số inlier/dropout, không gate “bằng mắt” |
| 3 | `benchmark_vo_drift.py` full 3 sequence | số median/p95 ra, bất kể đạt hay không |
| 4 | Fusion vòng 4 `--odom vo` trên garage_2 NMEA thật | so relock vs vòng 3 |
| 5 | Cập nhật STEP1-FIXES (kết luận proxy-drift) + PLAN checkbox | — |

Nếu bước 2 sai hình dạng quỹ đạo → dừng debug geometry (sign convention
camera↔vehicle là nghi phạm số 1), không đi tiếp.

Kết quả gate implementation: office_loop 500,13 m cho stereo-only drift
4,763%, path ratio 0,963, dropout 1,49% và 44,26 FPS. IMU-yaw fallback giảm
drift còn 4,202%, rotation drift còn 0,439°/100 m, FPS 46,66. PnP một chiều
trước symmetric yaw cho drift 9,292%. Artifact tương ứng nằm ở
`artifacts/vo-drift-500m-gate/`, `artifacts/vo-drift-500m-symmetric/` và
`artifacts/vo-drift-500m-imu/`.

Kết quả full B1 (`artifacts/vo-drift-imu/`):

- office_loop: 7 cửa sổ, median 2,886%, p95 5,538%, 6/7 cửa sổ pass; cửa sổ
  fail duy nhất 6,010%.
- neighborhood_4: 4 cửa sổ, median 2,190%, p95 4,343%, 4/4 pass.
- garage_2: 1 cửa sổ, drift 1,759%, pass.
- Tổng 11/12 cửa sổ pass. Không claim “mọi cửa sổ đạt B1”; báo riêng failure
  office 6,010% trong failure analysis.
- `benchmark.json` có aggregate 11/12 và metadata stationary-gate/bias IMU
  cho từng sequence. Các trường này được script `--refresh-existing` bổ sung
  từ raw IMU mà không thay đổi metric VO đã chạy.

Fusion vòng 4:

- Real NMEA garage_2 (`artifacts/gps-fusion-round4-vo/`): B8 vẫn fail,
  max error sau 2 s = 21,557 m và không stable ≤5 m trong 10 s. Episode RTK
  cuối cải thiện 19,090 → 14,804 m; correction jump giảm 0,400 → 0,332 m.
- Simulated GPS (`artifacts/gps-fusion-round4-vo-simulated/`): garage_3
  stable trong 0,012 s; neighborhood stable trong 3,838 s, cả hai
  `all_stable_within_10s=true`. Neighborhood có error-after-2s 8,155 m nên
  guardrail tốc độ fail nhưng stable-B8 pass.
- Kết luận: thay proxy bằng VO không tự sửa real NMEA; alignment/GPS bias là
  blocker còn lại. Fusion recovery vẫn pass khi GPS được kiểm soát.

Fix 3 (`artifacts/gps-alignment-fix3/`,
`artifacts/gps-fusion-round5-vo/`):

- Dùng chronological split 60/40 trên 758 RTK quality-4 sample; mọi fit và
  chọn clock offset chỉ nhìn 454 sample calibration, 304 sample holdout chỉ
  dùng báo cáo.
- Transform chain chính thức WGS84 → ECEF → GPS-world/ENU → world → SLAM
  giảm calibration median 0,790 → 0,607 m so với rigid fit, nhưng holdout
  median xấu hơn 1,534 → 2,120 m. Offset được chọn trên calibration là
  +0,10 s.
- `transform_gps_imu` của recording là identity; camera–IMU lever arm
  0,185 m. Ablation lever arm cần reference orientation và không cải thiện
  rõ, nên không đưa vào fusion để tránh ground-truth leakage.
- Fusion vòng 5 chỉ cải thiện max error sau 2 s 21,557 → 20,760 m, vẫn
  không stable ≤5 m trong 10 s; median trajectory xấu hơn 7,484 → 8,943 m.
  Vì vậy **bác transform chain làm alignment chính**, giữ artifact như
  failure analysis và giữ rigid baseline. Blocker còn lại là chất lượng/bias
  của fix sau dropout, không còn là thiếu transform-chain implementation.

## 6. Rủi ro & fallback

- **Garage low-texture / motion blur**: ORB đói feature → dropout dài. Chấp
  nhận, báo tỷ lệ dropout; EKF được thiết kế sống qua dropout. Không đổi
  detector giữa chừng trừ khi office_loop cũng fail (fallback: GFTT + ORB
  descriptor; đây là nhánh implementation riêng vì phải bucket GFTT
  keypoints rồi gọi ORB `compute`, không coi là đổi một dòng).
- **FPS**: 800×400 + ORB ~1000 kp, ước < 20 ms/frame CPU. Nếu chậm: giảm
  nfeatures trước, không vá đa luồng.
- **Scale sai hệ thống** (baseline/fx): lộ ngay ở gate #2 — so path length
  VO vs reference cùng đoạn.

## Quy tắc dữ liệu (kế thừa Bước 1, không thương lượng)

- Reference poses: chỉ evaluation. VO không nhìn.
- Không tune tham số trên cửa sổ/đoạn được chấm.
- Không trim outlier khỏi KPI báo cáo.
- Mọi số trong báo cáo do script sinh.
