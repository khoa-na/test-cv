# STEP 3 — Landmark DB + correction (B2 12%, B5 8%)

Mục tiêu: dựng landmark database từ **mapping traversal** (garage_3) của parking
garage, re-identify trên **query traversal** (garage_2, khác mùa/giờ — chiều
chốt ở mục 0), và biến match thành **global position update**
qua hook `LocalizationFusion.update_position` đã có sẵn — không sửa EKF, không
sửa state machine, không hạ ngưỡng nào.

Hai số phải ra: **B2 recall** (≥85% đạt, ≥90% xuất sắc) và **B5 quantitative
report** trong garage khi GPS mất.

## 0. Dữ kiện đã xác minh

Số dưới đây đo bằng script probe ad-hoc trong scratchpad, **chưa** phải artifact
chính thức. Ràng buộc: `benchmark_landmark_reid.py` phải tự sinh lại toàn bộ
(offset liên-traversal, coverage, residual) vào `artifacts/landmark-reid/` —
không copy tay số từ mục này vào báo cáo.

Đưa poses của cả 4 recording về ECEF chung bằng transform chain
(`inv(transform_s_as) · inv(scale) · transform_w_gpsw · inv(transform_e_gpsw)`,
đúng chain đã dùng ở Fix 3):

| Cặp | Khoảng cách gần nhất | Kết luận |
|---|---|---|
| garage_2 ↔ garage_3 | 0.0 m, 98–100% frame nằm trong 10 m của traversal kia | **cặp B2 duy nhất** |
| office_loop ↔ mọi cặp khác | ≥ 277 m | không overlap |
| neighborhood ↔ garage | ≥ 850 m | không overlap |

- garage_2 = `recording_2021-02-25_13-39-06`, 5485 pose, 852.4 m, 182 s.
- garage_3 = `recording_2021-05-10_19-15-19`, 5253 pose, 788.4 m, 174 s.
- Khác mùa **và** khác giờ (13:39 tháng 2 vs 19:15 tháng 5) → ánh sáng lối vào
  khác hẳn; phần trong hầm là đèn nhân tạo.

**Chiều DB/query chốt theo chất lượng NMEA thật, không theo recall.** Cả hai
recording đều có `septentrio.nmea` thật:

| Recording | quality 0 | 1 | 2 | 4 | 5 | run quality-0 dài nhất |
|---|---|---|---|---|---|---|
| garage_2 | 526 | 481 | 15 | 905 | 45 | 355 msg ≈ 33 s |
| garage_3 | 178 | 862 | 37 | 752 | 20 | 70 msg ≈ 6.5 s |

Nguyên tắc chọn (chốt trước mọi số): **map dựng từ session có GPS tốt hơn,
localize session có mất tín hiệu thật**. → **DB = garage_3, query = garage_2**.
Lý do cụ thể: đoạn GPS-denied thật của garage_3 chỉ ~6.5 s ≈ 30 m, nếu lấy
garage_3 làm query thì B5 gần như hoàn toàn dựa vào kịch bản cắt nhân tạo;
garage_2 có outage thật 33 s và cũng là session mà vòng 3–5 đã benchmark, nên
B5/B8 nối tiếp được. Chiều ngược lại vẫn phải báo (mục 6) làm sanity check —
không phải để chọn chiều nào số đẹp hơn.

> **Mục 0 đã bị kết quả gate #1 thay thế một phần — đọc mục 12 trước.** Các số
> ECEF/offset dưới đây là khảo sát ban đầu, không còn là cơ sở gán nhãn.

**Bẫy 1 — hai recording không cùng frame.** Reference của hai traversal lệch
nhau một offset cứng ENU ≈ `(+0.60, −2.37, −4.34) m` (E, N, U). Sau khi khử
offset đó: nearest-neighbour 3D median 1.72 m / p95 5.85 m; 2D median 1.13 m /
p95 3.56 m. Trước khi khử: 2D-nn có `|dz|` median 5.16 m — tức **nếu gán ground
truth bằng khoảng cách 2D thuần thì 79% cặp lệch tầng**.

Hệ quả bắt buộc:

- Ground truth B2 phải dùng gate 3 thành phần (mục 6), không dùng 2D thuần.
- Có một **sàn sai số đánh giá** giữa hai traversal; mọi số B5 phải báo kèm nó
  và không được claim chính xác hơn nó. **1.72 m ở trên đo qua transform chain
  ECEF, không phải qua phép transform mà mục 2 chốt cho evaluation** (rigid-fit
  reference → ENU; riêng garage_2 fit này có residual median 1.117 m /
  p95 7.06 m theo `artifacts/gps-fusion-round4-vo/`). Benchmark phải **đo lại**
  sàn này trong đúng frame đã chọn và ghi ra artifact. Giá trị này **chỉ dùng
  để diễn giải metric evaluation**, không được truyền ngược vào covariance,
  config hay bất kỳ quyết định nào của production pipeline.
- Câu hỏi mở phải trả lời ở gate #1: offset `−4.34 m` theo trục U là sai số
  đăng ký giữa hai recording, hay hai traversal thật sự đi khác tầng? Kiểm bằng
  ảnh (5–10 cặp nearest-neighbour) chứ không suy đoán. Nếu khác tầng thật thì
  **không** đổi ground truth sang arc-length (cùng arc-length nhưng khác tầng
  không phải cùng place). Chỉ giữ các đoạn đã xác minh thật sự overlap/cùng
  tầng; nếu không còn đoạn hợp lệ thì kết luận bộ dữ liệu không đủ cặp
  cross-traversal cho B2 toàn tuyến.

**Dev probe (phải disclose trong báo cáo).** Đã chạy thử retrieval trên toàn
tuyến garage — đây là dữ liệu sẽ được chấm, nên tính là dev contamination cùng
loại với dev segment 1800 frame của Bước 2. Cấu hình: keyframe mỗi 30 frame
(≈4.7 m), descriptor thumbnail 64×32 patch-normalized, cosine similarity,
183 DB / 176 query keyframe, gate positive 2D ≤ 5 m và `|dz|` ≤ 2 m. Probe chạy
**chiều ngược** với chiều chốt ở trên (DB = garage_2), nên số dưới đây chỉ dùng
để chọn kiến trúc, không phải dự báo cho chiều nộp.

| Regime | Coverage | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|
| thumbnail | 0.64 | 0.504 | 0.726 | 0.752 | 0.770 |
| + sequence window 5 | 0.64 | 0.690 | 0.796 | 0.805 | 0.823 |

Đọc số này đúng cách: recall tính **chỉ trên query có ít nhất một positive**;
coverage 0.64 là do DB thưa 4.7 m so với bán kính 5 m, không phải do descriptor.
Retrieval bão hoà quanh 0.82 ở top-20 → *retrieval một mình không đạt B2*. Ba
đòn bẩy còn lại: DB dày hơn (≤2 m), spatial prior nhân quả, và geometric
verification. Kiến trúc mục 4 được chọn từ đây, trước khi có số cuối.

## 1. Phạm vi

**Làm:**

- `pipelines/landmark_db.py`: build DB từ mapping traversal, query + association.
- Position update vào fusion qua `update_position` (hook đã tồn tại).
- `benchmarks/benchmark_landmark_reid.py` (B2), `benchmarks/benchmark_garage_localization.py` (B5).

**Không làm ở bước này:** ghost projection + Jacobian semantic (bonus của đề,
chỉ mở sau khi MVP đạt), NetVLAD/EigenPlaces/AnyLoc ONNX (bonus, chỉ nếu MVP
fail và còn thời gian), pothole landmark trên 4Seasons (audit Bước 0 đã kết
luận detector RGB nghi FP trên monochrome → pothole adapter benchmark riêng
trên Fan stereo, **không sinh pothole giả để tính recall**), loop closure /
bundle adjustment / pose graph.

## 2. Frame contract (làm trước mọi thứ khác)

**Bẫy 2 — hai traversal replay NMEA với datum ENU riêng.** `load_nmea_replay`
hiện lấy datum là fix quality-4 đầu tiên *của chính recording đó*. Map dựng
trong frame của traversal này sẽ không dùng được cho traversal kia.

Chốt:

- Thêm tham số `datum: tuple[float, float] | None` cho `load_nmea_replay`.
  `None` = hành vi cũ (mọi benchmark vòng 3–5 tái lập nguyên vẹn — bắt buộc
  chạy lại regression proxy để chứng minh).
- **Map frame** = ENU với datum cố định = fix quality-4 đầu tiên của garage_2
  (`48.19529599, 11.62341647` — datum vòng 3–5 đang dùng; chọn cố định, không
  chọn theo số). DB landmark, fusion output của cả hai traversal đều sống trong
  frame này.
- **Production alignment mode chốt: `raw_enu` + datum chung.**
  `alignment_mode="reference_rigid"` **bị cấm trong production path của Bước 3**
  — nó định nghĩa frame làm việc bằng chính reference pose. Vòng 3–5 chạy
  `reference_pose_rigid_2d_offline` (xác nhận trong
  `artifacts/gps-fusion-round4-vo/benchmark.json`), nên **đây chính là lý do B5
  config A ở mục 7 là baseline mới, không phải tái lập vòng 4/5** — đổi frame,
  không phải đổi thuật toán.
- **Chiều fit evaluation bắt buộc:** vòng 3–5 fit
  `NMEA raw ENU → reference frame riêng`; Bước 3 cần chiều ngược mục đích:
  `reference XY riêng → common ENU map frame`. Không gọi lại
  `load_nmea_replay(..., alignment_mode="reference_rigid")` rồi coi output là
  common map frame.
- **Hai fit riêng, datum chung:** reference của mỗi recording fit với quality-4
  NMEA **của chính recording đó**, sau khi cả hai đã đổi sang ENU datum chung.
  Cấm fit reference garage_2 vào NMEA garage_3 hoặc ngược lại. Chỉ trong
  benchmark/evaluation.
- Fusion và DB production chỉ dùng XY/heading. Gate `dz` của B2 dùng thành phần
  Z của **chính phép fit 3D này** (reference XYZ → ENU 3D, neo bằng cao độ
  ellipsoid của fix quality-4). Gate #1 đã đo: `dz` median +0.17 m giữa hai
  traversal, tức **không cần offset Z liên-traversal** — yêu cầu ước lượng
  offset bằng ảnh trong bản kế hoạch cũ đã bị bãi bỏ (mục 12). Z chỉ tạo nhãn
  evaluation, không đi vào spatial prior, DB hay measurement EKF.
- Split dùng lại đúng convention Fix 3: **chronological 60/40** trên sample
  quality-4 của từng recording. Không random split (sample GPS tương quan thời
  gian mạnh, random split rò thông tin giữa hai phía).
- Artifact phải lưu chiều transform, calibration samples, offset Z và residual
  calibration/holdout. Mọi fitting chỉ dùng calibration split; holdout không
  được dùng để chọn transform.

## 3. DB schema + build

Schema mỗi entry (bám đề bài `L_i = {id, class, p_3D, d_visual, t_first, t_last, n_obs}`
+ covariance theo PLAN dòng 160):

```
id            int
class         "keyframe_orb"          # MVP chỉ một class; semantic là bonus
position      [x, y, 0] map frame     # pose base/vehicle planar của keyframe
heading       float                   # yaw base/vehicle trong map frame
descriptor    float32[2048]           # global descriptor retrieval
keypoints     float32[N,2]            # pixel trong ảnh left
points_3d     float32[N,3]            # cam0 optical frame của keyframe
orb_desc      uint8[N,32]             # descriptor local để verification
t_first,t_last float                  # MVP: bằng nhau nếu chưa gộp observation
n_obs         int                     # MVP: 1 nếu entry chỉ từ một keyframe
covariance    float32[2,2]            # bất định vị trí keyframe trong map frame
```

`position[2] = 0` thể hiện DB production là planar SE(2), không phải độ cao
survey-grade. Reference Z dùng để tạo nhãn B2 nằm trong benchmark artifact,
không trộn vào schema production. Metadata bắt buộc ghi
`point_frame="cam0_optical"` và `pose_frame="base_link_se2"`.

Build (mapping traversal = **garage_3**, mục 0):

1. Chạy fusion `--odom vo` trên garage_3 với NMEA thật (`raw_enu`, datum chung)
   → pose log nhân quả.
   **Đây là nguồn vị trí keyframe mặc định: hệ tự dựng map bằng chính output
   của nó, không đọc reference pose.**
2. Keyframe khi đi được ≥ 2 m hoặc quay ≥ 15° kể từ keyframe trước (mục tiêu
   ~390 keyframe / 788 m).
3. Mỗi keyframe: ORB ≤ 500 kp (bucket lưới như `StereoVO`), stereo match →
   depth `Z = fx·b/d` với đúng gate `d_min/d_max` từ calibration → điểm 3D
   giữ trong `cam0 optical frame`. Không dựng full `T_map_camera` giả từ pose
   SE(2).
4. Global descriptor: ảnh left → 64×32 `INTER_AREA` → patch-normalize block
   8×8 → L2 normalize (2048-d). Deterministic, không training, không dependency
   mới.
5. `covariance` keyframe = covariance global của fusion tại timestamp đó
   (`global_position_covariance()`), symmetrize và eigenvalue floor
   `(0.25 m)²` chỉ để tránh ma trận suy biến. **Không** cộng thêm gì khác vào
   đây: sai số chưa mô hình hoá nằm ở `sigma_model` của mục 5, và registration
   residual đo bằng reference không được đi vào covariance nào của production.
6. Serialize `.npz` một file, kèm metadata JSON: nguồn pose, datum, số keyframe,
   spacing thực tế, frame contract, phiên bản config. MVP chưa gộp track thì
   đặt thẳng `t_first=t_last`, `n_obs=1`; không claim multi-observation.

**Ablation bắt buộc (không phải baseline):** biến thể `--map-source reference`
dùng reference pose làm vị trí keyframe = "survey-grade map", cho biết trần lý
thuyết khi map không có drift. Artifact riêng, gắn cờ `uses_ground_truth: true`,
**không được dùng làm số B2/B5 chính** — đúng tiền lệ lever-arm ablation ở Fix 3.

## 4. Association pipeline (4 tầng, thứ tự cố định)

**Keyframe query dùng đúng luật khoảng cách của DB** (≥ 2 m hoặc ≥ 15°), không
dùng luật thời gian. Lý do: tầng 3 so đường chéo `i−4..i ↔ j−4..j`, giả định
spacing hai bên tương đương. Lấy keyframe theo 2–5 Hz thì trên ramp chậm spacing
query tụt xuống dưới 1 m trong khi DB vẫn 2 m, đường chéo lệch hệ thống và
sequence score mất tác dụng đúng chỗ khó nhất. Rate 2–5 Hz chỉ là **trần tần
suất** để giữ B7, không phải tiêu chí chọn keyframe.

1. **Spatial prior (nhân quả):** loại DB entry cách vị trí global hiện tại của
   fusion > `prior_radius`. Prior lấy từ pose EKF của chính hệ, **tuyệt đối
   không từ reference**. Bán kính scale theo covariance:
   `r = clip(3·sqrt(trace(P)), 10 m, 50 m)`.
   Khi chưa init global (`global_initialized=false`) → bỏ tầng này, xét toàn DB.
   Nếu không có match geometric hợp lệ trong 5 query keyframe liên tiếp, query
   kế tiếp chạy full-DB reacquisition một lần rồi reset bộ đếm; tránh deadlock
   khi VO drift đã đẩy true place ra ngoài `r_max`. B2 báo riêng số lần
   reacquisition.
2. **Descriptor top-K:** cosine similarity, K = 20 (probe cho thấy top-20 là
   nơi retrieval bão hoà; K lớn hơn chỉ tốn verification).
3. **Sequence consistency:** điểm trung bình similarity dọc đường chéo cửa sổ
   W = 5 keyframe (query `i−4..i` ↔ DB `j−4..j`), rerank top-K. Probe: +0.19
   R@1 so với descriptor thuần.
4. **Geometric verification:** ORB mutual-KNN ratio match query↔candidate →
   `solvePnPRansac(points_3d trong DB cam0, keypoints query)` → **invert** như
   `StereoVO.invert_pnp_transform` để ra relative camera motion. Đổi camera
   motion sang base/vehicle SE(2) bằng **đúng `StereoVO.camera_motion_to_se2`**,
   rồi compose với pose map-frame của DB keyframe để ra pose query. Không đưa
   pose camera trực tiếp vào EKF base pose.

   **Một convention duy nhất.** Pose map trong DB do VO tích phân sinh ra bằng
   axis-swap của `camera_motion_to_se2`; nếu landmark dùng `TS_cam_imu` thì hai
   nhánh lệch nhau một rotation mounting cố định. Đo thật: `TS_cam_imu` chỉ
   lệch signed-axis permutation gần nhất **0.903°**; Frobenius residual
   `0.0223`, còn `0.0135` là phần tử residual lớn nhất. Chốt: dùng lại
   `camera_motion_to_se2` ở cả VO và landmark nên không tạo sai lệch convention
   giữa hai nhánh. Ghi mounting residual như giới hạn calibration riêng; không
   coi nó được "hấp thụ" bởi lever arm vì góc và độ lệch tịnh tiến là hai đại
   lượng khác nhau. Muốn chuyển sang `TS_cam_imu` thì phải đổi cả `StereoVO`,
   không đổi một nhánh.

Thông số geometric chốt theo `StereoVO`, không tune lại trên B2:
mutual-KNN ratio `0.8`, RANSAC reprojection `2.0 px`, tối thiểu `15` inlier,
`100` iteration, confidence `0.999`. Accept khi đủ inlier và reprojection RMS
`≤ 2.0 px`.

Giữ ranked list sau tầng 3 để tính Recall@1/@5. Chỉ nhận **một** verified
match/keyframe query (candidate hạng cao nhất qua tầng 4) để correction.

## 5. Correction vào EKF

```python
accepted, nis = fusion.update_position(position_xy, covariance, gate=True)
```

- `position_xy` = thành phần x,y của **base/vehicle pose** trong map frame sau
  khi compose relative PnP với DB keyframe pose.
- Measurement covariance chốt trước khi chạy:

  ```
  sigma_geom = max(0.25 m,
                   median_depth / fx * reprojection_rms / sqrt(n_inliers))
  sigma_model = 1.5 m
  R_landmark = covariance_db
               + I * (sigma_model^2 + sigma_geom^2 + (0.185 m)^2)
  ```

  `sigma_model = 1.5 m` **kế thừa noise model GPS đã có trong code**, không đặt
  tay: `LocalizationFusion._gps_covariance` dùng `base_sigma[1] = 1.5 m` cho fix
  quality-1, và cả hai session garage đều chủ yếu quality-1 (862 và 481 message,
  mục 0). Đại lượng nó đại diện là **bias GPS giữa hai session** — map neo bằng
  GPS garage_3, query neo bằng GPS garage_2 — cộng sai số planar/PnP chưa mô
  hình hoá. Đây là sai số production thật nhưng không đo được nếu không dùng
  reference, nên phải lấy mốc reference-free; không lấy từ reference pose, không
  lấy từ kết quả B2/B5.
  `0.185 m` chỉ là camera–IMU lever-arm allowance bảo thủ, không đại diện cho
  mounting-angle residual ở mục 4. Mounting residual 0.903° đóng góp
  `sin(0.903°)·d_match` ≈ **0.08 m ở tầm match 5 m** — bị `sigma_model` át hoàn
  toàn ở tầm hiện tại; nếu nới tầm match thì phải thêm hẳn term
  `(sin(0.903°)·d_match)²`, không im lặng dựa vào lever arm. Artifact production
  báo từng thành phần `covariance_db`, `sigma_model`, `sigma_geom`, lever arm.
  Registration residual đo bằng reference chỉ nằm trong evaluation artifact và
  in cạnh metric; code correction không được đọc giá trị đó.
- Gate NIS dùng `normal_nis_threshold = 5.991` có sẵn. Không thêm ngưỡng mới,
  không sửa `FusionConfig` cho hợp số đẹp.
- Landmark correction chỉ gọi khi `global_initialized=true`; MVP là fallback
  sau GPS nên luôn có global anchor trước khi vào hầm. Khi chưa có anchor,
  full-DB query vẫn có thể chạy để đo retrieval nhưng không được âm thầm dùng
  `update_position` như một API initialization.
- Không update heading từ landmark ở MVP (API heading hiện chỉ có đường
  GPS-anchor). Ghi là giới hạn, không vá vội.
- Policy chốt **trước** khi thấy số: landmark update **luôn bật**, đúng phương
  trình `x̂ = f(x, z_GPS, u_VO, z_landmark, z_lane)` của đề. Biến thể
  "chỉ chạy khi state ≠ GOOD" là **ablation báo kèm**, không phải để chọn số đẹp.

Không đổi một dòng nào trong `LocalOdometryEKF`, `GPSIntegrityMonitor`,
`FusionConfig`.

## 6. Benchmark B2 (`benchmarks/benchmark_landmark_reid.py`)

**Định nghĩa positive (chốt trước, không đổi sau khi thấy recall):** query
keyframe `q` và DB entry `d` là cùng chỗ khi

```
‖xy_q − xy_d‖ ≤ 5.0 m   AND   |z_q − z_d| ≤ 2.0 m   AND   |Δheading| ≤ 45°
```

**Frame để tính nhãn (không dùng mục 0):** cả `xy` lẫn `z` lấy từ **một phép
fit duy nhất** của mục 2 — `map_frame.fit_reference_to_map`, rigid 3D Kabsch
`reference XYZ → common ENU`, hai fit riêng, calibration/holdout split. Không
offset Z bổ sung (gate #1: `dz` median +0.17 m). Offset ECEF
`(+0.60, −2.37, −4.34) m` ở mục 0 chỉ là số khảo sát ban đầu, **không** dùng để
gán nhãn — trộn hai phép biến đổi vào cùng một nhãn là dùng hai hệ toạ độ khác
nhau.

Gate `|dz|` là bắt buộc vì garage nhiều tầng. Báo thêm sensitivity ở 3 m và
10 m như số phụ, **headline vẫn là 5 m**.

Metric báo cáo:

- **Retrieval Recall@1, Recall@5** — ranked list sau tầng 3, trước PnP; đây là
  metric B2. Query eligible khi có ít nhất một positive trong **toàn DB**, không
  phải chỉ trong candidate set còn lại sau spatial prior.
- **Verified recall@1** = số query có verified match đúng / số query eligible.
  Đây là metric end-to-end phụ nối B2 sang correction B5; không đánh tráo với
  retrieval Recall@K.
- **Coverage** = tỷ lệ query keyframe có positive; recall không được tính lấp
  liếm bằng cách bỏ query khó ra khỏi mẫu số mà không báo coverage.
- **Verified precision** = accepted đúng / tổng accepted.
- **Incorrect-accept rate** = accepted sai / tổng accepted (`1 - precision`).
  Không gọi đại lượng này là false-positive rate vì mẫu số không phải toàn bộ
  negative query.
- Localization error của match được accept (median/p95).
- **NIS reject rate của landmark update** + histogram NIS. `R_landmark` quá chặt
  sẽ bị gate 5.991 loại hàng loạt và hiện ra dưới dạng "coverage thấp"; không có
  số này thì rất dễ đổ nhầm cho descriptor.
- Số keyframe DB, spacing thực tế, thời gian query trung bình/p95 và số
  full-DB reacquisition.

Hai regime, báo **cả hai**, headline chốt trước:

| Regime | Prior | Vai trò |
|---|---|---|
| **Full pipeline** (headline B2) | spatial prior từ pose EKF nhân quả | đúng module đề bài mô tả, số nộp |
| Prior-free retrieval | không prior | ablation, cho thấy phần nào do descriptor |

Chiều query/DB: **DB = garage_3, query = garage_2** — chốt ở mục 0 theo chất
lượng NMEA, trước mọi số recall. Báo thêm chiều ngược lại làm sanity check; nếu
hai chiều lệch nhau lớn thì có bug frame, không phải "chọn chiều tốt hơn".

Artifact: `artifacts/landmark-reid/` — JSON + PNG do script sinh.

## 7. Benchmark B5 (`benchmarks/benchmark_garage_localization.py`)

Chạy fusion đầy đủ trên garage_2 (query) với DB dựng từ garage_3:

- Cấu hình A — baseline: VO + NMEA thật, **không** landmark. Cùng recording với
  vòng 4/5 nhưng **frame khác** (`raw_enu` + datum chung thay cho
  `reference_rigid`, mục 2), nên đây là baseline mới, không phải tái lập. Chênh
  lệch A vs vòng 4 phải được giải thích bằng đổi frame, không bỏ qua.
- Cấu hình B — có landmark correction.
- Cấu hình C — giữ GPS đến frame cuối ngay trước cửa hầm để initialize/latch,
  sau đó replay quality-0 và không position từ timestamp vào hầm đến timestamp
  ra hầm (kịch bản S1/B5 thuần visual).

garage_2 có outage thật dài nhất 355 msg ≈ 33 s, nên A/B đo được GPS-denied
thật; C là kịch bản cắt dài hơn có kiểm soát. Báo rõ đoạn nào là thật, đoạn nào
là cắt nhân tạo.

Timestamp vào/ra hầm được chốt ở gate #1 bằng ảnh và lưu vào một segment JSON
trước khi chạy B5. Cả A/B/C dùng nguyên segment này; không trim theo error.

Metric (so reference garage_2 trong map frame):

- Position error median/p95/max, riêng cho đoạn trong hầm, **kèm error theo
  từng đoạn 30 s và full percentile curve**. Gate #2 đã chứng minh median toàn
  tuyến của sequence này mong manh: phân bố dốc đứng quanh phân vị 50 (p45 =
  5.49 m, p55 = 9.96 m) nên ~1% mẫu transient đủ đẩy median 2 m. **Cấm** kết
  luận A/B/C chỉ bằng median toàn tuyến.
- Drift cuối đoạn = `‖error_exit‖ / reference_path_length_inside × 100%`, kèm
  error-vs-distance curve; không dùng alignment lại trajectory sau khi chạy.
- Localization coverage: tỷ lệ thời gian trong hầm có landmark update accept,
  tách riêng "không có verified match" và "có match nhưng NIS reject".
- Recovery khi ra khỏi hầm: error tại thời điểm GPS re-lock, dùng lại định
  nghĩa B8 hiện có.
- Sàn đánh giá liên-traversal đo ở gate #3 (mục 0/2) in cạnh mọi con số error;
  production pipeline không đọc giá trị này.
- B7 đo wall-clock của **toàn pipeline VO + fusion + landmark query** trên toàn
  video, báo FPS tổng, p95 frame latency và phần thời gian landmark. `ms/query`
  riêng không được dùng thay cho end-to-end FPS.

Không claim "<10 lux" hay IR: dataset không có lux metadata. Gọi đúng tên là
"parking garage GPS-denied", như PLAN đã chốt.

Artifact: `artifacts/garage-localization/`.

## 8. Tests (`tests/test_landmark_db.py`)

1. Descriptor: deterministic; bất biến với scale/offset độ sáng toàn ảnh
   (patch-normalize phải khử được), thay đổi khi nội dung ảnh đổi.
2. Retrieval synthetic: DB gồm N ảnh pattern khác nhau, query = một ảnh DB đã
   nhiễu nhẹ → top-1 đúng.
3. Sequence scoring: candidate đúng thứ tự thắng candidate cùng similarity
   nhưng sai thứ tự.
4. Geometric verification: scene 3D synthetic project từ hai pose đã biết →
   PnP + invert + camera/base transform + compose trả lại base pose đúng
   < 0.1 m / 1°; cặp không liên quan bị reject.
5. GT labeler: hai pose cùng (x,y) khác tầng `dz = 3.5 m` → **negative**;
   cùng chỗ ngược hướng 180° → negative; cùng chỗ cùng hướng → positive.
6. Serialize round-trip `.npz`: mọi field giữ nguyên, `orb_desc` bit-exact.
7. Fusion contract: landmark update gọi `update_position`, bị reject khi NIS
   vượt ngưỡng, và **không** đụng vào velocity/omega của local EKF.
8. Spatial prior: candidate ngoài bán kính prior bị loại; prior đọc từ pose
   fusion, test fail nếu code chạm vào reference pose.
9. Datum regression: `load_nmea_replay` không truyền `datum` cho ra kết quả
   byte-identical với hành vi cũ.
10. Frame direction: synthetic `reference → common ENU` fit đúng; gọi nhầm
    chiều `common ENU → reference` phải làm test fail.
11. Reacquisition: 5 query liên tiếp không match làm query kế tiếp bỏ prior,
    sau đó reset đúng một lần.
12. Metric denominator: spatial prior loại positive vẫn làm recall giảm, không
    làm query biến mất khỏi mẫu số eligible.
13. No-reference covariance: thay đổi hoặc xoá registration residual trong
    evaluation artifact không làm đổi `R_landmark` hay kết quả production.

## 9. Thứ tự làm + gate dừng

| # | Việc | Gate qua bước sau |
|---|---|---|
| 1 | Xác minh offset U −4.34 m và segment vào/ra hầm bằng 5–10 cặp ảnh nearest-neighbour | kết luận rõ "lệch đăng ký" hay "khác tầng"; nếu khác tầng, chỉ giữ overlap/cùng tầng hoặc kết luận không đủ dữ liệu; segment JSON được chốt |
| 2 | Implement `datum` cho `load_nmea_replay`, evaluation fitter + regression và positive check frame mới | (a) `datum=None` tái lập vòng 3 chính xác (median 7.875 m); (b) garage_2 chạy `raw_enu`+datum chung, so reference sau fit evaluation, error xấp xỉ vòng 4 (~7.5 m). Lệch lớn = bug frame, và bug đó im lặng nếu chỉ chạy (a) |
| 3 | Sinh registration residual trong đúng evaluation frame từ calibration split, đo trên holdout | artifact có median/p95/RMSE và transform direction; chỉ dùng báo sàn evaluation, production không đọc |
| 4 | `benchmark_vo_drift` trên garage_3 (session mapping, chưa từng đo B1) | có số drift, bất kể đạt hay không; drift này là trần chất lượng của map |
| 5 | DB builder + tests 1–6, 10 | pass, DB spacing thực tế ≤ 2.5 m; frame direction đúng |
| 6 | Association pipeline + B2 benchmark | số retrieval recall/verified recall/coverage/precision ra được, **bất kể đạt hay không** |
| 7 | Correction + tests 7–13 + B5/B7 benchmark | 3 cấu hình A/B/C có số; không sửa EKF |
| 8 | Cập nhật PLAN + STEP3 kết quả | — |

Gate #1 sai → dừng, không code DB. Gate #2 fail → dừng, vì mọi số sau đó nằm
sai frame.

## 10. Rủi ro & fallback

| Rủi ro | Bằng chứng hiện có | Ứng phó |
|---|---|---|
| Retrieval bão hoà 0.82, không chạm 85% | dev probe mục 0 | DB dày 2 m (thay vì 4.7 m) + spatial prior + verification. Nếu vẫn thiếu: descriptor học sẵn ONNX là bonus, **chỉ khi còn thời gian**, và phải là ONNX CPU đúng ràng buộc đề |
| Hầm tối / ít texture → ORB đói feature ở tầng 4 | B1 garage_2 pass 1.759% nên VO còn sống; **garage_3 chưa có cửa sổ B1 nào** (788 m nhưng chưa chạy) | Chạy `benchmark_vo_drift` trên garage_3 trước khi dựng DB — map dựng từ session chưa đo drift là rủi ro mù. Hạ `min_inliers` là **cấm** sau khi thấy số |
| Map tự dựng drift → landmark kéo pose sai | drift VO trong hầm + sàn đăng ký evaluation đo ở gate #3 | DB covariance + production `sigma_model`/geometry/lever-arm covariance ở mục 5, NIS gate, và ablation survey-grade map cho biết trần; registration residual chỉ dùng báo cáo |
| Query traversal đi khác tầng thật | offset U −4.34 m chưa giải thích | Gate #1 giải quyết trước khi code |
| Spatial prior khóa nhầm vùng sau drift | true place có thể ra ngoài bán kính 50 m | full-DB reacquisition sau 5 query miss, báo riêng số lần |
| FPS tụt vì verification | verification chỉ chạy keyframe rate | Headline giữ K=20; báo K=10 như ablation hiệu năng nếu cần, không đổi headline sau khi thấy số |

## 11. Quy tắc dữ liệu (kế thừa Bước 1–2, không thương lượng)

- Reference pose: chỉ evaluation và chỉ trong benchmark script. DB mặc định
  dựng bằng output của chính hệ; biến thể reference bị gắn cờ và không lên
  headline.
- Không đổi định nghĩa positive, bán kính GT, ngưỡng inlier hay policy
  always-on/fallback-only **sau** khi nhìn thấy recall.
- Không trim outlier khỏi KPI. Coverage luôn báo cạnh recall.
- Mọi số trong báo cáo do script sinh, artifact riêng thư mục mỗi vòng.
- Disclosure bắt buộc kèm số B2: dev probe ở mục 0 đã chạy trên toàn tuyến
  garage — cùng loại contamination với dev segment 1800 frame của Bước 2.

## 12. Kết quả gate #1–#2 (đã chạy)

Artifact: `artifacts/garage-pair-audit/` (`audit.json`, `segments.json`,
`same_level_*.png`, `different_level_*.png`). Code:
`data_tools/map_frame.py`, `data_tools/audit_garage_pair.py`.

### Gate #1 — chênh cao là lệch đăng ký hay khác tầng? **Cả hai câu trả lời đều là "không như mục 0 đoán"**

Trong common map frame của mục 2 (reference → ENU 3D, Kabsch rigid không scale,
neo bằng RTK quality-4 của chính recording, datum chung `48.19529599,
11.62341647`, cao độ ellipsoid 540.87 m):

| Đại lượng | Giá trị |
|---|---|
| Fit garage_2 (query) | calibration median 0.833 m / p95 1.662 · holdout median 1.478 / p95 9.415 |
| Fit garage_3 (mapping) | calibration median 1.631 m / p95 2.915 · holdout median 4.900 / p95 11.946 |
| Cross-traversal nearest-neighbour | planar median **0.349 m** / p95 1.896 · 3D median **0.630 m** / p95 2.590 |
| `dz` tại cặp planar < 1 m (n=4630) | median **+0.171 m** |

- **Không có offset đăng ký cứng.** `dz` median 0.17 m. Con số `−4.34 m` ở mục 0
  là hệ quả của việc đi qua transform chain ECEF **và** ép một mô hình offset
  tịnh tiến; đổi sang frame neo bằng GPS thì nó biến mất. Mục 2 do đó **không
  cần** offset Z liên-traversal ước lượng bằng ảnh — bỏ hẳn yêu cầu đó.
- **Hai traversal thật sự chạy khác tầng ở 35% vị trí trùng 2D.** Histogram `dz`
  đa mode: cụm ~−3.3 m (383 mẫu), cụm ~0 m (2988), cụm ~+2.4…+3.1 m (1243) —
  đúng bước một tầng ~3 m. Gate `|dz|` là bắt buộc, và nó hoạt động.
- Xác nhận bằng ảnh: `same_level_03945.png` (dxy 0.26 m, dz −0.10 m) là cùng chỗ
  thật; `different_level_02955.png` (dxy 0.57 m, dz +3.21 m) là query trên sàn
  mái, DB dưới một tầng, cùng toạ độ 2D. Nếu bỏ gate `dz` thì cặp thứ hai thành
  positive giả.
- **Sàn đăng ký evaluation = 0.349 m planar / 0.630 m 3D**, không phải 1.72 m.

Segment hầm (`segments.json`, xác nhận bằng ảnh biên ±3 s, exposure nhảy
50 → 1306 µs khi vào và 1592 → 87 µs khi ra):

| | enter | exit | dài |
|---|---|---|---|
| Covered segment (query garage_2) | 1614256872.74 | 1614256923.13 | 50.4 s |
| Outage GPS thật | 1614256892.10 | 1614256927.50 | 35.4 s |

Outage thật **không trùng biên** covered segment: receiver còn giữ fix 19 s sau
khi xe đã vào mái che, và mất fix thêm 4.4 s sau khi ra. B6 phải báo đúng độ trễ
này, không coi "vào mái che" = "mất GPS".

### Gate #2 — datum chung + positive check frame mới

- `load_nmea_replay(..., datum=...)` đã có; `datum=None` byte-identical với
  hành vi cũ (`tests/test_shared_datum.py`).
- (a) Regression: chạy lại garage_2 `reference_rigid` cho **median 7.48 m**,
  khớp `artifacts/gps-fusion-round4-vo/` (7.484). Đường cũ không bị chạm.
- (b) Positive check frame mới: raw_enu + datum chung, chấm trong map frame cho
  median 9.64 m. **Chênh 2.2 m này không phải bug frame** — bằng chứng:

  | Đoạn | fitted-GPS (vòng 4) | raw_enu map frame |
  |---|---|---|
  | 0–30 s | 2.01 m | 2.00 m |
  | 30–60 s | 12.28 m | 12.34 m |
  | 60–120 s | 0.98 m | 0.98 m |
  | 120–200 s | 14.86 m | 14.86 m |

  Map quỹ đạo raw_enu qua đúng rigid transform của vòng 4 thì hai predicted
  trùng nhau median 0.0006 m. Median toàn tuyến lệch chỉ vì phân bố error dốc
  đứng quanh phân vị 50 (p45 = 5.49 m, p55 = 9.96 m), nên ~1% mẫu transient đủ
  đẩy median 2 m.

- **Hệ quả bắt buộc cho mục 7:** median toàn tuyến là headline **mong manh** cho
  sequence này. B5 phải báo error **theo đoạn** (và full percentile curve) cạnh
  median tổng; cấm kết luận A/B/C chỉ bằng median toàn tuyến.
