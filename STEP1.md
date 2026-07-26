# Bước 1 — GPS integrity + EKF + state machine (odometry proxy)

Mục tiêu: pipeline fusion end-to-end chạy được và chấm B6/B8 (+B3 synthetic)
**trước khi** có stereo VO. Odometry là proxy từ reference pose + nhiễu;
Bước 2 swap sang VO thật, không đổi API.

KPI liên quan: B6 handover ≤ 2 s (xuất sắc ≤ 0,5 s), B8 re-lock ≤ 5 m
(xuất sắc ≤ 2 m), B3 U-turn ≤ 2 s (synthetic, nhãn "validation mô phỏng").
Timebox: 4 giờ cho core simulator + EKF + state machine + unit test; dành
thêm tối đa 2–3 giờ cho căn ENU/NMEA thật và benchmark/plot. Quá giờ: cắt
simulator nâng cao và thống kê heading-reversal garage, nhưng giữ NMEA replay,
EKF, state machine và benchmark JSON tối thiểu.

> **Post-benchmark:** `STEP1-FIXES.md` supersede các chi tiết về debounce
> quality-0, recovery correction và cách thẩm định alignment. Bảng transition
> bên dưới đã được đồng bộ với Fix 1; các tuning sau benchmark phải đọc cùng
> `STEP1-FIXES.md`, không dùng lại rule cũ.

## Các quyết định sửa so với bản nháp

| Sửa đổi | Lý do |
|---|---|
| Message `fix_quality=0` đầu tiên chuyển sang `DEGRADED`; đủ 3 message liên tiếp mới `LOST`; timeout vẫn `LOST` trực tiếp | Cảnh báo phải xuất hiện ngay nhưng một quality-0 lẻ không nên tạo LOST→re-lock episode giả; debounce 3 vẫn giữ handover dưới KPI |
| Chạy timeout bằng `tick(timestamp)` ở mỗi odometry frame | Khi receiver im lặng không có NMEA callback để tự phát hiện timeout |
| Local EKF ở frame `odom`; GPS chỉ sửa `T_map_odom` | Nếu GPS cập nhật trực tiếp local pose thì `odom → base_link` có thể jump, trái mục tiêu seamless |
| Recovery dùng quality gate trước, recovery-NIS gate nới có kiểm soát và không rớt `LOST` chỉ vì một reject | Sau dropout, drift làm innovation lớn; NIS chuẩn có thể reject vĩnh viễn và không bao giờ re-lock |
| `recovery_required` và consensus buffer thuộc recovery episode, không thuộc riêng state `RECOVERING` | Debounce có thể giữ máy ở `DEGRADED`, hoặc integrity lên `GOOD` trước khi consensus đủ; buộc recovery logic theo state sẽ gây deadlock/mất fast correction |
| `ODOM_proxy` giữ vận tốc có dấu và đổi delta world → vehicle | Dùng độ lớn vector làm mất dấu khi xe lùi; stereo VO Bước 2 cũng trả relative motion trong camera/vehicle frame |
| Thêm `garage_3` làm kịch bản degrade-dần; `office_loop` giữ làm kịch bản thứ ba nếu kịp giờ | Lý do gốc ("data không chứa office_loop") sai — `recording_2020-03-24_17-36-22` đã tải và giải nén (15.177 frames). Giữ garage_3 vì cùng địa điểm với garage_2, tiện đối chứng; office_loop là sequence ngoài trời dài nhất (~326 s) nên đáng làm sim thứ ba, không bắt buộc |
| Đổi “max local jump” thành “correction-induced discontinuity” | Bước dịch chuyển thật của xe không phải jump; metric cần đo phần gián đoạn do global correction gây ra |
| Garage heading-reversal là bonus, synthetic U-turn là bắt buộc | Không để thống kê sự kiện không phải U-turn phố chặn pipeline fusion cốt lõi |

## File sẽ tạo

| File | Nội dung |
|---|---|
| `data_tools/gps_sources.py` | NMEA replay (đường chính) + `GPS_sim` từ reference pose (kịch bản kiểm soát) + `ODOM_proxy` |
| `pipelines/localization_ekf.py` | EKF local/global + GPS integrity state machine + U-turn detector |
| `benchmarks/benchmark_gps_fusion.py` | Chấm B6/B8 trên garage_2 (NMEA thật) + kịch bản sim |
| `tests/test_localization_ekf.py` | Unit tests: predict/update, NIS gate, transitions, re-lock, U-turn, no-jump |

## 1. Nguồn dữ liệu và căn chỉnh thời gian

- **NMEA thật**: parser GGA đã có (`fourseasons.load_nmea_gga`). Căn thời
  gian: `times.txt` là epoch ns (UTC); GGA chỉ có seconds-of-day → ghép
  ngày từ epoch của frame đầu. Replay GGA ở rate gốc (~10 Hz); tại frame
  camera/odometry chỉ phát các measurement có timestamp `≤ current_time`,
  không nội suy quality/HDOP và không dùng measurement tương lai. Kiểm tra
  bằng cách so vị trí RTK (quality 4) với reference pose sau khi đổi hệ —
  lệch phải < vài mét.
- **WGS84 → ENU local**: geodetic → ECEF → ENU quanh datum = fix RTK tốt
  đầu tiên (~20 dòng numpy, không thêm dependency). Trục ENU phải khớp
  hệ của reference pose qua `transform_e_gpsw`/`transform_w_gpsw` trong
  `Transformations.txt` — nếu ghép transform chain mất >30 phút thì fit
  Umeyama 2D (rotation + translation, KHÔNG scale) giữa track RTK và
  reference trajectory metric trên đoạn quality-4. Fallback Umeyama chỉ
  dùng để xác lập frame/evaluation offline, phải ghi rõ là alignment dùng
  reference pose và không được tính như localization online độc lập.
- **`ODOM_proxy`**: delta pose giữa frame liên tiếp từ reference pose,
  nhân `gnss_scale` để ra mét, cộng nhiễu Gauss (per-step, seed cố định)
  + bias yaw nhỏ để tạo drift giả ~1–3%/100 m. Xuất `(dt, dx, dy, dθ)`
  trong frame xe — đúng format stereo VO sẽ xuất ở Bước 2. Phải đổi delta
  world → vehicle bằng heading tại frame trước; chuẩn hóa `dθ` về
  `[-π, π]`. Vận tốc có dấu lấy `v_meas = dx/dt`, không dùng
  `sqrt(dx²+dy²)/dt`; `dy` được log như lateral residual/confidence.

> **Lý do:** replay causal ngăn rò rỉ measurement tương lai; quy ước frame
> và vận tốc có dấu giúp Bước 2 thay proxy bằng stereo VO mà không đổi nghĩa
> của API.

## 2. EKF — thiết kế

Local state `x = (x, y, θ, v, ω)` nằm trong frame `odom`, không phải ENU.
Frame `map` là ENU local; quan hệ giữa hai frame được giữ bởi
`T_map_odom`.

- **Predict** (mỗi odometry step, dt ~33 ms): unicycle
  `x += v·dt·cos θ; y += v·dt·sin θ; θ += ω·dt`. Q từ noise proxy đã biết
  (sau này từ confidence VO).
- **Update odometry**: `(dx, dy, dθ)/dt → (v_meas=dx/dt,
  ω_meas=dθ/dt)`, update trực tiếp v, ω. `dy` không bị ép vào mô hình
  unicycle; dùng để tăng R hoặc đánh dấu odometry kém tin cậy. R từ seed
  noise (Bước 2: từ inlier ratio/reprojection RMS).
- **GPS/global correction**: dự đoán pose global bằng
  `T_map_odom ∘ pose_local`; GPS không update trực tiếp local state.
  Measurement `z = (x_gps, y_gps)` cập nhật target/covariance của
  `T_map_odom`.
- **GPS covariance**:
  `σ_gps = max(σ_floor, hdop · σ_base)` và `R_gps = σ_gps²·I`;
  khởi tạo `σ_base` khoảng 1,5 m (quality 1), 0,05 m (quality 4 RTK),
  0,5 m (quality 2/5). Dùng `σ_floor` để tránh covariance gần 0 và log
  các giá trị thực thay vì tune trên đoạn benchmark.
- **NIS gate bình thường**:
  `ν' S⁻¹ν ≤ 5.991` (χ² 2 DOF, 95%), với `S` gồm covariance global
  predicted + `R_gps`. Log tỷ lệ reject.
- **Local/global tách đôi** (pattern `robot_localization`):
  - Local EKF: chỉ odometry — `odom → base_link`, không bao giờ jump.
  - Global: offset `T_map_odom` (x, y, θ) cập nhật từ GPS/landmark đã qua
    gate. Làm mượt theo thời gian và giới hạn correction mỗi frame thay vì
    set cứng; không phụ thuộc riêng vào FPS.
  - GGA không có heading: khởi tạo yaw của `T_map_odom` từ hướng dịch chuyển
    GPS/local trên baseline ≥ 2 m; sau đó chỉ cập nhật trên baseline ≥ 10 m
    và low-pass để giảm nhiễu. Khi đổi yaw, xoay quanh local pose hiện tại
    (bù translation) để global position không jump.
  - Pose xuất = `T_map_odom ∘ pose_local`.

Landmark update để hook sẵn interface (`update_position(z, R, gate=True)`)
— Bước 3 gọi cùng đường với GPS, không sửa EKF.

> **Lý do:** tách frame như trên giữ local odometry liên tục. GPS và landmark
> chỉ thay đổi quan hệ `map → odom`; đây là đúng điểm cần làm mượt khi re-lock.
> Baseline dài hơn cho cập nhật yaw tránh khuếch đại noise vị trí thành dao
> động góc; phép xoay quanh pose hiện tại tránh tạo bước nhảy lớn khi xe đã
> đi xa gốc `odom`.

## 3. GPS integrity state machine

Trạng thái: `GOOD / DEGRADED / LOST / RECOVERING`.

| Transition | Điều kiện |
|---|---|
| GOOD → DEGRADED | HDOP > 5, satellites < 4, NIS reject 3 lần liên tiếp hoặc message `fix_quality=0` đầu tiên |
| RECOVERING → DEGRADED | message `fix_quality=0` đầu tiên hoặc NIS reject 3 lần liên tiếp khi receiver vẫn có quality hợp lệ |
| DEGRADED/RECOVERING → LOST | đủ 3 message `fix_quality=0` liên tiếp |
| DEGRADED/RECOVERING → LOST | không có fix usable > 1,5 s dù receiver vẫn phát message |
| GOOD/DEGRADED/RECOVERING → LOST | không có message > 1,5 s |
| LOST → RECOVERING | có fix quality ≥ 1 và HDOP ≤ 5 và satellites ≥ 4 |
| DEGRADED → RECOVERING | `recovery_required` và có fix quality-good, hoặc 3 fix quality-good liên tiếp bị NIS thường reject |
| RECOVERING → GOOD | 5 fix liên tiếp qua quality gate và ít nhất 3 fix gần nhất qua recovery-NIS gate |
| DEGRADED → GOOD | 5 fix liên tiếp qua quality gate (HDOP ≤ 5, satellites ≥ 4) và NIS gate thường — hysteresis cùng cỡ RECOVERING → GOOD |

- State machine có `tick(current_timestamp)` gọi ở **mỗi odometry frame**.
  Timeout không phụ thuộc việc NMEA callback có được gọi hay không.
- State machine giữ `fix_zero_streak`: fix hợp lệ reset streak; quality-0
  đầu tiên cảnh báo `DEGRADED`, message thứ ba liên tiếp xác nhận `LOST`.
- Giữ riêng `last_message_time` và `last_usable_fix_time`. Fix usable phải
  qua quality gate và NIS gate đang áp dụng. Receiver vẫn nói nhưng không có
  fix usable trong 1,5 s vẫn phải `LOST`.
- Khi quality-0 làm rời `GOOD`, đặt `recovery_required=True`. Fix tốt trở
  lại từ `DEGRADED` dùng recovery-NIS ngay; không buộc qua NIS thường rồi
  mới được vào `RECOVERING`.
- Khi rời `GOOD`: latch pose global tin cậy cuối (`(x,y,θ)` + covariance +
  timestamp). Chỉ tạo latch mới nếu chưa ở một episode lỗi; không ghi đè
  latch trong `DEGRADED/LOST/RECOVERING`.
- LOST: 100% dead-reckoning trên local EKF; `T_map_odom` đóng băng.
- RECOVERING: quality gate chạy trước. NIS dùng `R_gps` inflate ×10,
  `σ_recovery ≥ 2 m` và threshold χ² 2 DOF, 99% = `9.210`; giảm dần về
  chuẩn trong 1–2 s. Một NIS reject không chuyển ngay về LOST. Khi
  measurement được nhận, dịch `T_map_odom` về target với correction-rate
  limit để chống jump.
- Recovery consensus buffer thuộc toàn episode và sống xuyên
  `DEGRADED ↔ RECOVERING → GOOD`; lên `GOOD` không được tự xóa buffer nếu
  fast-correction target chưa hình thành.
- Fix quality 1 vẫn được dùng cho NIS/update và recovery state, nhưng không
  được đưa vào fast-correction consensus. Chỉ quality ≥ 2 mới được phép kích
  hoạt correction nhanh; quality 1 có thể ổn định theo thời gian nhưng vẫn
  lệch tuyệt đối do multipath.
- Weight GPS trong DEGRADED: R scale theo HDOP nên tự giảm — không cần
  logic riêng.

> **Lý do:** dữ liệu thật có cả run quality-0 dài và message quality-0 lẻ.
> Debounce qua `DEGRADED` cảnh báo ngay nhưng tránh tạo LOST/re-lock episode
> giả. Khoảng receiver im lặng vẫn đi thẳng `LOST` bằng `tick`. Recovery gate
> riêng tránh tình trạng drift làm tất cả GPS mới bị NIS loại, trong khi
> correction-rate limit vẫn bảo vệ tính liên tục.
>
> **Lý do thêm `DEGRADED → GOOD`:** HDOP thực tế dao động (5,1 → 1,2) mà
> không mất fix — ví dụ xe chạy gần nhà cao tầng rồi ra khoảng trống. Thiếu
> đường này, state machine kẹt `DEGRADED` vĩnh viễn dù tín hiệu đã tốt lại,
> vì `LOST → RECOVERING → GOOD` chỉ đi được từ phía mất hẳn fix. Hysteresis
> 5 fix chống flapping khi HDOP nhấp nháy quanh ngưỡng 5. Chỉ run
> `fix_quality=0` đủ debounce hoặc timeout mới xác nhận `LOST`;
> HDOP/satellite xấu nhưng receiver vẫn phát fix ở `DEGRADED` trong grace
> period; nếu không có fix usable quá 1,5 s thì xác nhận `LOST`.

## 4. U-turn detector (B3, synthetic ở bước này)

- Heading θ từ local EKF, unwrap, cửa sổ trượt 8 s: `|θ(t) − θ(t−w)| ≥ 150°`.
- Latency = thời điểm phát hiện − thời điểm heading thật vượt 150° (GT từ
  synthetic trajectory). KPI ≤ 2 s.
- Synthetic: trajectory chữ U bán kính 3–8 m ở nhiều tốc độ + nhiễu proxy.
- **Bonus sau khi fusion đạt DoD:** chạy trên garage ramp và báo là
  `heading-reversal events`, không claim U-turn phố. Không dùng số đếm sự
  kiện chưa được benchmark tái lập trong KPI chính.

> **Lý do:** synthetic có ground truth rõ để kiểm latency; vòng/ramp garage
> không tương đương U-turn trên phố và không nên chặn B6/B8.

## 5. Benchmark (`benchmark_gps_fusion.py`)

Chạy trên garage_2 NMEA thật + ≥2 kịch bản sim (neighborhood cắt GPS giữa
đoạn thẳng; garage_3 degrade dần; office_loop degrade-hồi phục không mất fix
— bài test riêng cho đường `DEGRADED → GOOD` — nếu kịp giờ):

| Metric | Định nghĩa | Ngưỡng |
|---|---|---|
| Handover latency (B6) | t(state=LOST hoặc DEGRADED) − t(điều kiện thật xuất hiện trong NMEA/sim) | ≤ 2 s |
| Local continuity | `odom → base_link` không nhận correction GPS; không có discontinuity ngoài odometry input | bắt buộc |
| Correction-induced discontinuity | `‖Δpose_output − Δpose_predicted_from_odom‖` quanh transition | < 0,5 m/frame (ngưỡng tự đặt) |
| Drift trong dropout | ‖pose − reference‖ tại cuối đoạn LOST / quãng đường LOST | báo cáo, so KPI B1 5% |
| Re-lock speed guardrail | ‖pose − reference‖ sau khi GOOD trở lại 2 s | báo riêng, không thay B8 |
| Re-lock stable error (B8) | từ GPS accept đầu tiên sau LOST, tìm lần đầu error ≤ 5 m liên tục 1 s trong cửa sổ tối đa 10 s | `error_at_stable ≤ 5 m`; không đạt trong 10 s là fail |
| Re-lock convergence | thời gian từ RECOVERING đến error ổn định | báo cáo |
| NIS accept rate | tỷ lệ GPS update qua gate, theo trạng thái | log |

Xuất JSON + plot trajectory (map frame) đánh dấu các đoạn state — dùng lại
cho báo cáo và video demo.

## 6. Unit tests

1. Predict thẳng đều: covariance tăng, pose đúng kỳ vọng giải tích.
2. GPS update kéo `T_map_odom`/global pose về measurement nhưng không đổi
   local pose; NIS gate chặn outlier 50 m.
3. Transition đủ các đường theo bảng: quality-0 đầu tiên
   `GOOD/RECOVERING → DEGRADED`, message thứ ba liên tiếp → `LOST`, fix hợp
   lệ reset streak, timeout đi thẳng `LOST`, và `DEGRADED → GOOD` không flap
   khi HDOP nhấp nháy quanh 5.
4. Không deadlock: `DEGRADED` do quality-0 hoặc normal-NIS reject phải vào
   được recovery gate; chuỗi `0,4,0,4,...` với drift 20 m vẫn thu recovery
   candidates. Không có fix usable >1,5 s phải `LOST`.
5. Latch: pose latch bất biến trong suốt LOST.
6. No-jump: chuỗi LOST → RECOVERING → GOOD; local pose không bị correction
   và correction-induced discontinuity nhỏ hơn ngưỡng.
7. U-turn synthetic: phát hiện đúng, latency ≤ 2 s, không false positive
   trên trajectory thẳng + nhiễu.
8. `ODOM_proxy` seed cố định: cùng seed → cùng trajectory (tái lập).
9. `ODOM_proxy`: xe lùi giữ `v_meas < 0`, `dθ` wrap đúng qua ±π.
10. Recovery: innovation lớn hợp lý không làm hệ thống kẹt vĩnh viễn ở
   `LOST/RECOVERING`; outlier cực lớn vẫn bị reject.
11. Consensus buffer không bị xóa khi state lên `GOOD` sớm; các fix `GOOD`
    đầu tiên vẫn có thể hoàn tất fast-correction target.

## 7. Định nghĩa xong (Definition of Done)

- [ ] Tests trên pass toàn bộ.
- [ ] Benchmark chạy trên garage_2 NMEA thật ra JSON + plot.
- [ ] Handover ≤ 2 s và re-lock ≤ 5 m trên ít nhất kịch bản sim; số NMEA
      thật báo cáo as-is (không tune để đẹp).
- [ ] Interface odometry documented để Bước 2 swap VO không sửa EKF.
- [ ] Benchmark chỉ dùng measurement đến `current_time`; reference pose
      chỉ tạo proxy/simulator trước khi chạy và đánh giá output.
- [ ] Không dependency mới (numpy + stdlib; plot bằng OpenCV như Phần A).

## Ghi chú trung thực số liệu

Reference pose garage sinh từ stereo VIO + RTK của dataset; trong vùng mất
GNSS nó là VIO-dominated — dùng làm GT cho drift trong garage phải ghi rõ
giới hạn này (đã nêu ở PLAN Bước 6). Không tune ngưỡng state machine trên
cùng đoạn dùng để báo KPI: ngưỡng lấy từ đề (HDOP 5, sats 4) + χ² chuẩn,
không fit theo data.
