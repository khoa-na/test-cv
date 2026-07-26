# Bước 1 — Đề xuất sửa sau lần chạy benchmark đầu

Hiện trạng (seed 7, `artifacts/gps-fusion/benchmark.json`):

| Case | Handover | Re-lock (B8 ≤ 5 m) | Discontinuity (< 0,5 m) |
|---|---:|---:|---:|
| garage_3 sim | 0,030 s | 0,80 m ✅ | 0,198 m ✅ |
| neighborhood sim | 0,018 s | 0,93 m ✅ | 0,198 m ✅ |
| **garage_2 NMEA thật** | 0,0 s | **20,79 m ❌** | 0,198 m ✅ |

16/16 unit test Bước 1 pass (37/37 toàn repo). Sim đạt hết KPI; NMEA thật
trượt B8. Các sửa đổi dưới đây không đổi API odometry/GPS.

## Kết quả thẩm định đề xuất ban đầu

Đã thử trực tiếp trên cùng seed và benchmark, không chỉ suy luận:

- Debounce 3 message chỉ loại **3/10** episode quality-0; không thể giảm còn
  1–2 episode. Dropout cuối có 355 message quality-0 nên vẫn giữ nguyên.
- Công thức cũ
  `max(3, 1.5·sqrt(trace(P_map)))` cho rate khoảng 5,9 m/s khi trở lại
  `GOOD`: re-lock chỉ giảm từ 20,79 m xuống 14,85 m và discontinuity tăng
  lên 0,580 m — vẫn trượt cả B8 lẫn ngưỡng no-jump.
- Trim p90 rồi refit làm residual trên **toàn bộ** quality-4 xấu hơn:
  p95 7,06 → 7,95 m, RMSE 2,34 → 2,44 m. P95 chỉ đẹp nếu bỏ chính outlier
  khỏi tập báo cáo; không hợp lệ cho KPI chính.

Vì vậy Fix 1 được điều chỉnh, Fix 2 được thay thiết kế và Fix 3 chỉ còn là
diagnostic/calibration-only.

## Fix 1 — Debounce `fix_quality = 0` qua `DEGRADED`

**Bằng chứng:** 41 transitions trong 197 s. Receiver nhả message quality-0
lẻ tẻ khi xe qua ranh giới trong/ngoài nhà: chuỗi dropout 0,2–0,7 s
(`dropouts[0]`: 0,2 s; `[3]–[6]`: 0,2–0,4 s). Mỗi vòng LOST→GOOD ngắn đẻ
một relock event rác. Tuy nhiên debounce chỉ giảm flapping; nó không sửa
episode cuối đang quyết định `max_error_after_2s`.

**Sửa (`pipelines/localization_ekf.py`, `GPSIntegrityMonitor`):**

- Thêm `IntegrityConfig.fix_zero_debounce: int = 3`.
- Message `fix_quality = 0` đầu tiên chuyển `GOOD/RECOVERING → DEGRADED`
  và latch pose ngay khi rời `GOOD`; không được giữ nhãn `GOOD` khi receiver
  đã báo mất fix.
- Đủ 3 message quality-0 liên tiếp chuyển `DEGRADED/RECOVERING → LOST`.
- Một fix hợp lệ reset zero streak; đường `DEGRADED → GOOD` vẫn cần 5 fix
  tốt liên tiếp như state machine hiện tại.
- Timeout receiver-câm (`tick`) giữ nguyên — gap 517 s không bị ảnh hưởng.
- Handover sang trạng thái cảnh báo vẫn xảy ra ngay; latency sang `LOST`
  khoảng 0,2 s ở 10 Hz (message thứ nhất tại t0, thứ ba tại t0+0,2 s).

### Fix 1b — Đường thoát recovery từ `DEGRADED`

Debounce tạo một deadlock mới nếu recovery gate chỉ tồn tại trong state
`RECOVERING`: sau quality-0, máy ở `DEGRADED`; fix tốt quay lại nhưng drift
lớn làm NIS thường reject; vì không qua `LOST`, máy không bao giờ được dùng
recovery-NIS và kẹt `DEGRADED`.

Tách `recovery_required` khỏi tên state:

- Khi quality-0 làm rời `GOOD`, đặt `recovery_required=True`.
- `DEGRADED + quality-good + recovery_required` chuyển sang `RECOVERING`
  và measurement đó phải được xét bằng recovery-NIS (`R` inflate,
  threshold 9.210), không thử gate thường trước.
- Nếu `DEGRADED` bắt nguồn từ HDOP/satellite/NIS thường và có 3 fix
  quality-good liên tiếp bị **NIS thường** reject, cũng chuyển
  `DEGRADED → RECOVERING` và đặt `recovery_required=True`.
- Chỉ clear `recovery_required` khi recovery correction đã có consensus,
  correction gap đã xuống dưới ngưỡng kết thúc và integrity đã ổn định;
  không clear chỉ vì state vừa lên `GOOD`.

Như vậy chuỗi flapping `0,4,0,4,...` có thể dùng các fix quality-good qua
recovery gate để sửa global target, dù integrity vẫn ở
`DEGRADED/RECOVERING` cho tới khi có 5 fix tốt ổn định.

### Fix 1c — Timeout khi không có GPS usable

Receiver vẫn phát message không đồng nghĩa GPS còn dùng được. Thêm
`last_usable_fix_time` và:

- Một fix usable phải qua quality gate và gate NIS đang áp dụng
  (normal hoặc recovery).
- Nếu đang `DEGRADED/RECOVERING` và không có fix usable > 1,5 s, chuyển
  `LOST` với reason `no_usable_fix_timeout`.
- Timeout receiver-câm vẫn đo từ `last_message_time`; hai timer độc lập.
- Fix quality tốt nhưng consensus chưa đủ vẫn được tính là usable nếu qua
  recovery-NIS; consensus quyết định fast target, không quyết định receiver
  có sống hay không.

Điều kiện này thay cho rule mơ hồ “quality xấu > 0,5 s”: nó xử lý cả
HDOP/satellite xấu kéo dài, NIS reject kéo dài và flapping không tạo được
measurement dùng được, nhưng không ép một quality-0 lẻ thành `LOST`.

**Test thêm:**

- `[4,4,0]`: phải vào `DEGRADED`, latch được tạo.
- `[4,4,0,4]`: không vào `LOST`, zero streak reset.
- `[4,4,0,0,0]`: vào `LOST` tại message 0 thứ ba.
- Timeout vẫn vào `LOST` dù không có message thứ ba.
- `DEGRADED` do quality-0, sau đó fix tốt lệch 20 m: measurement dùng
  recovery-NIS, không bị gate thường khóa vĩnh viễn.
- `DEGRADED` với 3 fix quality-good bị normal-NIS reject:
  chuyển `RECOVERING`, không kẹt `DEGRADED`.
- Chuỗi `0,4,0,4,...` với drift 20 m: vẫn thu được recovery candidates và
  global error phải có khả năng hội tụ; integrity chỉ về `GOOD` sau 5 fix
  tốt liên tiếp.
- Message vẫn tới nhưng không có fix usable trong 1,5 s: vào `LOST` bằng
  `no_usable_fix_timeout`.

## Fix 2 — Recovery consensus + correction theo innovation có hard cap

**Bằng chứng:** dropout cuối 35,5 s / 165 m có target gap khoảng 18,8 m khi
trở lại `GOOD`. Rate 3 m/s không đóng gap trong 2 s. Nhưng tăng rate chỉ theo
`P_map` cũng không đủ và đã làm discontinuity vượt 0,5 m trong thử nghiệm.
`P_map` hiện còn hấp thụ một phần local covariance nên không phải đại lượng
thuần để quyết định tốc độ correction.

**Sửa (`FusionConfig` + `_advance_global_correction`):**

- Khi `recovery_required=True`, thu ít nhất 3 GPS update đã qua quality +
  recovery-NIS gate và có `fix_quality ≥ 2`. Buffer thuộc **recovery
  episode**, không thuộc riêng state `RECOVERING`. Fix quality 1 vẫn được
  dùng cho state machine và correction thường, nhưng không được kích hoạt
  fast correction vì dữ liệu garage cho thấy các cụm quality 1 có thể nhất
  quán mà vẫn lệch tuyệt đối do multipath.
- Buffer phải tiếp tục sống xuyên các transition
  `DEGRADED ↔ RECOVERING → GOOD`. Nếu integrity lên `GOOD` trước khi đủ
  consensus, các update `GOOD` đầu tiên vẫn tiếp tục bổ sung candidate và
  có thể bật fast correction muộn.
- Chỉ tạo fast-correction target khi các candidate translation của
  `T_map_odom` nhất quán trong bán kính cấu hình; dùng median để giảm một
  outlier đơn. Consensus chỉ áp cho translation vì GGA không đo heading.
- Nếu chưa có consensus thì không bật fast correction, nhưng không được
  clear buffer chỉ vì state đổi. Hết timeout recovery-correction cấu hình
  (khởi tạo 5 s) mà vẫn không có consensus: log failure và quay về correction
  thường; không tự chọn một candidate xấu.
- Sau khi có consensus, đặt `gap = ‖target_translation-current_translation‖`
  và:

  ```text
  rate = max(base_rate, gap / recovery_tau)
  step = min(rate · dt, max_correction_step)
  ```

  Giá trị khởi tạo: `base_rate=3 m/s`, `recovery_tau=1,0 s`,
  `max_correction_step=0,4 m/frame`.
- Hard cap 0,4 m/frame là invariant bảo vệ KPI no-jump. Với gap 20 m ở
  khoảng 30 FPS, cấu hình trên phải đưa residual xuống dưới 5 m trong
  cửa sổ 2 s; không yêu cầu residual về đúng 0.
- Khi gap nhỏ hoặc hết recovery, rate tự trở về `base_rate`; không dùng
  `map_covariance` làm rate trực tiếp. Covariance vẫn dùng đúng vai trò cho
  Kalman gain và NIS.
- Rotation limit giữ nguyên (heading không có nguồn đo trực tiếp từ GGA).

**Test thêm:**

- Ba recovery fix không nhất quán: không bật fast correction.
- Ba recovery fix nhất quán: target dùng median.
- Consensus chưa đủ khi `RECOVERING → GOOD`: buffer còn nguyên; các fix
  `GOOD` tiếp theo có thể hoàn tất consensus và bật fast correction.
- Flapping `DEGRADED ↔ RECOVERING` không được xóa recovery buffer.
- Gap 20 m ở 30 FPS xuống ≤ 5 m trong 2 s.
- Mọi frame có correction-induced discontinuity ≤ 0,4 m.
- Outlier 50 m vẫn bị NIS gate chặn và không đi vào consensus.
- Fix quality 1 được accept qua recovery-NIS nhưng không đi vào consensus;
  quality ≥ 2 mới có thể hoàn tất consensus.

> **Giới hạn claim:** correction nhanh chỉ giúp hệ thống hội tụ tới GPS target.
> Nếu GPS target hoặc alignment sai, nó có thể hội tụ nhanh tới vị trí sai.
> Vì vậy vẫn báo error thật và không coi Fix 2 là thay thế cho VO tốt.

## Fix 3 — Alignment diagnostic trên calibration split, không trim KPI

**Bằng chứng:** `alignment_error`: median 1,12 m, p95 7,06 m, RMSE 2,34 m
trên 758 điểm quality-4. Không được mặc định toàn bộ đuôi lỗi là do fit:
nó có thể gồm multipath, timestamp latency, antenna-camera lever arm hoặc
sai số reference. Thử trimmed fit không cải thiện residual held-out/toàn bộ.

**Sửa (`data_tools/gps_sources.py`, `load_nmea_replay`):**

- Ưu tiên kiểm lại transform chain do 4Seasons cung cấp và lever arm
  GPS↔IMU↔camera.
- Nếu vẫn dùng rigid fit fallback, chia một **calibration interval trước
  episode dropout được chấm**; không dùng reference pose của evaluation
  interval để chọn transform/outlier.
- Có thể thử Huber/trimmed fit chỉ trên calibration interval, nhưng chọn
  raw hay robust transform bằng held-out calibration, không bằng KPI B8.
- Luôn log ba nhóm riêng:
  - residual raw-fit trên toàn bộ evaluation points;
  - residual robust-fit trên toàn bộ evaluation points;
  - residual inlier-only, gắn nhãn diagnostic và không dùng làm p95 chính.
- Trước khi kết luận lỗi hệ tọa độ, scan một time offset nhỏ có giới hạn
  (ví dụ ±0,5 s) trên calibration interval và kiểm tra lever arm. Median tốt
  nhưng chỉ đuôi xấu không phải dấu hiệu điển hình của lỗi leap-second.

**Test thêm:**

- Synthetic 90% inlier + 10% outlier: robust fit phục hồi transform đúng.
- Calibration/evaluation split không chồng timestamp.
- Báo cáo all-point residual không được tự động bỏ outlier.
- Robust transform chỉ được dùng nếu held-out calibration không xấu hơn raw.

> **Không dùng Fix 3 để làm đẹp benchmark hiện tại:** thử nghiệm cho kết quả
> trimmed all-point p95 7,95 m và re-lock 15,07 m khi ghép adaptive-rate,
> đều không tốt hơn cách raw tương ứng. Bốn số lịch sử được giữ trong
> `artifacts/gps-fusion-trials/historical_trial_summary.json`; artifact
> tái chạy trên worktree hiện tại được tách riêng trong
> `benchmark_trials.json`.

## Vòng 2 — Sau khi implement Fix 1–2 và thẩm định Fix 3

43/43 test toàn repo pass. Kết quả
`artifacts/gps-fusion-after-fix/benchmark.json`: hai case sim đạt re-lock
0,56/0,85 m; jump chạm đúng hard cap 0,4 m; NMEA thật vẫn có
`max_error_after_2s = 21,4 m`.

Fix 3 hiện mới là diagnostic/thử nghiệm bị loại, chưa thay alignment
production. Không gọi Fix 3 là đã implement cho tới khi calibration split
và held-out validation chứng minh transform mới tốt hơn.

**Quan sát cần lưu thành artifact trước khi dùng làm claim:** cửa sổ re-lock
trượt chủ yếu chứa message quality-1/quality-0 và không có quality ≥ 2, nên
`min_fast_correction_quality=2` không tạo candidate
(`consensus_timeout`, `candidate_count=0`). Các thống kê theo cửa sổ
(số lượng message từng quality, error tại từng recovery event) phải được
xuất bởi benchmark/script; không chỉ ghi số phân tích thủ công trong tài
liệu.

### Fix 4 — Bổ sung metric steady có cửa sổ hữu hạn

Đề gốc chỉ ghi B8 “GPS re-lock drift correction: sai số ≤ 5 m”, không quy
định phải đạt sau 2 s. Mốc +2 s là guardrail tự đặt trong STEP1, vì vậy
benchmark nên giữ cả tốc độ hội tụ và độ chính xác sau hội tụ:

- `error_after_2s`: error tại `GOOD + 2 s`, giữ nguyên để báo tốc độ hội tụ;
- `time_to_stable_5m`: thời gian từ **mốc neo = GPS measurement đầu tiên
  được accept sau khi rời `LOST`** (định nghĩa một chỗ trong evaluator,
  không dùng mốc nội bộ của fusion) đến lần đầu error GT ≤ 5 m
  **liên tục ít nhất 1 s**;
- `error_at_stable`: error tại đầu khoảng ổn định nói trên;
- chỉ tìm trong cửa sổ tối đa 10 s. Không đạt ổn định trong 10 s thì ghi
  `time_to_stable_5m=null` và episode fail, không chờ vô hạn.

Không dùng thời điểm internal `consensus_achieved` hoặc
`episode_completed` làm ground truth: gap nội bộ đóng chỉ chứng minh output
đã đuổi kịp target, không chứng minh target đúng. B8 chỉ được claim đạt khi
`error_at_stable ≤ 5 m` trong cửa sổ 10 s; báo song song `error_after_2s`
để không che tốc độ hội tụ chậm.

**Test thêm:**

- Error đi xuống dưới 5 m trong một frame rồi bật lên: chưa được tính stable.
- Error ≤ 5 m liên tục 1 s: trả đúng `time_to_stable_5m`.
- Không ổn định trong 10 s: trả `null` và fail.
- Metric chỉ dùng reference offline trong evaluator, không rò GT vào fusion.

### Fix 5 — Phân loại kết quả recovery khi consensus timeout

Code hiện tại cho phép `consensus_timeout` + gap nội bộ ≤ 1 m hoàn tất
episode. Đây không hẳn là “complete giả”: các GPS quality-1 đã qua
recovery-NIS vẫn cập nhật `target_map_to_odom` bằng Kalman update, và
correction thường có thể đuổi kịp target đó. Vấn đề thật là event đang được
gọi chung `episode_completed` dù mức tin cậy khác nhau; gap nội bộ nhỏ cũng
không bảo đảm error GT nhỏ.

**Sửa semantics và logging:**

- Nếu có consensus quality ≥ 2 và gap ≤ 1 m:
  `outcome=consensus_settled`, high-confidence.
- Nếu consensus timeout nhưng correction thường đã đưa gap ≤ 1 m và
  integrity đang `GOOD`: `outcome=timeout_settled_low_confidence`.
- Nếu hết cửa sổ đánh giá mà gap chưa đóng:
  `outcome=unconverged`.
- Chỉ `consensus_settled` được gọi là fast-recovery success. Hai outcome còn
  lại vẫn được evaluator chấm độc lập bằng GT; không dùng outcome nội bộ để
  tự tuyên bố B8 đạt.
- Với `timeout_settled_low_confidence`, lưu latch/covariance cuối vào log rồi
  đóng episode, clear latch và quay về normal gate. Không giữ
  `recovery_required=True` vô hạn vì sẽ tạo một deadlock/liveness issue mới.
- Với `unconverged`, dùng **cờ confidence riêng** (`global_confidence=low`),
  không ép integrity state về `DEGRADED`: state machine mô tả chất lượng
  receiver, còn unconverged là chuyện của global estimate — GPS có thể
  genuinely GOOD trong khi correction chưa đóng gap. Trộn hai thứ vào một
  state sẽ phá điều kiện transition dựa trên quality/NIS.

**Test thêm:**

- Timeout không consensus nhưng gap đã đóng: outcome phải là
  `timeout_settled_low_confidence`, không phải high-confidence success.
- Timeout và gap chưa đóng: không clear episode như một recovery thành công.
- Cả ba outcome phải có timestamp, gap, consensus flag và error GT chỉ do
  benchmark bổ sung.
- Low-confidence completion không làm hệ thống kẹt recovery gate mãi.

### Fix 6 — Không bật fast correction từ quality-1 theo mặc định

**Không triển khai tier quality-1 fallback vào production ở Bước 1.**
Năm candidate quality-1 nằm trong radius 2 m chỉ chứng minh chúng nhất quán,
không chứng minh chúng đúng tuyệt đối; multipath có thể tạo cả cụm ổn định
nhưng lệch lớn.

Quan trọng hơn, khi consensus hoàn tất, implementation hiện đặt median
candidate trực tiếp vào `target_map_to_odom`. Vì vậy lập luận “R inflate giữ
Kalman weight thấp” không còn bảo vệ bước fast correction đó. NIS nới cũng
có thể accept bias lớn khi covariance đã phình sau dropout. Quality ≥ 2 xuất
hiện sau này có thể sửa target, nhưng output đã có thể bị kéo nhanh về vị
trí sai trước đó.

Giữ policy production:

- quality-1 vẫn qua quality/NIS gate, cập nhật target bằng Kalman update và
  correction thường có rate limit;
- quality ≥ 2 mới được tạo fast-correction consensus;
- buffer consensus sống qua `DEGRADED ↔ RECOVERING → GOOD`, nhưng reset khi
  `LOST` đã được xác nhận. Không trộn candidate qua hai episode mất GPS;
- cửa sổ chỉ có quality-1 mà không đạt B8 được báo failure analysis, không
  hạ gate sau khi nhìn thấy KPI.

Nếu còn thời gian, quality-1 fallback chỉ được chạy như **A/B experiment**
trong artifact riêng. Chỉ cân nhắc đưa vào production nếu đồng thời:

- không làm xấu max/p95 error trên tất cả episode quality-1;
- không vi phạm discontinuity < 0,5 m/frame;
- có test multipath bias nhất quán, không chỉ test noise tản mát;
- không đặt median trực tiếp làm target mà bỏ qua confidence/Kalman weight.

## Kết quả Vòng 3 — Fix 4–5

49/49 test toàn repo pass. Artifact:
`artifacts/gps-fusion-round3/benchmark.json`.

| Case | `max_error_after_2s` | Stable trong 10 s | `max_time_to_stable_5m` | `max_error_at_stable` |
|---|---:|---:|---:|---:|
| garage_3 sim | 0,56 m | ✅ | 0,011 s | 1,51 m |
| neighborhood sim | 0,85 m | ✅ | 0,636 s | 4,92 m |
| **garage_2 NMEA thật** | **21,44 m** | **❌ (0/6 episode)** | — | — |

Benchmark vòng 3 giữ nguyên kết luận kỹ thuật: logic recovery và hai case
kiểm soát đạt, nhưng không claim B8 trên NMEA thật. Sáu cửa sổ NMEA thật đã
có `gps_quality_counts_10s`; năm cửa sổ chỉ có quality-1/0, cửa sổ cuối mới
có quality 2/4/5 nhưng error vẫn không ổn định dưới 5 m. Điều này củng cố
việc không bật quality-1 fast fallback và để stereo VO ở Bước 2 xử lý nguồn
drift chính.

`recovery_events` nay có `reference_error_m` do evaluator bổ sung và outcome
được tách:

- `consensus_settled` và `timeout_settled_low_confidence` là event terminal;
- `unconverged` là checkpoint không terminal ở mốc 10 s: episode vẫn mở và
  có thể settle muộn, nhưng B8 của cửa sổ hữu hạn vẫn fail.

## Không sửa (ghi nhận, để Bước 2 giải quyết)

- **Drift proxy 15–17%/dropout** vượt xa mục tiêu thiết kế 1–3%: nghi heading
  từ camera +Z nhiễu trên ramp dốc (pitch lớn — garage là bài 3D, proxy 2D).
  Không sửa vì proxy sẽ bị thay bằng stereo VO (Bước 2). Báo cáo phải ghi rõ:
  **số drift NMEA-thật hiện tại đo chất lượng proxy, không phải chất lượng
  fusion**; số fusion "sạch" là 2 case sim với drift được kiểm soát.
- Median trajectory error garage_2 là 14,7 m ở baseline và khoảng 7,88 m
  sau Fix 1–2; cả hai vẫn chịu ảnh hưởng của cùng odometry proxy — đánh giá
  lại sau khi có stereo VO thật.

## Đánh giá lại sau Bước 2 — Fusion vòng 4

Stereo VO + IMU-yaw đã thay proxy trên garage_2 thật
(`artifacts/gps-fusion-round4-vo/`):

- median trajectory error giảm nhẹ 7,875 → 7,484 m;
- episode RTK cuối giảm 19,090 → 14,804 m;
- correction jump giảm 0,400 → 0,332 m;
- nhưng max error sau 2 s vẫn 21,557 m và 0/6 episode stable ≤5 m trong 10 s.

Hai case simulated dùng cùng VO đều `all_stable_within_10s=true`
(`artifacts/gps-fusion-round4-vo-simulated/`). Vì vậy giả thuyết “proxy là
blocker chính của B8 thật” bị bác bỏ: VO tốt hơn giúp một phần nhưng không đủ;
NMEA bias/alignment evaluation là blocker còn lại. Không claim B8 real.

## Đánh giá Fix 3 — transform chain, lever arm, time offset

Artifact: `artifacts/gps-alignment-fix3/` và
`artifacts/gps-fusion-round5-vo/`.

- Chronological calibration/holdout split = 454/304 quality-4 sample. Holdout
  không tham gia fit hoặc chọn phương án.
- Chain chính thức của `libartipy` đã được đối chiếu convention và triển khai:
  WGS84 → ECEF → GPS-world/ENU → world → metric SLAM. GGA dùng ellipsoid
  altitude = MSL altitude + geoid separation, thay vì bỏ altitude.
- Rigid calibration-only: median 0,790 m calibration, 1,534 m holdout.
  Transform chain: 0,607 m calibration, 2,120 m holdout; clock offset chọn
  trên calibration là +0,10 s.
- GPS–IMU lever arm trong file là 0 m; camera–IMU là 0,185 m. Ablation cần
  orientation reference, không cải thiện đáng kể và không đủ điều kiện đưa
  vào runtime.
- Vòng 5 với transform chain giảm max error sau 2 s 21,557 → 20,760 m nhưng
  median trajectory tăng 7,484 → 8,943 m và B8 vẫn fail. Do đó Fix 3 được
  **điều tra đầy đủ nhưng bị bác làm alignment chính**; không thay rigid
  baseline chỉ vì calibration median đẹp hơn.

## Thứ tự thực hiện và tiêu chí nghiệm thu lại

1. Fix 1/1b/1c → debounce, đường `DEGRADED → RECOVERING` và hai timeout
   phải pass trước. Chạy riêng test deadlock flapping + drift 20 m.
2. Thêm recovery-episode consensus của Fix 2; kiểm buffer xuyên state và
   outlier/NIS trước.
3. Thêm bounded innovation correction; nghiệm thu đồng thời re-lock và
   discontinuity, không chấp nhận đổi một KPI lấy KPI khác.
4. Điều tra transform/time/lever arm theo Fix 3 trên calibration split.
   Chỉ thay alignment chính nếu holdout và fusion KPI chứng minh tốt hơn.
   **Đã làm:** chain không qua điều kiện này, nên giữ rigid baseline.
5. Chạy lại sim trước để chống regression, sau đó NMEA thật. Lưu kết quả
   baseline và after-fix ở hai thư mục khác nhau, không overwrite đối chứng.
6. Vòng 2: Fix 4 trước để có metric GT hữu hạn và tái lập được; sau đó Fix 5
   để tách outcome high/low-confidence. Không triển khai Fix 6 mặc định;
   nếu còn thời gian chỉ chạy A/B experiment trong artifact riêng. Baseline
   vòng 2 là `artifacts/gps-fusion-after-fix/`, kết quả vòng 3 ghi thư mục
   mới.

KPI sau fix: sim giữ nguyên đạt; NMEA thật kỳ vọng handover/discontinuity
giữ đạt. Re-lock có thể cải thiện nhưng phải báo as-is; **chỉ chốt số B8
chính thức sau khi stereo VO thay proxy** (Bước 2).
