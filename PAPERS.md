# Papers cho Phần B

Đánh giá tài liệu tham khảo dưới ràng buộc của bài: CPU-only, ONNX Runtime,
end-to-end ≥ 15 FPS (B7). Chia hai nhóm: paper đề gợi ý (đa số chỉ để trích
dẫn) và paper tự tìm (dùng để triển khai).

## Nhóm 1 — Paper đề gợi ý ([7]–[12])

| # | Paper | Nội dung | Kết luận |
|---|---|---|---|
| [7] | [NetVLAD (CVPR 2016)](https://arxiv.org/abs/1511.07247) | Lớp VLAD học được trên CNN, global descriptor cho place recognition | Trích dẫn làm nền tảng VPR. Không chạy: bài B2 có pose prior từ EKF nên không cần search toàn cục |
| [8] | [EigenPlaces (ICCV 2023)](https://arxiv.org/abs/2308.10832) | VPR robust viewpoint, descriptor nhỏ hơn 50%, có bản ResNet-18, [code](https://github.com/gmberton/EigenPlaces) | Ứng viên duy nhất nhóm VPR chạy nổi CPU (keyframe rate 1–2 Hz). Chỉ cần nếu làm relocalization toàn cục — backup, chưa dùng |
| [9] | [AnyLoc (2024)](https://arxiv.org/abs/2308.00688) | DINOv2 + VLAD không cần train, tổng quát nhất | Loại: ViT trên CPU ~3 FPS (đã đo với Depth Anything ViT-S 518px trong BENCHMARK.md), giết KPI B7 |
| [10] | [ORB-SLAM3 (2021)](https://arxiv.org/abs/2007.11898) | Visual + visual-inertial + multi-map SLAM, mono/stereo/RGB-D, real-time CPU | **Paper duy nhất trong list đáng làm theo.** Chứng minh ORB tracking real-time CPU. Cắt loop closing + multi-map, giữ tracking là thành VO cho B1 |
| [11] | [DPVO (NeurIPS 2023)](https://arxiv.org/abs/2208.04726) | Learned patch tracking + differentiable BA, hơn DROID-SLAM và nhanh gấp 3 | Loại: "nhanh" là so với DROID trên GPU; recurrent net + BA lặp không export ONNX CPU 15 FPS được. Trích làm SOTA đã cân nhắc |
| [12] | [CLRKDNet (2024)](https://arxiv.org/abs/2405.12503) | Distill CLRNet, giảm 60% inference time, SOTA CULane/TuSimple | Loại: giải bài khó hơn bài đề hỏi. B4 chỉ cần phân loại làn trái/phải — IPM + histogram vạch kẻ OpenCV đủ. Giữ làm phương án nếu cần robust đêm/mưa |

Nhận xét chung: list của đề là bẫy scope. Lời giải đề mong đợi nằm trong thân
đề (ghost projection + semantic bounding box + EKF) — toàn thứ viết được bằng
OpenCV/NumPy. Giá trị chính của nhóm này là nguyên liệu cho mục "phương án đã
cân nhắc và loại" (ăn điểm C3/C4 + failure analysis).

## Nhóm 2 — Paper tự tìm, dùng để triển khai

### B1: Scale cho monocular VO

| Paper | Dùng gì |
|---|---|
| [Song & Chandraker, "Robust Scale Estimation in Real-Time Monocular SFM for Autonomous Driving" (CVPR 2014)](https://openaccess.thecvf.com/content_cvpr_2014/html/Song_Robust_Scale_Estimation_2014_CVPR_paper.html) + [bản journal TPAMI 2015](https://dl.acm.org/doi/abs/10.1109/TPAMI.2015.2469274) | Blueprint chính cho B1: mono VO real-time, scale từ chiều cao camera + ground plane, tiệm cận stereo trên KITTI. Scale correction chỉ trigger ~1 lần/100 frame nên gần như không tốn FPS. Phần failure modes khi mặt đường xấu khớp đúng bối cảnh ổ gà |
| ["Accurate and Robust Scale Recovery for Monocular VO Based on Plane Geometry" (arXiv 2101.05995)](https://arxiv.org/pdf/2101.05995) | Bản 2021 nhẹ hơn: RANSAC fit ground plane từ feature vùng đường, recover scale mỗi frame, real-time CPU. Dễ implement hơn Song (không cần cue-combination học). Tái dùng trực tiếp kỹ thuật fit mặt đường + RANSAC đã có từ Phần A. **Đọc trước tiên** |

### B2 + B8: Landmark correction trong EKF

| Paper | Dùng gì |
|---|---|
| [Qu, Soheilian, Paparoditis, "Vehicle localization using mono-camera and geo-referenced traffic signs" (IV 2015)](https://www.researchgate.net/publication/308864003_Vehicle_localization_using_mono-camera_and_geo-referenced_traffic_signs) | Chính là "ghost projection" đề gợi ý, có trước đề ~10 năm: EKF dự đoán pose, chiếu biển báo geo-referenced vào ảnh, detect biển thật, reprojection error làm EKF update. Lấy sẵn measurement model + Jacobian cho Module 7 |
| ["Autonomous vehicle localization based on EKF and geo-referenced landmarks"](https://www.researchgate.net/publication/360380300_Autonomous_vehicle_localization_method_based_on_an_extended_Kalman_filter_and_geo-referenced_landmarks) · ["Utilizing semantic visual landmarks for precise vehicle navigation"](https://www.researchgate.net/publication/323786432_Utilizing_semantic_visual_landmarks_for_precise_vehicle_navigation) | Bearing-only landmark update với map 2D sparse. Backup nếu không ước lượng được khoảng cách tới landmark — bearing-only chỉ cần góc, không cần depth |

### Kiến trúc + luận điểm CPU

| Paper | Dùng gì |
|---|---|
| [LEVIO (2026, ETH)](https://arxiv.org/abs/2602.03294) + [code](https://github.com/ETH-PBL/levio) | VIO ORB-based real-time trên SoC siêu yếu, 110 KB RAM. Bằng chứng cho luận điểm "VIO không cần GPU" (điểm C4). Python reference model dùng tham khảo cấu trúc pipeline |
| MSCKF ([msckf_vio](https://github.com/KumarRobotics/msckf_vio)) | Kiến trúc filter đúng nếu cần fuse IMU chặt. Chưa cần cho KPI drift 5%/500 m — EKF loosely-coupled + scale ground plane đủ. Phương án nâng cấp nếu đo thấy thiếu |

## Nhóm 3 — Paper bổ sung cho stereo, association và GPS integrity

Nhóm này lấp các khoảng trống thực thi của nhóm 2: cách tạo metric VO trực
tiếp từ stereo, gán độ tin cậy cho VO, chống nhận nhầm landmark theo chuỗi và
loại GPS outlier trước khi EKF update.

### B1: Stereo visual odometry nhẹ trên CPU

| Paper / nguồn | Dùng gì |
|---|---|
| [LIBVISO2 — Geiger et al.](https://www.cvlibs.net/software/libviso/) | Blueprint chính nếu Phần B dùng lại stereo rig của Phần A: sparse stereo matching → triangulate 3D → temporal matching → tối thiểu hoá reprojection error. Chạy CPU, không cần loop closure/mapping và không có scale ambiguity. Có thể tự viết bản rút gọn bằng `solvePnPRansac` + pose refinement của OpenCV thay vì tích hợp cả C++ library |
| [Lightweight Visual Odometry for Autonomous Mobile Robots (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6165120/) | Thiết kế stereo/RGB-D VO low-overhead và benchmark với LIBVISO2 trên KITTI, EuRoC, TUM RGB-D. Dùng làm bằng chứng rằng frame-to-frame VO nhẹ phù hợp hơn full SLAM khi chỉ cần dead reckoning B1 |
| [Joint Forward-Backward Visual Odometry for Stereo Cameras (2019)](https://arxiv.org/abs/1912.10293) | Tính pose thuận/nghịch để có reliability measure không cần GT online. Bản triển khai tối giản dùng sai số `Log(T_forward @ T_backward)` cùng PnP inlier ratio và reprojection RMS để scale `R_vo` hoặc reject VO update |

Stereo là đường chính nếu đã có cặp camera rectify và calibration. Hai paper
monocular scale ở nhóm 2 chuyển thành fallback khi deployment chỉ có một
camera; không nên tự tạo bài toán scale ambiguity khi stereo đã sẵn có.

**Lưu ý:** nhánh stereo/mono chưa chốt — nó phụ thuộc dataset open được chọn
(không còn thời gian tự thu data). Bộ stereo Fan của Phần A là ảnh tĩnh,
không chạy VO được. KITTI odometry, 4Seasons có stereo sequence; nhiều bộ
khác chỉ mono. Chọn dataset xong mới chốt nhánh.

### B2: Landmark map và sequence-based re-identification

| Paper | Dùng gì |
|---|---|
| [Zhuo, Fu, Xue, "Monocular Visual Localization for Autonomous Vehicles Based on Lightweight Landmark Map" (SAE 2022)](https://doi.org/10.4271/2022-01-7094) | Lưu stop-line landmark bằng Bag-of-Words + vị trí map rồi dùng lần tái quan sát để sửa drift. Áp dụng trực tiếp cho pothole landmark: `class + ENU position + depth + area + descriptor`; map thưa vẫn đủ reset drift trên tuyến cố định. *Paywall SAE — nếu không lấy được full text thì chỉ trích dẫn, không dựa chi tiết triển khai* |
| [Garg & Milford, "Fast, Compact and Highly Scalable Visual Place Recognition through Sequence-based Matching of Overloaded Representations" (2020)](https://arxiv.org/abs/2001.08434) | Không accept match từ một frame đơn. Cộng similarity trên chuỗi 5–10 keyframe để khử perceptual aliasing trong garage/hầm; thử cả thứ tự xuôi và ngược để hỗ trợ U-turn. Dùng sau spatial gate của EKF và trước ORB/RANSAC geometric verification |
| [ConvSequential-SLAM (2020)](https://arxiv.org/abs/2009.13454) | VPR training-free kết hợp sequence matching với descriptor chịu viewpoint change. Chỉ lấy ý tưởng sequence consistency; không cần bê toàn bộ pipeline nếu ORB/EigenPlaces descriptor hiện tại đã đủ |

Pose prior của EKF chỉ thu hẹp candidate set, không thay thế data association.
Landmark update chỉ được phép chạy sau chuỗi gate:

`spatial → class/geometry → descriptor top-K → sequence consistency → geometric verification`.

### B4: Lane detection nhẹ

| Paper | Dùng gì |
|---|---|
| [Qin, Wang, Li, "Ultra Fast Structure-aware Deep Lane Detection" (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690273.pdf) + [code](https://github.com/cfzd/Ultra-Fast-Lane-Detection) | Row-anchor classification thay pixel segmentation, có bản ResNet-18 và postprocess nhẹ. Phương án ONNX fallback nếu IPM + histogram không đạt B4; chỉ dùng pretrained model, không train lại trong deadline |
| [Ultra Fast Deep Lane Detection v2 (TPAMI 2022)](https://doi.org/10.1109/TPAMI.2022.3182097) | Hybrid anchors cải thiện lane shape/viewpoint so với bản 2020. Dùng làm hướng nâng cấp; không ưu tiên hơn UFLD v1 cho MVP vì export/postprocess phức tạp hơn |

MVP vẫn bắt đầu bằng IPM + histogram OpenCV. Chỉ chuyển sang UFLD nếu có thể
export ONNX và benchmark CPU trong vài giờ. Cả hai đường phải có trạng thái
`unknown` khi thiếu vạch hoặc confidence thấp; không ép `left/right`.

### B6 + B8: GPS integrity, handover và re-lock

| Paper / nguồn | Dùng gì |
|---|---|
| [Miao et al., "Extended Robust Kalman Filter Based on Innovation Chi-Square Test" (2016)](https://doi.org/10.13203/j.whugis20130666) | Thêm normalized innovation squared (NIS) trước GPS update để loại multipath/outlier mà HDOP và số vệ tinh không phát hiện. GPS 2D dùng gate χ² 95% = 5.991 hoặc 99% = 9.210. *Journal tiếng Trung — kỹ thuật NIS gating là chuẩn textbook (Bar-Shalom), có thể trích thay bằng nguồn khác nếu cần* |
| [Adham et al., "Adaptive VINS-GNSS Fusion with Intelligent Switching and Failure Detection" (2025)](https://doi.org/10.1109/ICEENG64546.2025.11031382) | Mở rộng state machine theo confidence của cả GPS và VO: `fusion`, `GPS-dominant`, `VO-only`, `degraded`. Ý tưởng dùng ngay; không triển khai backend graph optimization hay dynamic-object network của paper. *IEEE paywall — vai trò trích dẫn* |
| [A Multi-Sensor Fusion MAV State Estimation from Long-Range Stereo, IMU, GPS and Barometric Sensors (2017)](https://doi.org/10.3390/s17010011) | Cơ sở cho loosely-coupled fusion giữa absolute measurements (GPS/landmark) và relative measurements (stereo VO/IMU) trong EKF khi GPS intermittent/lost. Lấy kiến trúc measurement separation, không cần stochastic-cloning EKF đầy đủ cho state 2D |
| [Nav2 GPS localization / `robot_localization`](https://docs.nav2.org/tutorials/docs/navigation2_with_gps.html) | Không phải paper nhưng là reference triển khai ROS 2: local EKF phát `odom → base_link`; global EKF nhận GPS ENU và phát `map → odom`. Giữ local trajectory liên tục trong lúc global correction thay đổi |

GPS acceptance không chỉ dựa trên ngưỡng đề. Measurement update tối thiểu:

```text
quality_good = hdop <= 5 and satellites >= 4 and snr_ok
innovation_good = (z - Hx)' inv(H P H' + R) (z - Hx) <= chi2_gate
accept_gps = quality_good and innovation_good
```

Khi re-lock, yêu cầu 3–5 fix liên tiếp qua cả hai gate, bắt đầu với `R_gps`
lớn rồi giảm dần trong 0,5–2 giây. Không sửa trực tiếp local odometry; global
correction đi qua `map → odom`.

## Thứ tự đọc đề xuất

Nhánh do dataset quyết định (xem lưu ý ở mục stereo VO). Nếu dùng stereo:

1. LIBVISO2 — frontend metric VO tối thiểu.
2. Joint Forward-Backward VO — cách sinh confidence/covariance cho VO.
3. Qu IV 2015 — measurement model cho EKF landmark update.
4. Robust EKF innovation χ² — GPS outlier gate và re-lock.
5. Lightweight Landmark Map + sequence VPR — association và drift correction.

Nếu chỉ có monocular:

1. arXiv 2101.05995 (plane-geometry scale) — mỏng, code được ngay.
2. Qu IV 2015 — measurement model cho EKF landmark update.
3. Song CVPR 2014 / TPAMI 2015 — failure modes của ground-plane scale trên
   đường xấu.
4. ORB-SLAM3 — phần tracking + IMU initialization, để biết cắt gì.

## Stack chốt cho Phần B

**Đường stereo ưu tiên:** LIBVISO2-style sparse stereo VO + forward/backward
confidence → local EKF 2D `(x, y, θ, v, yaw_rate)` → global EKF loosely
coupled GPS/landmark → landmark update kiểu ghost projection semantic bbox
(Qu 2015) với sequence association → lane trái/phải bằng IPM + histogram
OpenCV → GPS integrity state machine dùng cả HDOP/satellite/SNR và NIS χ².

**Fallback monocular:** ORB VO + scale từ ground plane/chiều cao camera, sau đó
dùng cùng fusion stack. Ground-plane scale chỉ update khi RANSAC support và
plane residual đạt quality gate.

Giữ `odom → base_link` liên tục; GPS và landmark chỉ sửa `map → odom`. Không
thêm model nặng mới ngoài detector biển báo ONNX nhỏ; UFLD/EigenPlaces chỉ
được bật ở keyframe rate nếu baseline không đạt B2/B4.
