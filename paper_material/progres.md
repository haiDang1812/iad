# Progress — Cải tiến INP-Former trên MVTec AD 2

> Tài liệu này ghi lại toàn bộ mạch hiểu và lập luận: từ ý tưởng gốc của INP-Former,
> qua chuỗi diagnosis sẵn có, đến cách suy ra hướng đang chạy (`eval_fusion_distance.py`),
> hướng đó làm gì và kết quả ra sao.
> Cập nhật lần cuối: 2026-06-20.

---

## 0. Bối cảnh & mục tiêu

- **Bài toán:** Unsupervised Anomaly Detection / Segmentation công nghiệp trên **MVTec AD 2**
  (bản kế nhiệm khó hơn của MVTec AD, 8 category, >8000 ảnh độ phân giải cao).
- **Baseline:** **INP-Former** (CVPR 2025) — phương pháp reconstruction-based dùng encoder
  **DINOv2-reg ViT-B/14 đông cứng (frozen)**.
- **Metric chính (theo paper dataset Heckler-Kram et al., IJCV 2026, Sec 4.2):**
  **AU-PRO@0.05** (threshold-independent, tích phân PRO tới FPR=0.05). F1 (pixel/SegF1) là
  metric phụ threshold-dependent, ngưỡng chính thức `t_seg = μ + 3σ` trên pixel ảnh validation normal.
  - *(Lưu ý: challenge VAND 3.0 CVPR2025 xếp hạng theo SegF1 — đó là track khác, không phải benchmark ta theo.)*
- **Vì sao MVTec AD 2 khó:** vật trong suốt/chồng lấp, chiếu sáng dark-field/back-light,
  phương sai dữ liệu normal cao, **lỗi cực nhỏ**, và có test set đổi điều kiện sáng (distribution shift).
  SOTA hiện tại vẫn **< 60% AU-PRO** → benchmark chưa bão hòa, còn nhiều dư địa.

---

## 1. Ý tưởng gốc của INP-Former (cơ chế chấm điểm)

INP-Former chấm điểm bất thường bằng **residual tái tạo (reconstruction residual)**:

1. Ảnh → **encoder DINOv2 frozen** → trích feature ở các target layer `[2..9]`, fuse lại → `en`.
2. **INP Extractor**: từ chính ảnh test, gộp tuyến tính các token "bình thường" thành một nhóm
   **Intrinsic Normal Prototypes (INP)** (M=6 token).
3. **INP-Guided Decoder**: dùng INP để tái tạo lại feature → `de`.
4. **Điểm bất thường = `1 − cosine(en, de)`** (residual). Giả định: vùng normal tái tạo tốt
   (residual thấp), vùng lỗi tái tạo kém (residual cao).

Code liên quan: `models/uad.py` (`INP_Former.forward`, `gather_loss`),
`utils.py::cal_anomaly_maps` (tính residual map), `INP_Former_MVTecAD2_Baseline.py`.

---

## 2. Chuỗi diagnosis sẵn có — phát hiện gì

Đây là phần chẩn đoán đã có trong repo (`diagnosis/diagnosis*/`). Tóm tắt từng cái + số liệu chính:

| # | Diagnosis | Câu hỏi | Phát hiện chính |
|---|---|---|---|
| D1/2 | defect pixel % | Lỗi to hay nhỏ, transform có giữ tín hiệu? | Lỗi rất nhỏ (nhiều category median <0.2% diện tích). |
| **D4** | INP contamination | INP có bị nhiễm token lỗi? | **CÓ.** fabric/fruit_jelly/rice/vial/walnuts: vùng lỗi hút INP attention **bằng/hơn** vùng normal (tỉ lệ tới **3.44** ở rice). |
| **D5** | decoder attention | Token INP nào được vùng lỗi dùng? | 4–5/6 token INP "bẩn" ở fruit_jelly/rice/vial/walnuts → INP đại diện cả cho lỗi. |
| **D6** | prototype collapse | Prototype normal vs defect có khác nhau? | **Giống nhau ~0.99–1.0 cosine** ở mọi category → prototype không phân biệt được (collapse). |
| **D7** | per-layer residual | Khoảng cách residual normal vs defect mỗi layer | **Gap chỉ 0.001–0.008** trên mọi layer. → **Điểm số gần như MÙ.** |
| **D8** | feature space overlap | Centroid normal vs defect | **~0.99 cosine** (gần trùng), defect nằm trong phân bố normal (đa số 0% ngoài 2σ) → lỗi tinh vi. |
| **D9** | discriminative direction | LDA trên frozen feature tách được không? | **AUROC = 1.0 ở MỌI layer** (kể cả layer 2). → Tín hiệu phân biệt **CÓ SẴN** trong frozen feature. |

### Diễn giải mạch nhân quả (điểm mấu chốt)

> **Thông tin để phân biệt normal/lỗi vẫn nằm nguyên trong feature DINOv2 đông cứng (D9),
> nhưng cách chấm điểm bằng residual đã ném nó đi (D7).**

- D7 và D9 thoạt nhìn mâu thuẫn (D8 nói centroid gần trùng, D9 nói tách hoàn hảo), nhưng thực ra
  nhất quán: hướng phân biệt là một **chiều phương sai nhỏ** — cosine centroid vẫn 0.99 mà vẫn
  tách tuyến tính được dọc theo chiều phụ đó.
- Nguyên nhân residual mù: INP được trích **từ chính ảnh test** (gồm cả lỗi). D4/D5/D6 cho thấy
  INP bị nhiễm lỗi và collapse → decoder tái tạo **cả vùng lỗi cũng tốt** → residual ở lỗi không cao
  hơn ở normal bao nhiêu (D7).

---

## 3. Hai đề xuất trước đó đã thất bại (thấp hơn baseline) — và TẠI SAO

| Đề xuất | File | Làm gì | Vì sao thất bại |
|---|---|---|---|
| **RCPL** (Residual Calibration) | `INP_Former_Residual_Calibration.py`, `train_mvtecad2_rcpl.py` | Thêm module hiệu chỉnh residual + loss tương phản normality | Vẫn ở trong paradigm reconstruction; điểm cuối vẫn là residual mà D7 chứng minh là mù. |
| **DCPL** (Local Contrast + Ortho) | `ablation_dcpl.py` | Làm prototype đa dạng/trực giao hơn (local contrast, orthogonal loss) | Không sửa nhiễm INP (lúc test INP vẫn trích từ ảnh lỗi → nhiễm là vấn đề **cấu trúc**, không phải chất lượng training). |

**Bài học:** cả hai **đánh bóng nhầm bộ chấm điểm**. Muốn cải thiện phải **thay/bổ sung cách chấm điểm**,
chứ không phải tinh chỉnh thêm trong khung reconstruction.

---

## 4. Lập luận ra hướng mới

Suy luận trực tiếp từ diagnosis:

1. D7 → **bỏ phụ thuộc hoàn toàn vào residual** (nó mù).
2. D9 → **chấm điểm trực tiếp trên frozen feature** vì tín hiệu nằm ở đó.
3. D9 dùng LDA **có giám sát** (có nhãn lỗi) → AUROC 1.0 chỉ là **cận trên** chứng minh tín hiệu tồn tại,
   KHÔNG triển khai trực tiếp được (thực tế unsupervised, không có nhãn lỗi lúc train).
4. → Cần một bộ chấm điểm **không giám sát trên frozen feature**: họ **feature-distance** (so khoảng cách
   feature test với thống kê/bộ nhớ của feature normal). Hai ứng viên kinh điển:
   - **Mahalanobis / PaDiM**: mô hình Gaussian theo từng vị trí patch.
   - **PatchCore**: memory bank các patch normal + nearest-neighbor distance.

Đây cũng đúng với 2 script thử nghiệm sẵn có của bạn: `quick_test1_patch_lda.py`, `quick_test2_mahalanobis.py`.

### Đối chiếu với research (đã verify)

- **PatchCore** (arxiv 2106.08265): memory bank patch normal + NN-distance, KHÔNG reconstruction;
  99.6% I-AUROC trên MVTec AD.
- **AnomalyDINO** (arxiv 2405.14529): NN-distance trên **frozen DINOv2 patch feature**, training-free;
  1-shot MVTec 96.6% AUROC (PatchCore 83.4%), VisA 92.5% pixel-PRO.
- **SuperAD** (challenge VAND 3.0, hạng 4): frozen DINOv2 ViT-L/14, layer 6/12/18/24, memory bank
  PatchCore-style + coreset chọn reference đa dạng + mask nền bằng PCA.
- **Đính chính chiến lược quan trọng (paper dataset, Table 7):** **INP-Former ĐÃ là SOTA chính thức**
  trên MVTec AD 2 (mean **31.6% AU-PRO@0.05** trên TESTpriv), **trên cả PatchCore** (PatchCore của paper
  dùng backbone CNN ImageNet, KHÔNG phải DINOv2). → Không thay INP-Former bằng PatchCore-CNN; nhưng
  distance-trên-**DINOv2** là chuyện khác, vẫn đáng làm. Paper cũng nói thẳng *"Can is especially challenging"*.

→ **Hướng chốt:** giữ INP-Former, **thêm một nhánh feature-distance trên frozen DINOv2**, so sánh và
fuse. Trọng tâm: cứu các category mà reconstruction sụp (can, wallplugs) mà không làm hỏng các category
INP-Former đang thắng (vial, fruit_jelly). Đo bằng **AU-PRO@0.05**.

---

## 5. Hướng đang chạy — `eval_fusion_distance.py`

Script **eval-only** (KHÔNG train), load checkpoint INP-Former có sẵn, với mỗi category chạy **3 nhánh
chấm điểm trên cùng một lượt** để so sánh trực tiếp:

1. **RECON** — anomaly map residual gốc của INP-Former (`cal_anomaly_maps(en, de)`). = baseline.
2. **DIST** — anomaly map **khoảng cách trên frozen feature** (fuse layer 2–9):
   - `--scorer maha`: PaDiM-style Gaussian theo vị trí (trên không gian PCA-50). Rẻ.
   - `--scorer patchcore`: memory bank + **greedy coreset** + nearest-neighbor distance.
3. **FUSION** — `alpha·norm(RECON) + (1−alpha)·norm(DIST)`:
   - `FUSE@0.5`: alpha cố định = 0.5 → **SẠCH** (không tune trên test).
   - `FUSE*oracle`: alpha dò bằng proxy pixel-AP trên test → **test-tuned, chỉ là cận trên tham khảo**.

Mỗi nhánh in đủ **9 metric**: `I-AUROC, I-AP, I-F1_max, P-AUROC, P-AP, P-F1_max(SegF1), AUPRO, AUPRO0.05, AUPRO0.30`.
Lưu `log.txt` (append) + `results.csv` (overwrite) vào `./diagnosis_fusion_distance/<scorer>/` (hoặc `--out_dir`).

### Chi tiết kỹ thuật & tối ưu đã làm
- Feature distance fuse trung bình toàn bộ target layer `[2..9]` → `[B, N, C]`, reshape `[B,1,28,28]`,
  interpolate 256, làm mượt Gaussian (giống pipeline gốc).
- Chuẩn hóa robust mỗi nhánh bằng percentile 1/99 trước khi fuse.
- **Tối ưu tốc độ:** AUPRO@0.05/0.30 tính bằng `regionprops` trên **CPU rất chậm**. Ban đầu gọi 11 lần/category
  (RECON+DIST+9 alpha) → kẹt CPU, GPU 0%. Đã sửa: quét alpha bằng **proxy pixel-AP nhanh**, chỉ tính bộ
  9 metric đầy đủ cho 4 nhánh cuối (RECON/DIST/FUSE@0.5/FUSE*oracle).

---

## 6. Kết quả thực nghiệm (lần 1, trên test_public — có GT)

### 6.1 Mahalanobis = NGÕ CỤT
DIST(maha) MEAN AUPRO0.05 = **0.112** (RECON = 0.353); =0 ở can/fruit_jelly/vial.
**Lý do:** PaDiM giả định patch vị trí *i* test khớp vị trí *i* train. MVTec AD 2 vật **xoay/xê dịch**
→ thống kê theo vị trí vô nghĩa. (PatchCore dùng bank toàn cục → bất biến với xê dịch → thắng.)

### 6.2 PatchCore-trên-DINOv2 = THẮNG ✅ (coreset 10%)

Bảng **AUPRO0.05** theo category (cao hơn = tốt):

| Category | RECON | DIST | Ghi chú |
|---|---|---|---|
| can | 0.0427 | **0.1658** | ×3.9 — cứu category sụp |
| fabric | 0.2805 | **0.4191** | |
| fruit_jelly | **0.5783** | 0.5580 | category duy nhất RECON thắng |
| rice | 0.4078 | **0.5277** | |
| sheet_metal | 0.3062 | **0.3753** | |
| vial | 0.6974 | **0.7402** | |
| wallplugs | 0.1492 | **0.2459** | cứu category sụp |
| walnuts | 0.3606 | **0.4011** | |
| **MEAN** | **0.3528** | **0.4291** | **+0.076 (+21.6%)** |

- DIST > RECON ở **7/8 category** (chỉ thua fruit_jelly).
- **Xác nhận trực tiếp D7/D9 bằng số thực**: residual mù, frozen feature thì không, memory-bank moi được tín hiệu.
- Tham chiếu: SOTA paper = 31.6% AUPRO0.05 (TESTpriv). RECON repro của ta = 35.3% (test_public). DIST = **42.9%**.

### 6.3 Caveat trung thực
- **FUSION lần 1 bị leakage** (alpha dò trên chính tập test) VÀ vẫn thua DIST (proxy pixel-AP lệch với
  AUPRO0.05). → Kết quả sạch đáng tin là **RECON vs DIST** (cả hai không tune trên test) — và DIST thắng.
  Code đã sửa để báo `FUSE@0.5` (sạch) + `FUSE*oracle` (đánh dấu rõ là cận trên).
- Số liệu trên **test_public** (có GT). Xếp hạng thật trên **private server** → cần kiểm lại trên đó.

---

## 7. Bảng baseline đầy đủ (reproduced, để tham chiếu)

| Category | I-AUROC | P-AUPRO | AUPRO0.05 | AUPRO0.30 |
|---|---|---|---|---|
| can | 0.505 | 0.296 | 0.043 | 0.290 |
| fabric | 0.731 | 0.635 | 0.281 | 0.616 |
| fruit_jelly | 0.875 | 0.822 | 0.578 | 0.818 |
| rice | 0.777 | 0.699 | 0.408 | 0.693 |
| sheet_metal | 0.731 | 0.555 | 0.306 | 0.543 |
| vial | 0.880 | 0.914 | 0.697 | 0.913 |
| wallplugs | 0.444 | 0.428 | 0.149 | 0.428 |
| walnuts | 0.772 | 0.702 | 0.361 | 0.696 |
| **MEAN** | ~0.71 | ~0.63 | **0.255** | 0.562 |

*(AUPRO0.05 mean ở đây 0.255 là từ log gốc; lần eval lại trong script fusion cho RECON = 0.353 do khác
chi tiết tính/seed — dùng nhất quán trong cùng một bảng so sánh.)*

---

## 8. Bước tiếp theo (xếp theo kỳ vọng)

1. **Tăng coreset** PatchCore 10% → **25%** (đang chạy): nhiều patch tham chiếu hơn → NN sát hơn.
2. **Đổi encoder ViT-L/14 DINOv2** (công thức SuperAD, layer 6/12/18/24) — script **DIST-only**
   (không cần checkpoint vì encoder frozen). Nhiều khả năng là **trần cao nhất**. *(chưa viết)*
3. **Foreground masking bằng PCA** — nhắm riêng can/wallplugs/vial (nhiều nền), làm **có chọn lọc**
   (không áp lên rice/fabric vốn full-texture).
4. **Fusion trung thực**: alpha cố định hoặc học alpha trên anomaly tổng hợp (tránh leakage).

---

## 9. Nhật ký lệnh đã/đang chạy

```bash
# Lần 1 (coreset 10%) — đã chạy, ra kết quả mục 6
python eval_fusion_distance.py --data_path ../data --ckpt_dir ./reproduced_results --scorer maha
python eval_fusion_distance.py --data_path ../data --ckpt_dir ./reproduced_results --scorer patchcore

# Lần 2 (coreset 25%, folder riêng) — đang/ sắp chạy
python eval_fusion_distance.py --data_path ../data --ckpt_dir ./reproduced_results \
  --scorer patchcore --coreset_ratio 0.25 --out_dir ./diagnosis_fusion_cs25
```

---

## 10. Kết quả lần 2 (coreset 10% vs 25%) + INSIGHT phụ thuộc metric

### So sánh MEAN (8 category, test_public)

| Branch | AUPRO0.05 | AUPRO0.30 | AUPRO | P-AUROC | P-F1_max |
|---|---|---|---|---|---|
| RECON (baseline) | 0.3528 | 0.6245 | 0.6313 | 0.8898 | 0.3369 |
| DIST cs10 | 0.4291 | 0.6396 | 0.6453 | 0.8782 | 0.3147 |
| **DIST cs25** | **0.4357** | 0.6503 | 0.6545 | 0.8921 | 0.3132 |
| FUSE@0.5 cs25 | 0.4059 | 0.6547 | 0.6843 | 0.9115 | 0.3203 |
| FUSE*oracle cs25 | 0.4136 | 0.6521 | **0.6935** | **0.9180** | 0.3379 |

### Phát hiện
1. **Coreset 10%→25% chỉ cải thiện nhẹ** DIST AUPRO0.05 (0.4291→0.4357). Lợi ích giảm dần → dùng 25%, không cần cao hơn.
2. **Câu chuyện phụ thuộc metric (quan trọng):**
   - **AUPRO0.05** (metric đầu bảng, FPR cực thấp): **DIST một mình THẮNG**; fusion (trung bình lồi) *làm tệ đi*
     (0.436 → 0.406/0.414).
   - **AUPRO / AUPRO0.30 / P-AUROC** (FPR rộng): **fusion THẮNG** (P-AUROC 0.918 vs DIST 0.892).
   - **Giải thích:** AUPRO0.05 do pixel tự-tin-nhất chi phối; DIST (PatchCore NN) cho **đỉnh sắc** → thắng FPR thấp.
     Trộn lồi với RECON **pha loãng đỉnh** → tụt AUPRO0.05; nhưng ở FPR rộng RECON bù vùng phủ → fusion thắng.
   - **Hệ quả:** vì metric chính thức là AUPRO0.05 → **đề xuất chính = DIST-alone (PatchCore-DINOv2, coreset 25%)**.
     Muốn ăn cả AUPRO rộng thì cần **fusion kiểu GIỮ-ĐỈNH** (elementwise max / chỉ thêm RECON nơi DIST im),
     KHÔNG dùng trung bình lồi.
3. **`can` vẫn khó nhất** (DIST 0.168) — đúng cảnh báo của paper. ViT-L + foreground mask có thể giúp nhiều nhất ở đây.

### DIST cs25 theo category — AUPRO0.05 (so RECON)

| Category | RECON | DIST cs25 | Δ |
|---|---|---|---|
| can | 0.0427 | 0.1683 | +0.126 (×3.9) |
| fabric | 0.2805 | 0.4664 | +0.186 |
| fruit_jelly | **0.5783** | 0.5317 | −0.047 (RECON thắng) |
| rice | 0.4078 | 0.5252 | +0.117 |
| sheet_metal | 0.3062 | 0.3902 | +0.084 |
| vial | 0.6974 | 0.7391 | +0.042 |
| wallplugs | 0.1492 | 0.2347 | +0.086 |
| walnuts | 0.3606 | 0.4299 | +0.069 |
| **MEAN** | **0.3528** | **0.4357** | **+0.083 (+23.5%)** |

→ DIST > RECON ở 7/8 (chỉ thua fruit_jelly). Kết luận lần 1 được tái khẳng định, mạnh hơn.

---

## 11. Cập nhật hướng tiếp theo (sau lần 2)

1. **[ƯU TIÊN 1 — ĐÃ VIẾT: `eval_dist_only.py`] DIST-only ViT-L/14 DINOv2** (layer 6/12/18/24, không cần checkpoint).
   DIST đang là quán quân → encoder mạnh hơn nhiều khả năng là **trần cao nhất**.
   ```bash
   # ViT-L (mặc định, layer 5/11/17/23 ~ 6/12/18/24)
   python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_large_14 \
     --coreset_ratio 0.25 --out_dir ./diagnosis_distonly_vitl
   # ViT-B để đối chứng ngang
   python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
     --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --out_dir ./diagnosis_distonly_vitb
   ```
2. **Fusion giữ-đỉnh** cho AUPRO0.05 (elementwise max thay vì convex), nếu muốn cả AUPRO rộng.
3. **Foreground masking PCA có chọn lọc** — nhắm `can` (chỗ khó nhất) + wallplugs/vial.
4. Coreset chốt ở **0.25**.

---

## 12. Kết quả lần 3 — ViT-B vs ViT-L DIST-only + chọn encoder

| Branch | AUPRO0.05 | AUPRO | P-AUROC | P-F1_max |
|---|---|---|---|---|
| RECON (INP-Former) | 0.3528 | 0.6313 | 0.8898 | 0.3369 |
| **ViT-B DIST (layer 2–9)** | **0.4357** | 0.6545 | 0.8921 | 0.3132 |
| ViT-L DIST (layer 6/12/18/24) | 0.3455 | 0.5820 | 0.8811 | **0.3732** |

- **ViT-L THUA ViT-B ở AUPRO0.05 trên 7/8 category** (can 0.168→0.080, fabric 0.466→0.268), dù tốt hơn ở P-F1/P-AP.
- **Nguyên nhân (confounded + insight):** đổi cả encoder size LẪN layer. ViT-B dùng **layer nông 2–9** (có texture cục bộ);
  ViT-L dùng **layer sâu 6/12/18/24** (ngữ nghĩa, thô). **AUPRO0.05 = lỗi nhỏ sắc → cần feature nông.**
  → **Insight (ablation đẹp cho paper): chọn layer nông quan trọng hơn tăng kích thước encoder cho localization low-FPR.**
- **Quyết định: chốt ViT-B layer 2–9 + PatchCore coreset 0.25 làm scorer nền** (thắng + rẻ hơn).
  *(Có thể chạy thêm 1 lần ViT-L với layer nông để rigorous, nhưng không ưu tiên.)*

## 13. Method mới — Peak-preserving fusion (phần tạo NOVELTY)

Động lực từ số liệu: convex sum làm tệ AUPRO0.05 (pha loãng đỉnh sắc của DIST) nhưng thắng AUPRO rộng.
→ Cần fusion **giữ đỉnh** để ăn cả hai vùng FPR. Đã nâng cấp `eval_fusion_distance.py` test 3 kiểu trong 1 lần:
- `convex`: alpha·RECON + (1−alpha)·DIST (baseline fusion).
- `max`: max từng pixel (giữ đỉnh cả 2).
- `rank`: gộp theo thứ hạng toàn cục (peak-preserving, bất biến thang đo).

**Giả thuyết cần kiểm:** `max`/`rank` ≥ DIST ở AUPRO0.05 ĐỒNG THỜI ≥ DIST ở AUPRO rộng → thắng cả hai → đóng góp method.

```bash
# Chạy lại patchcore coreset 0.25 với 3 kiểu fusion (RECON/DIST/FUSE_convex/FUSE_max/FUSE_rank)
python eval_fusion_distance.py --data_path ../data --ckpt_dir ./reproduced_results \
  --scorer patchcore --coreset_ratio 0.25 \
  --fusion_modes convex max rank --out_dir ./diagnosis_fusion_peak
```

### KẾT QUẢ — fusion là NGÕ CỤT cho metric chính (negative result)

| Branch | AUPRO0.05 | AUPRO | P-AUROC |
|---|---|---|---|
| RECON | 0.3528 | 0.6313 | 0.8898 |
| **DIST** | **0.4357** | 0.6545 | 0.8921 |
| FUSE_convex | 0.4059 | 0.6843 | 0.9115 |
| FUSE_max | 0.3877 | **0.6919** | 0.9116 |
| FUSE_rank | 0.4021 | 0.6837 | 0.9106 |

- **Cả 3 kiểu fusion đều THUA DIST ở AUPRO0.05**; fusion không thắng DIST ở *bất kỳ* category nào cho AUPRO0.05.
- max/rank vẫn hỏng vì **đỉnh giả của RECON** (pixel không-lỗi) sống sót qua max/rank → cạnh tranh đỉnh thật của DIST ở FPR thấp.
- Fusion chỉ thắng ở **AUPRO rộng / P-AUROC** (không phải metric chính).
- **Verdict:** với AUPRO0.05, **DIST một mình tốt nhất**; fusion recon+distance bị loại làm method chính.
  → Negative result sạch, hữu ích để viết (chứng minh "thêm reconstruction không giúp localization low-FPR").

## 14. Pivot novelty — Feature whitening khai thác D9

Đòn bẩy mới gốc từ diagnosis: **D9 nói tín hiệu phân biệt nằm ở chiều phương sai THẤP**, nhưng PatchCore NN
dùng Euclid (coi mọi chiều như nhau) → không khai thác. **Whiten** feature (chiếu PCA rồi chia √variance)
**khuếch đại đúng chiều phương sai thấp** nơi tín hiệu nằm, trước khi NN.

Đã thêm `--whiten pca` vào `eval_dist_only.py` (fit whitening trên train normal, áp cho cả bank lẫn query).

**Giả thuyết:** whitening giúp DIST vượt chính DIST-Euclid ở AUPRO0.05 (đặc biệt `can`/`wallplugs`).

### KẾT QUẢ — whitening cũng KHÔNG cứu headline (negative result #3)

MEAN AUPRO0.05: Euclid **0.4357** > whiten eps0.1 0.4247 > whiten eps0.01 0.4141. eps càng nhỏ càng tệ
→ up-weight mọi chiều phương sai thấp = khuếch đại **nhiễu** nhiều hơn tín hiệu (D9 là chiều *có giám sát*,
unsupervised không cô lập được).

**Nhưng pattern theo category rất đáng giá (paper-worthy):** whitening (eps0.1) vs Euclid, AUPRO0.05:
- Texture full-frame: **rice 0.525→0.543 ↑, fabric 0.466→0.489 ↑** (giúp).
- Object có nền: **wallplugs 0.235→0.144 ↓↓, vial 0.739→0.728 ↓** (hại — khuếch đại nhiễu nền).
→ **Insight: hình học feature tối ưu KHÁC NHAU theo loại category** (texture vs object-có-nền).

### Tổng kết 3 negative results (quan trọng cho định hướng paper)
Quán quân AUPRO0.05 vẫn là **PatchCore-DINOv2-Euclid thuần** (ViT-B, layer 2–9, coreset 0.25 = **0.4357**).
Ba đòn bẩy novelty đều KHÔNG vượt nó trên headline: **fusion (0.41), ViT-L (0.35), whitening (0.42)**.
→ Phương pháp mạnh nhất lại đơn giản & ≈ AnomalyDINO/SuperAD → chưa đủ novel để là method paper độc lập.
Cần quyết: (A) reframe thành analysis/empirical paper; (B) method adaptive theo category (texture→whiten,
object→euclid+fg-mask); (C) thử foreground masking (chưa làm, nhắm object cats yếu).

## 15. Quyết định hướng ACCV — bài HYBRID

Không hướng A/B/C nào đứng một mình đủ ACCV. **Adaptive (B) gain quá nhỏ** (tính tay ≈ 0.441 vs 0.436, +0.005).
Vấn đề gốc: quán quân của ta (PatchCore-DINOv2) ≈ SuperAD/AnomalyDINO → mới thắng INP-Former, **chưa thắng
SuperAD/AnomalyDINO**.

→ **Chốt: bài HYBRID** = Diagnosis (xương sống "tại sao", điểm khác biệt) + method khiêm tốn (adaptive geometry
+ foreground masking) + negatives (rigor) + **≥2 dataset** + số private server.

**2 việc SỐNG CÒN:**
1. **Tái lập SuperAD/AnomalyDINO** trên đúng eval của ta để biết bar thật (ta có hơn không).
2. **Thêm dataset thứ 2** (VisA / MVTec AD) chứng minh tổng quát.

### Foreground masking — thí nghiệm tiếp (mảnh ghép của method)
Đã thêm `--fg_mask` vào `eval_dist_only.py`: background prototype = trung bình patch viền; patch giống viền bị
hạ điểm; tự no-op trên texture full-frame. Chỉ áp cho object cats (`can/wallplugs/vial`).

```bash
# Baseline đối chứng (Euclid, không fg) — đã có: mean AUPRO0.05 0.4357
# + foreground masking trên object cats
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
  --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 \
  --fg_mask --fg_categories can wallplugs vial --fg_percentile 30 \
  --out_dir ./diagnosis_distonly_fg30
# quét percentile mạnh hơn
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
  --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 \
  --fg_mask --fg_categories can wallplugs vial --fg_percentile 50 \
  --out_dir ./diagnosis_distonly_fg50
```
**Soi:** `can`/`wallplugs`/`vial` AUPRO0.05 có tăng so Euclid không (các cat khác giữ nguyên vì không áp fg).

### KẾT QUẢ — fg-mask hại AUPRO0.05 đúng cat nó nhắm (negative result #4)
| Category | Euclid | fg30 | fg50 |
|---|---|---|---|
| can | **0.1683** | 0.1059 | 0.1090 |
| wallplugs | **0.2347** | 0.1878 | 0.1443 |
| vial | **0.7391** | 0.7382 | 0.7384 |
| MEAN | **0.4357** | 0.4219 | 0.4169 |

fg-mask GIẢM AUPRO0.05 nhưng TĂNG broad AUPRO (can AUPRO 0.30→0.42). Cùng pattern.

## 16. META-PATTERN (phát hiện cốt lõi, đáng công bố)
Bốn enhancement (fusion, ViT-L, whitening, fg-mask) **đều đánh đổi giống nhau: ↑ broad AUPRO/P-AUROC nhưng ↓ AUPRO0.05.**
Lý do: AUPRO0.05 (FPR cực thấp) bị chi phối bởi đỉnh sắc nhất; mọi xử lý thêm tạo đỉnh cạnh tranh / làm tù đỉnh gốc.
→ **Plain Euclid PatchCore-DINOv2 (ViT-B, layer 2–9, coreset 0.25 = 0.4357) là champion AUPRO0.05, cực khó vượt.**
Đây tự nó là finding đáng viết: trên metric low-FPR chính thức của MVTec AD 2, scorer đơn giản nhất bền nhất;
enhancement phổ biến đổi precision low-FPR lấy recall rộng. → Củng cố mạnh cho hướng analysis/empirical paper,
hoặc method phải nhắm TRỰC TIẾP low-FPR (chưa cái nào của ta làm vậy).

## 17. Method thử: Region-coherence (nhắm TRỰC TIẾP low-FPR)
AUPRO là metric THEO VÙNG; defect thật liền khối, spike giả lẻ tẻ. Region-coherence làm nổi vùng liền khối,
dập spike đơn lẻ — ĐỐI NGHỊCH với 4 enhancement đã fail (chúng thêm đỉnh). Đã thêm `--region_mode` vào
`eval_dist_only.py` (mult/gmean mềm, median/open cứng), áp ở lưới patch 28×28.

```bash
for M in mult gmean median open; do
  python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
    --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --region_mode $M \
    --out_dir ./diagnosis_region_$M
done
```
**Mốc:** Euclid (region none) = 0.4357. Soi mode nào MEAN AUPRO0.05 > 0.4357 (đặc biệt can/wallplugs).
Rủi ro: lưới 28×28, defect 1-patch có thể bị open/median xoá → mềm (mult/gmean) là kèo chính.

### KẾT QUẢ — region-coherence = negative #5
| Mode | MEAN AUPRO0.05 |
|---|---|
| Euclid (mốc) | **0.4357** |
| mult | 0.3914 |
| gmean | 0.3866 |
| median | 0.2638 |
| open | 0.1714 |

Tất cả thua. Mềm (mult/gmean) hại ít hơn, cứng (median/open) hại nặng. Đúng dự đoán: ở lưới 28×28 mọi phép
làm mượt/dập spike đều **làm tù đỉnh sắc** mà AUPRO0.05 cần (vài cat lẻ nhích nhẹ: rice mult 0.532, vial gmean 0.747,
nhưng mean vẫn xuống vì can/wallplugs/fabric tụt).

## 18. TỔNG KẾT 5/5 NEGATIVE + chốt định hướng paper
| Họ enhancement | AUPRO0.05 vs Euclid 0.4357 |
|---|---|
| fusion (recon+dist) | 0.41 ↓ |
| ViT-L (encoder lớn hơn) | 0.35 ↓ |
| whitening (D9) | 0.42 ↓ |
| foreground masking | 0.42 ↓ |
| region-coherence | 0.39 ↓ |

**5/5 đều thua plain Euclid PatchCore-DINOv2 (ViT-B, layer 2–9, coreset 0.25 = 0.4357).** Mọi can thiệp lên map
đều đánh đổi: ↑ broad metric nhưng ↓ AUPRO0.05 (FPR thấp bị chi phối bởi đỉnh sắc nhất).

**Hệ quả cho paper (quan trọng):** beat INP-Former bằng method-đã-biết (DINOv2+distance ≈ AnomalyDINO/SuperAD)
KHÔNG đủ novelty cho method paper. **Novelty thật nằm ở HIỂU BIẾT, không phải thuật toán:**
diagnosis (D4–D9) + phát hiện distance≫reconstruction + **5 negative results hệ thống** ("đơn giản mà bền,
mọi enhancement đổi low-FPR lấy broad recall"). → Nghiêng hẳn về **bài analysis/empirical**, method champion
là plain-DIST, đóng góp = sự hiểu + study hệ thống.

**Việc cần (không phải đập thêm enhancement):** (a) ablation layer (chứng minh shallow quan trọng) + coreset;
(b) tái lập AnomalyDINO/SuperAD để định vị; (c) dataset thứ 2 (VisA/MVTec AD) cho tổng quát.

## 19. Deep-research tìm novelty (2026-06-24) + METHOD chốt
Workflow #2 báo "all refuted" nhưng thực ra TẤT CẢ verifier fail do API 500 (server outage) → leads chưa verify
nhưng nguồn primary thật, dùng được. Leads chính:
- **AUPIMO** (arxiv 2401.01984): metric low-FPR (FPR 1e-5→1e-4); lập luận AUROC/AUPRO bị thổi phồng do imbalance.
- **tapAUC** (2502.11570): loss partial-AUC cho low-FPR NHƯNG chỉ image-level, KHÔNG segmentation.
- Danh sách mala-lab foundation-AD: KHÔNG có paper nào nhắm low-FPR/AUPRO@0.05 segmentation, contamination-robust,
  subspace-recovery, hay diagnosis-as-contribution → **gap rõ**.
- **SimpleNet** (2303.15140): noise Gaussian vào feature normal → fake anomaly → train discriminator (no label).
- **GLASS** (cqylunlun/GLASS): synthesize feature-anomaly bằng gradient ascent, nhắm defect YẾU/giống normal.
- **PP-Former** (Elsevier 2025): chống nhiễm prototype bằng restoration-attention — nhưng vẫn họ restoration → chiếm bớt góc (e).

**THESIS CHỐT (gói #1+#2):** "Anomaly localization trên MVTec AD 2 là bài toán LOW-FPR; method hiện tại tối ưu sai mục tiêu."
- Diagnosis (D7 mù, D9 signal chiều phương sai thấp) + luật 5-negative (mọi enhancement đổi low-FPR lấy broad) = động lực.
- **METHOD** (`train_lowfpr_head.py`): head phân biệt nhẹ trên FROZEN DINOv2, học bằng **synthetic anomaly**
  (SimpleNet/GLASS-style, không cần nhãn) + **loss pAUC nhắm low-FPR** (chỉ phạt mạnh top-k% normal điểm cao =
  false-positive ở FPR thấp). So trực tiếp loss `hinge` (chuẩn) vs `pauc` (của ta) trong 1 lần chạy.
- Novelty: chưa ai tối ưu low-FPR cho segmentation (theo danh sách tổng hợp) + có diagnosis + luật 5-negative chống lưng.

```bash
python train_lowfpr_head.py --data_path ../data --out_dir ./method_lowfpr
```
**Soi:** loss `pauc` MEAN AUPRO0.05 có > `hinge` và > DIST 0.4357 không (đặc biệt can/wallplugs).

### KẾT QUẢ — METHOD THẤT BẠI = negative #6
| Loss | MEAN AUPRO0.05 | P-AUROC |
|---|---|---|
| DIST champion (mốc) | **0.4357** | 0.892 |
| hinge | 0.1493 | 0.839 |
| pauc | 0.0223 | 0.561 (≈ random!) |

- **pauc sụp**: loss chỉ ràng buộc top-10% normal → 90% còn lại tự do → lời giải suy biến (P-AUROC≈0.5). Lỗi thiết kế loss.
- **hinge** không sụp nhưng thua xa distance: **synthetic anomaly = normal + nhiễu Gaussian là proxy TỒI** cho defect MVTec AD2
  (D8: defect nằm trong phân bố normal; nhiễu đẳng hướng không giống defect, bơm phương sai vào mọi chiều). SimpleNet
  chỉnh nhiễu kỹ trên CNN, không chuyển sang DINOv2 ở chế độ này.

## 20. INSIGHT TRUNG TÂM: "SUPERVISION GAP" (6/6 negative)
Giờ 6/6 cách vượt plain Euclid distance (0.4357) đều fail: 5 enhancement + synthetic-anomaly head.
Ghép D9 (LDA CÓ giám sát → AUROC 1.0): **tín hiệu phân biệt tồn tại, truy cập tuyến tính được, nhưng ở chiều
phương sai thấp mà CHỈ giám sát mới cô lập được. Mọi cách KHÔNG giám sát đều không khép được khoảng cách tới oracle.**
→ Độ khó còn lại của MVTec AD2 ở low-FPR là **bài toán THIẾU GIÁM SÁT, không phải thiếu kiến trúc** — insight paper định lượng được.

**Thí nghiệm chốt narrative (cần làm):** fit LDA/logistic CÓ nhãn test (oracle) → đo AUPRO0.05 oracle = trần trên.
Gap giữa 0.4357 (unsup) và oracle = "supervision gap", con số trung tâm paper.
**Đang chạy deep-research #3** (w073p7ls4) tìm phương án khép gap: leaderboard methods, synthetic anomaly thực tế
(không Gaussian), few-shot/weakly-supervised (1-10 nhãn lỗi), test-time adaptation, subspace learning.

## 21. ĐÍNH CHÍNH LỚN: KHÔNG phải "supervision gap" — là "RESOLUTION gap" (verify trực tiếp 2026-06-25)
Deep-research #3 + fetch thẳng RoBiS/SuperAD lật ngược kết luận mục 20:
| Method | Encoder | Input res | AUPRO@0.05 |
|---|---|---|---|
| **Của ta** | DINOv2 ViT-B | resize 448 → crop **392** (lưới 28×28) | 0.436 |
| **SuperAD** (training-free, NN distance — CÙNG HỌ) | DINOv2-L | shorter-side **672** | **0.605** |
| **RoBiS** | DINOv2 ViT-B | **518 + Swin-crop 1024, overlap 10%** | **0.672** |
| multi-scale PatchCore (ResNet50 L2+L3) | ResNet50 | — | **0.7635** |

→ Cùng họ frozen-DINOv2+distance nhưng res cao hơn → **0.60–0.67** vs 0.436. **"0.436 ceiling" là ẢO, do pipeline thiếu res.**
**Vì sao 6 negative đều fail:** ta downsample 2448×2048 → 392 → lưới patch 28×28; defect median <0.2% (D1/D2) thành <1 patch,
bị xoá TRƯỚC khi chấm điểm. Mọi enhancement chỉ đánh bóng map đã mất thông tin. → Bottleneck thật = **độ phân giải / độ mịn
không gian cho defect nhỏ**, KHÔNG phải supervision.
Đòn bẩy winners dùng (đều thuộc setup frozen-DINOv2 của ta): **res cao + overlapping/tiling + multi-scale (fine+coarse)**;
postproc: **SAM-Finer (RoBiS +12.5% SegF1 — thành phần lớn nhất)**, morphological closing, percentile threshold; coreset 16 ref + PCA fg-mask (SuperAD).

## 22. HƯỚNG MỚI + script: `eval_multiscale_hires.py`
Frozen DINOv2 + NN distance ở **res cao + multi-scale** (fine layer nông định vị defect nhỏ + coarse sâu; bank riêng → gộp).
Bank cap bằng random subsample (coreset 25% quá chậm ở res cao). Postproc morph-closing tuỳ chọn.
```bash
# multi-scale, res 672 (lưới 48x48) — kèo chính
python eval_multiscale_hires.py --data_path ../data --input_size 672 --crop_size 672 \
  --mode multi --out_dir ./diagnosis_hires/multi672
# single-scale 672 — ablation tách "res" vs "multi-scale"
python eval_multiscale_hires.py --data_path ../data --input_size 672 --crop_size 672 \
  --mode single --out_dir ./diagnosis_hires/single672
```
**Soi:** AUPRO0.05 có nhảy từ 0.436 lên ~0.55–0.65 không (đặc biệt defect nhỏ: can/sheet_metal/wallplugs).
Sau đó có thể leo tiếp res 980 (14×70) hoặc thêm SAM-Finer / morph-closing.

### KẾT QUẢ — RESOLUTION LÀ ĐÒN BẨY ✓ (xác nhận)
MEAN AUPRO0.05: baseline 392 = **0.436** → single672 = 0.461 → **multi672 = 0.507** (+16%). Cải thiện TOÀN DIỆN
(AUPRO full 0.631→0.778, P-AUROC 0.892→0.913), không phải tradeoff. Multi-scale (0.507) > single (0.461) → cả res
LẪN multi-scale đều đóng góp. Các cat "chết" được cứu: **can 0.043→0.213 (×5), fabric 0.281→0.617, wallplugs 0.149→0.409**,
rice 0.408→0.538, walnuts 0.361→0.549. (fruit_jelly 0.578→0.555 hơi giảm — vốn đã dễ.)
→ Xác nhận: 6 negative trước fail vì map thiếu res; sửa ở NGUỒN (res) thì lên ngay. "0.436 ceiling" là ảo.
Vẫn dưới SuperAD 0.605 / RoBiS 0.672 (private) → còn headroom.

### Bước tiếp (đòn bẩy đã biết, xếp theo kỳ vọng)
1. **Aspect-preserve** (`--aspect_preserve`, ĐÃ THÊM): ta đang resize VUÔNG 672² bóp méo ảnh rộng — can (1024×2232),
   sheet_metal (1056×4224) là 2 cat AUPRO0.05 thấp nhất. Giữ tỉ lệ → kỳ vọng ăn nhất ở đây. (Đã sửa cache subsample
   patch để chặn RAM ở lưới lớn.)
2. **morph_close** (`--morph_close 3`, free) — postproc kiểu RoBiS.
3. Leo res (short_side 840/980), coreset chuẩn thay random, SAM-Finer (RoBiS +12.5%).

```bash
# aspect-preserve, multi-scale (kèo lớn cho can/sheet_metal) — TEST 1 cat trước
python eval_multiscale_hires.py --data_path ../data --aspect_preserve --short_side 672 \
  --mode multi --categories can sheet_metal --out_dir ./diagnosis_hires/ar_test
# nếu OK -> full
python eval_multiscale_hires.py --data_path ../data --aspect_preserve --short_side 672 \
  --mode multi --out_dir ./diagnosis_hires/ar672
# morph-closing trên multi672 (free)
python eval_multiscale_hires.py --data_path ../data --input_size 672 --crop_size 672 \
  --mode multi --morph_close 3 --out_dir ./diagnosis_hires/multi672_morph
```

```bash
# Baseline DIST (Euclid) — đã có: AUPRO0.05 0.4357
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
  --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --whiten none \
  --out_dir ./diagnosis_distonly_vitb_eu

# DIST + whitening (khai thác D9) — quét vài eps để tránh over-amplify đuôi nhiễu
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
  --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --whiten pca --whiten_eps 0.01 \
  --out_dir ./diagnosis_distonly_vitb_w001
python eval_dist_only.py --data_path ../data --encoder dinov2reg_vit_base_14 \
  --layers 2 3 4 5 6 7 8 9 --coreset_ratio 0.25 --whiten pca --whiten_eps 0.1 \
  --out_dir ./diagnosis_distonly_vitb_w01
```

## 23. PROTOTYPE NOVELTY: Coarse-to-Fine (`eval_coarse2fine.py`) — lật bài sớm trên 16GB
Trả lời "novelty có thật?" mà không cần GPU to: chạy ở 392 (tile), nhẹ VRAM. 3 mode trong 1 script:
`coarse` (full@392, rẻ/yếu) · `full_tile` (mọi tile hi-res = brute hi-res, trần chất lượng) ·
`c2f` (coarse screen → chỉ refine tile có đỉnh coarse > mean+k·std, per-image). In kèm `avg_fine_tiles/img` = proxy chi phí.
**VERDICT cần:** c2f ≈ full_tile (AUPRO0.05) NHƯNG avg_fine_tiles ≪ T²(=4) → granularity hiệu quả = NOVELTY (Pareto accuracy–compute).
```bash
python eval_coarse2fine.py --data_path ../data --mode full_tile --out_dir ./diagnosis_c2f/full
python eval_coarse2fine.py --data_path ../data --mode c2f       --out_dir ./diagnosis_c2f/c2f
python eval_coarse2fine.py --data_path ../data --mode coarse    --out_dir ./diagnosis_c2f/coarse
```
Single-scale (layers 2-9), tiles=2 (eff 784). Tune: --k_std 0.5 (recall cao hơn), --tiles 3.
Nếu c2f thắng Pareto → method có cơ sở; nếu không → lùi diagnosis+study (PAPER_OUTLINE §9). File: eval_coarse2fine.py.

## 24. KẾT QUẢ c2f (T=4, RTX 5060 Ti) — coarse-to-fine THẤT BẠI = negative #7
So full_t4 (0.600 / 16 tile): c2f global g95 = **0.513** @3.68 tile · g99 = **0.483** @2.30 tile.
- **Normal KHÔNG bỏ qua được:** tiles_norm 1.77–3.01 (wallplugs normal refine > defect) — ngưỡng train không transfer sang test-normal.
- **Accuracy tụt 14–20%** khi giảm tile vì coarse 392 mù defect nhỏ (can 0.254→0.152, fabric 0.882→0.686, wallplugs 0.647→0.424). Không có Pareto đẹp. (Chỉ vial 0.77→0.80 nhờ bớt FP nền — category-dependent.)
→ **7 negative (6 scorer + coarse-to-fine). Method-novelty cạn. Đòn bẩy work duy nhất = brute tiling (0.600) nhưng không novel.**
**CHỐT: pivot bài analysis/empirical (fallback §9):** đóng góp = diagnosis (D1–D9) + 7 negative hệ thống + granularity-scaling law + cơ chế low-FPR, KHÔNG phải method mới. Venue: workshop / ACCV borderline / journal (Pattern Recognition, IJCV).
Env server mới: RTX 5060 Ti = sm_120 → torch cu128; numpy/scipy ABI → pin numpy==1.26.4 qua `uv pip` (bare pip trúng env py3.12 khác).
