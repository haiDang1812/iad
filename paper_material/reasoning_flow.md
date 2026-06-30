# Mạch lập luận đầy đủ: từ INP-Former (0.353) đến few-shot (0.7181)

> **File này viết cho người KHÔNG trực tiếp làm** vẫn hiểu được tại sao đi tới 0.7181.
> Mỗi bước trình bày theo cấu trúc: **(a) lúc đó ta biết gì / hỏi gì → (b) thử cái gì và VÌ SAO hợp lý →
> (c) kết quả (số thật) → (d) bài học, nó đẩy ta sang bước nào.**
> Các thí nghiệm thất bại được GIỮ vì chúng là **mắt xích logic** (chính vì chúng fail nên mới đi tới few-shot),
> nhưng mỗi cái đều nói rõ "fail này dạy ra điều gì".
>
> Metric: **AU-PRO@0.05** (xem Glossary) là thước đo chính; **SegF1 / P-F1max** là thước đo phụ.
> Mọi số trên **test_public** (có nhãn để đo) trừ khi ghi rõ.

---

## Glossary — đọc trước (các khái niệm dùng xuyên suốt)

- **Anomaly localization (phân vùng bất thường):** với mỗi pixel, model cho 1 điểm "bất thường tới mức nào".
  Ghép lại thành **anomaly map**. So với mask defect thật (ground truth) để chấm.
- **Frozen encoder (encoder đông cứng):** mạng trích đặc trưng (ở đây DINOv2) **không huấn luyện lại**, chỉ dùng
  để biến ảnh → **feature** (vector mô tả từng vùng nhỏ của ảnh). Vì không train nên rẻ và ổn định.
- **Patch / patch grid:** ViT chia ảnh thành lưới ô vuông nhỏ (patch). Ảnh 392px, patch 14px → lưới **28×28**.
  Mỗi ô có 1 feature vector. "Defect < 1 patch" nghĩa là lỗi nhỏ hơn 1 ô → bị bỏ qua ở lưới thô.
- **Reconstruction residual (sai số tái tạo):** ý tưởng "học tái tạo ảnh bình thường; chỗ nào tái tạo SAI nhiều
  thì đó là lỗi". Điểm bất thường = độ lệch giữa feature gốc và feature tái tạo. INP-Former dùng cái này.
- **Memory bank + Nearest-Neighbor (NN) distance:** lưu một "kho" feature của ảnh **bình thường** (memory bank).
  Với mỗi patch test, đo **khoảng cách tới feature bình thường gần nhất** trong kho. Xa kho = lạ = nghi ngờ lỗi.
  (Đây là cách của PatchCore/AnomalyDINO/SuperAD — KHÔNG cần tái tạo.)
- **AU-PRO@0.05:** thước đo phân vùng ở **tỉ lệ báo động giả (FPR) cực thấp ≤ 5%**. Nói cách khác: "khi ta gần như
  KHÔNG cho phép báo nhầm pixel bình thường thành lỗi, model bắt được bao nhiêu vùng lỗi?". Quan trọng cho công
  nghiệp: nhà máy không chịu được nhiều báo nhầm. → **"low-FPR" = vùng FPR thấp này.**
- **SegF1 / P-F1max:** F1 ở mức pixel sau khi áp 1 ngưỡng (cân bằng precision/recall). Challenge VAND xếp hạng theo
  cái này. Khác AU-PRO@0.05 ở chỗ nó phụ thuộc ngưỡng.
- **Oracle:** thí nghiệm "ăn gian có kiểm soát" — cho model dùng **nhãn thật** để xem *về lý thuyết* tách được tới
  đâu. Oracle KHÔNG deploy được (thực tế không có nhãn), nhưng cho biết **trần trên** (signal có tồn tại không).
- **Few-shot / weakly-supervised:** dùng **rất ít** ảnh có nhãn (vài đến ~10) để học, thay vì 0 nhãn (unsupervised)
  hay hàng nghìn nhãn (supervised).

---

## Bản đồ 1 trang

```
INP-Former reconstruction (RECON)                          AUPRO0.05 = 0.353
   │  Chẩn đoán D1–D9: điểm-số (residual) MÙ, NHƯNG frozen feature TÁCH ĐƯỢC (oracle ~1.0)
   ▼
Distance trên frozen DINOv2 (memory-bank NN) @392          0.436   (+23%, thắng 7/8 category)
   │  Thử 7 cách cải tiến KHÔNG-giám-sát để vượt 0.436 -> TẤT CẢ thua (7 negative)
   │  Bài học chung: ở low-FPR mọi can thiệp unsup đổi "precision FPR thấp" lấy "recall rộng"
   ▼
Tăng granularity (độ phân giải + tiling)                   0.507 -> 0.585  (đòn bẩy THẬT nhưng vẫn còn trần)
   │  diag10 (hỏi: trần đó do đâu?):  FP ở low-FPR = "RARE-NORMAL"
   │            = patch THẬT SỰ bình thường nhưng hiếm/xa kho; distance KHÔNG tách nổi khỏi defect (0.45 ~ ngẫu nhiên)
   │  diag11 (hỏi: trần đó có phá được không?): ORACLE có nhãn tách được (0.93) -> KHÔNG phải bất khả, là THIẾU GIÁM SÁT
   ▼
Few-shot head trên frozen feature + fuse với distance      k=10: AUPRO0.05 0.718 / SegF1 0.46
   (NOVELTY: ~10 ảnh nhãn vừa đủ phá trần rare-normal)
```

---

## Giai đoạn 0 — Bài toán & baseline

**Bài toán.** MVTec AD 2: phát hiện & khoanh vùng lỗi sản phẩm công nghiệp (8 loại: can, fabric, rice, vial...).
Khó vì: (1) lỗi **cực nhỏ** (median <0.2% diện tích ảnh), (2) vật trong suốt/chồng lấp, (3) điều kiện sáng khi
test khác lúc train. SOTA hiện tại vẫn <60% → bài toán chưa "bão hòa".

**Baseline = INP-Former** (CVPR 2025), phương pháp tốt nhất mà chính paper công bố dataset báo cáo. Cơ chế
**reconstruction residual**: encoder DINOv2 đông cứng → feature; một module (INP) gộp các vùng "bình thường" của
**chính ảnh test** thành vài prototype; decoder dùng prototype đó **tái tạo lại** feature; chỗ nào tái tạo lệch
nhiều (residual cao) bị coi là lỗi.

**Số:** repro baseline = **AUPRO0.05 0.353**. Hai loại `can` và `wallplugs` gần như đoán mò (I-AUROC ~0.5).

**Câu hỏi mở đầu:** residual có thật sự "nhìn thấy" lỗi không, hay nó mù?

---

## Giai đoạn 1 — Chẩn đoán D1–D9: TẠI SAO reconstruction hỏng

*(Chuỗi thí nghiệm chẩn đoán có sẵn trong repo, `diagnosis/diagnosis*/`. Mục tiêu: tìm điểm gãy.)*

| # | Hỏi gì | Kết quả (số) | Nghĩa là gì |
|---|---|---|---|
| **D1/2** | Lỗi to hay nhỏ? | median **<0.2%** diện tích | Lỗi cực nhỏ → ở lưới 28×28 sẽ thành <1 patch (nhớ điều này, quay lại ở GĐ4). |
| **D4** | Module INP có "ăn" phải lỗi không? | vùng lỗi hút INP **bằng/hơn** vùng bình thường (tới ×3.44) | INP — vốn phải đại diện cái "bình thường" — lại học cả lỗi. |
| **D5** | Bao nhiêu prototype bị bẩn? | 4–5/6 prototype bẩn | Hầu hết prototype nhiễm lỗi. |
| **D6** | Prototype của ảnh-bình-thường vs ảnh-lỗi có khác nhau? | cosine **~0.99–1.0** (gần như y hệt) | **Prototype collapse**: không phân biệt nổi. |
| **D7 ⚠️** | Residual ở vùng lỗi có cao hơn vùng bình thường? | chênh chỉ **0.001–0.008** | **Điểm số gần như MÙ** — đây là thủ phạm gốc. |
| **D8** | Lỗi có "nằm ngoài" phân bố bình thường? | centroid lỗi vs bình thường cosine **~0.99**; đa số lỗi trong 2σ | Lỗi **tinh vi, nằm TRONG** phân bố bình thường (không phải outlier rõ ràng). |
| **D9 💡** | Liệu thông tin phân biệt có nằm sẵn trong feature? | LDA (bộ phân loại tuyến tính) trên frozen feature đạt **AUROC = 1.0 ở MỌI layer** | **Tín hiệu phân biệt CÓ SẴN, tách tuyến tính hoàn hảo** — chỉ là cách chấm điểm vứt nó đi. |

### Insight mấu chốt GĐ1
> **Thông tin để phân biệt bình thường/lỗi vẫn nằm nguyên trong frozen feature (D9), nhưng cách chấm điểm bằng
> residual đã ném nó đi (D7).**

Giải thích chỗ thoạt nhìn mâu thuẫn — D8 nói "centroid gần trùng (0.99)" mà D9 nói "tách hoàn hảo (1.0)": hai cái
nhất quán. Hướng phân biệt là một **chiều phương sai THẤP** (một trục phụ rất nhỏ trong không gian feature). Trung
bình hai lớp gần như trùng (nên cosine 0.99), nhưng vẫn có một mặt phẳng cắt tách sạch hai lớp dọc trục phụ đó.
Residual cosine không "nhìn" theo trục phụ này nên mù.

**Vì sao 2 đề xuất cũ (RCPL, DCPL) đều thất bại (thấp hơn baseline):** cả hai vẫn nằm trong khung reconstruction,
chỉ "đánh bóng" chất lượng prototype. Nhưng lúc test, INP vẫn trích từ **chính ảnh lỗi** → vẫn nhiễm (vấn đề
**cấu trúc**, không phải chất lượng train), và điểm cuối vẫn là residual mù.

**→ Quyết định:** bỏ residual. **Chấm điểm trực tiếp trên frozen feature**, nơi tín hiệu thật sự nằm.

---

## Giai đoạn 2 — Distance trên frozen DINOv2 thắng reconstruction

**Vì sao chọn distance.** D9 nói tín hiệu nằm trong feature, nhưng LDA của D9 cần **nhãn** (oracle) → không deploy
unsupervised được. Cách **không cần nhãn** để khai thác feature: **memory-bank NN distance** (xem Glossary) — lưu
kho feature bình thường, đo khoảng cách patch test tới kho. Đây đúng là họ PatchCore/AnomalyDINO/SuperAD.

**Kết quả** (`eval_dist_only.py`, ViT-B layer 2–9):

| | AUPRO0.05 |
|---|---|
| RECON (INP-Former) | 0.353 |
| **DIST (memory-bank NN)** | **0.436 (+23%)**, thắng **7/8** loại |

- Cứu đúng loại đang sụp: `can` 0.043→0.168 (×3.9), `wallplugs` 0.149→0.235.
- **Đây là xác nhận bằng số thật cho D7/D9:** residual mù, frozen feature thì không, kho feature moi được tín hiệu.
- *(Một biến thể — Mahalanobis/PaDiM, mô hình thống kê theo TỪNG VỊ TRÍ patch — là **ngõ cụt** (0.112): vì vật
  trong MVTec AD 2 **xoay/xê dịch**, "vị trí i" của ảnh test không khớp "vị trí i" lúc train. Memory bank toàn cục
  thì bất biến với xê dịch nên thắng. Bài học nhỏ: cấu trúc dữ liệu loại bỏ giả định theo-vị-trí.)*

**Nhưng:** 0.436 vẫn xa SOTA cùng họ (SuperAD 0.605, RoBiS 0.672 — số trên private). Câu hỏi tiếp: **làm sao vượt
0.436?**

---

## Giai đoạn 3 — Bảy thí nghiệm cải tiến KHÔNG-giám-sát: TẤT CẢ thua (và dạy ta điều quan trọng)

Mỗi cái dưới đây là một **giả thuyết hợp lý** để vượt distance. Quan trọng: chúng fail không phải vì làm ẩu, mà vì
**bản chất bài toán** — và chính sự fail đồng loạt này là dữ kiện dẫn tới pivot. Nhóm theo ý tưởng:

**Nhóm A — phối hợp/đổi nguồn tín hiệu**
1. **Fusion recon+distance** (giả thuyết: hai tín hiệu bù nhau). Thử convex/max/rank. → **0.41**, thua. Vì các
   "đỉnh giả" của RECON (pixel không-lỗi nhưng residual cao) sống sót qua phép gộp, cạnh tranh với đỉnh thật của DIST.
2. **ViT-L (encoder to hơn)** (giả thuyết: feature mạnh hơn → tốt hơn). → **0.35**, thua. Layer sâu của ViT-L mang
   ngữ nghĩa thô, **mất texture cục bộ** — mà lỗi nhỏ cần texture. *Bài học đẹp: với localization low-FPR, chọn
   layer NÔNG quan trọng hơn tăng kích thước encoder.*

**Nhóm B — đổi hình học feature**
3. **Whitening** (giả thuyết: D9 nói tín hiệu ở chiều phương sai thấp → "phóng đại" các chiều đó trước khi đo NN).
   → **0.42**, thua. Vì không-giám-sát thì **không biết chiều nào là tín hiệu** → phóng đại luôn cả **nhiễu** ở các
   chiều phương sai thấp. (D9 cô lập được chiều đó là nhờ CÓ nhãn.)

**Nhóm C — hậu xử lý không gian**
4. **Foreground masking** (giả thuyết: bỏ nền → bớt FP). → **0.42**, thua, hại đúng các loại nó nhắm (can/wallplugs).
5. **Region-coherence** (giả thuyết: lỗi thật liền khối, spike lẻ là giả → dập spike). → **0.39**, thua. Ở lưới
   28×28, mọi phép làm mượt/dập đều **làm tù đỉnh sắc** mà AUPRO0.05 cần.

**Nhóm D — tự tạo giám sát giả**
6. **Synthetic-anomaly head** (giả thuyết: tạo "lỗi giả" = bình thường + nhiễu Gaussian, train head phân biệt —
   kiểu SimpleNet/GLASS). → **0.15 / 0.02**, sụp. Vì D8 đã nói lỗi thật **nằm trong** phân bố bình thường; nhiễu
   Gaussian đẳng hướng **không giống** lỗi thật → head học sai thứ.

**Nhóm E — hiệu quả tính toán**
7. **Coarse-to-fine** (giả thuyết: quét thô rồi chỉ tinh chỗ nghi ngờ → rẻ mà vẫn tốt). → 0.51 nhưng **không**
   đạt Pareto (coarse 392 đã mù lỗi nhỏ; vùng bình thường không bỏ qua được). *(Off-thesis — chỉ là chuyện tốc độ.)*

### Insight GĐ3 (META-PATTERN — rất quan trọng)
> **Mọi can thiệp không-giám-sát đều đánh đổi GIỐNG NHAU: ↑ chỉ số ở FPR rộng nhưng ↓ AUPRO0.05.**
> Lý do: AUPRO0.05 (FPR cực thấp) bị chi phối bởi **vài đỉnh sắc nhất** của map. Mọi xử lý thêm hoặc tạo đỉnh
> cạnh tranh, hoặc làm tù đỉnh gốc. → distance trần trụi là vô địch ở low-FPR, **không thể vượt bằng cách unsup.**

Điều này để lại một câu hỏi treo: **tại sao** distance lại có trần ở low-FPR? Đỉnh-giả gây FP đó **là cái gì**?
Chưa trả lời được → cần thêm chẩn đoán (GĐ5).

---

## Giai đoạn 4 — Granularity (độ phân giải) là đòn bẩy THẬT (nhưng chưa đủ)

**Quan sát.** Các method thắng cùng họ frozen-DINOv2+distance đều xài **độ phân giải cao hơn nhiều**:

| Method | Input res | AUPRO0.05 |
|---|---|---|
| Của ta | 392 (lưới 28×28) | 0.436 |
| SuperAD | 672 | 0.605 |
| RoBiS | 518 + crop 1024 | 0.672 |

**Vì sao res quan trọng (nối lại với D1/D2):** ảnh gốc 2448×2048 bị thu nhỏ về 392 → lưới 28×28. Lỗi <0.2% diện
tích (D1/D2) trở thành **dưới 1 patch** → bị **xóa TRƯỚC khi chấm điểm**. Mọi enhancement ở GĐ3 chỉ đánh bóng một
cái map vốn đã mất thông tin. → **"trần 0.436" một phần là ẢO**, do pipeline thiếu res.

**Kết quả tăng granularity** (`eval_multiscale_hires.py`, `eval_scalepyramid.py`):
- multi-scale 672 = **0.507** (+16%); cứu toàn diện: can 0.043→0.213 (×5), fabric 0.281→0.617, wallplugs 0.149→0.409.
- Tiling tiles=2 (mỗi ảnh chia 2×2, mỗi ô xử lý ở 392 → hiệu dụng 784): DIST single-scale lên **0.585** (chính là
  cột "0-UNSUP" ở GĐ7).

**Nhưng** vẫn dưới SuperAD/RoBiS, và quan trọng hơn — **câu hỏi treo của GĐ3 vẫn còn**: kể cả khi đã đủ res,
distance vẫn có trần ở low-FPR. **FP còn lại là gì?** → đây là lúc cần chẩn đoán MỚI (đột phá thật bắt đầu ở đây).

---

## Giai đoạn 5 — diag10: FP ở low-FPR chính là "RARE-NORMAL"

**Ý tưởng thí nghiệm** (`diagnosis10_bank_failure.py`). FP ở low-FPR = những patch **bình thường** mà model lại
chấm điểm CAO. Giả thuyết: đó là các patch bình thường nhưng **hiếm** (texture lạ, viền, vùng sáng bất thường) →
chúng **xa memory bank** → distance cao → bị nhầm là lỗi. Gọi chúng là **rare-normal**.

Để kiểm: lấy top-5% patch bình thường có distance d1 lớn nhất (= "hard-normal" = rare-normal), rồi hỏi: distance có
phân biệt được **defect** với **hard-normal** không?

| Phép đo (pooled AUROC, càng gần 1 càng tách tốt) | FULL (defect vs MỌI normal) | HARD (defect vs rare-normal) |
|---|---|---|
| d1 (NN distance) | **0.882** | **0.453** (≈ tung đồng xu!) |
| d1/dk (relative isolation — chuẩn hóa theo mật độ địa phương) | — | 0.397 |

### Insight GĐ5 (xác định ĐÚNG kẻ thù)
> **FP ở low-FPR = rare-normal.** Trên toàn bộ normal thì distance tách tốt defect (0.882). Nhưng riêng đám
> rare-normal thì distance **bó tay** (0.453 ≈ ngẫu nhiên). **Đây là kẻ kéo trần low-FPR xuống.**

Và đây cũng **giải thích tại sao 7 negative ở GĐ3 đều fail**: tất cả đều là biến thể của "đo độ lạ không-giám-sát"
→ không cái nào phân biệt được **lạ-mà-bình-thường** với **lạ-vì-lỗi**. Ở FPR thấp, chính đám rare-normal nổi lên
thành FP và chặn trần.

**Câu hỏi tiếp:** trần này là **bất khả** (defect ≡ rare-normal, vô vọng) hay chỉ **thiếu giám sát**?

---

## Giai đoạn 6 — diag11: trần đó GỠ ĐƯỢC bằng giám sát

**Ý tưởng** (`diagnosis11_oracle_separability.py`). Dùng **oracle** (xem Glossary): cho phép một bộ phân loại
logistic **dùng nhãn thật** (đánh giá chéo 5-fold) học trên frozen feature, rồi đo nó tách rare-normal/defect tới đâu.

| | ORACLE_FULL | ORACLE_HARD (rare-normal vs defect) |
|---|---|---|
| AUROC (có giám sát) | **0.955** | **0.931** |

### Insight GĐ6 (định hướng lời giải)
> Ngay cả tập **HARD** mà distance không-giám-sát bó tay (0.45), **oracle có nhãn tách được 0.93**. → Trần low-FPR
> **KHÔNG bất khả**; nó là một **SUPERVISION GAP** (khoảng trống giám sát). Tín hiệu vẫn ở đó (đúng như D9), chỉ cần
> **một ít nhãn** để cô lập đúng chiều phân biệt mà distance unsup không với tới.

Ghép ba mảnh: **D9** (oracle LDA ~1.0 → tín hiệu tồn tại) + **diag10** (unsup HARD 0.45 → unsup không với tới) +
**diag11** (oracle HARD 0.93 → giám sát với tới). Kết luận: **độ khó còn lại của bài toán ở low-FPR là THIẾU GIÁM
SÁT, không phải thiếu kiến trúc.** → mở đường **few-shot / weakly-supervised**.

---

## Giai đoạn 7 — Few-shot weakly-supervised head (NOVELTY, đạt 0.7181)

**Ý tưởng** (`eval_fewshot.py`). Dùng **rất ít** ảnh lỗi có mask (k nhỏ) để học một bộ phân biệt nhẹ ("head") TRÊN
frozen feature (không train lại encoder), rồi **kết hợp** với distance. Đây là cách "rẻ nhất" để bơm đúng lượng
giám sát mà diag11 nói là cần.

**Chi tiết:**
- **Bank** distance từ ảnh train/good (tiles=2 → hiệu dụng 784).
- **Head:** trên feature của k ảnh nhãn → `StandardScaler → PCA(128) → LogisticRegression(class_weight=balanced)`.
  (Một bộ phân loại tuyến tính nhẹ — đúng tinh thần D9, chỉ khác là dùng ít nhãn thật thay vì oracle.)
- **3 cách kết hợp (operating point):**
  - `HEAD` = chỉ điểm head (precision cao).
  - `FUSE` = `0.5·rank(distance) + 0.5·head` (cân bằng — **bản tốt nhất cho AUPRO0.05**).
  - `FMULT` = `rank(distance)·head` (cổng nhân: cả hai phải đồng ý → precision).
- **Tách dữ liệu sạch:** giữ k ảnh lỗi làm "shot" để TRAIN head; **đánh giá trên phần lỗi còn lại + ảnh tốt**
  (disjoint với shot) → không gian lận.

**Kết quả (tiles=2, test_public):**

| k (số ảnh nhãn) | branch | AUPRO0.05 | SegF1 (P-F1max) |
|---|---|---|---|
| 0 | UNSUP (chỉ distance) | 0.5848 | 0.3000 |
| 1 | FUSE | 0.5837 | 0.2915 |
| 1 | HEAD | 0.3288 | 0.1917 |
| 5 | FUSE | 0.6542 | 0.3453 |
| 5 | HEAD | 0.5126 | 0.3090 |
| **10** | **FUSE** | **0.7181** | **0.4389** |
| 10 | HEAD | 0.6211 | **0.4576** |

### Insight GĐ7
- **k=1 hại** (1 ảnh quá ít → head overfit, HEAD sụp xuống 0.33). **k≥5 bắt đầu thắng**, **k=10 nhảy vọt:**
  AUPRO0.05 0.585→**0.718** (+0.13), SegF1 0.30→**0.46** (+0.16).
- **Đúng như diag11 dự đoán:** chỉ ~10 ảnh nhãn là đủ cô lập chiều phân biệt rare-normal/defect → **phá trần
  low-FPR** mà 7 cách unsup không phá nổi.
- **Trade-off operating point:** `FUSE` cho AUPRO0.05 cao nhất (giữ recall theo vùng); `HEAD` cho SegF1 cao nhất
  (precision). → báo cả hai, chọn theo metric đích.

### Mở rộng đang làm: category-adaptive
Bảng per-category cho thấy branch tốt nhất **khác nhau theo loại** (rice: distance đã đủ tốt, head làm hại; fabric:
head cứu mạnh 0.19→0.64). → chọn branch riêng cho từng loại bằng **k-fold trên chính k shot** (không leak) → đẩy
SegF1 trung bình ~0.45 lên ~0.51. (`eval_fewshot_adaptive.py`.)

---

## Tóm tắt: insight nào dẫn tới đâu (mạch nhân–quả)

| Insight | Từ đâu | Dẫn tới |
|---|---|---|
| Residual mù | D7 | bỏ reconstruction |
| Tín hiệu nằm trong frozen feature, ở chiều phương sai thấp | D8, D9 | dùng distance |
| Distance > reconstruction (+23%) | GĐ2 | bỏ INP-Former làm bộ chấm điểm |
| Mọi cải tiến unsup đổi low-FPR ↔ broad | 7 negative (GĐ3) | unsup đã chạm trần |
| Trần một phần là ảo do thiếu res | GĐ4 | tăng granularity → 0.585 |
| FP low-FPR = rare-normal, distance bó tay (0.45) | diag10 | xác định ĐÚNG kẻ thù |
| Rare-normal gỡ được bằng giám sát (0.93) | diag11 | few-shot là lời giải đúng hướng |
| ~10 nhãn phá trần | GĐ7 | **AUPRO0.05 0.7181 / SegF1 0.46** |

## Ba cảnh báo trung thực (để viết paper không bị reviewer bác)
1. **0.7181 đo trên eval-split của test_public** (giữ shot, eval phần còn lại), **KHÔNG phải private server**. Submit
   private là con số khác → phải caveat rõ hoặc nộp server. *(Bản unsup pyramid nộp server thật: AUPRO0.05 0.654,
   SegF1 35.15; winner VAND ~60 SegF1.)*
2. **Few-shot/weakly-supervised AD đã tồn tại** (PRN, DevNet, BGAD…). Novelty **KHÔNG phải** "dùng nhãn", mà là
   **diagnosis-driven**: chỉ ra *chính xác* trần low-FPR = rare-normal (diag10), nó supervised-recoverable (diag11),
   rồi gỡ bằng head **trên frozen feature** (không train backbone) với **cực ít nhãn**. Phải định vị rõ vs PRN/BGAD.
3. **Distance ≈ AnomalyDINO/SuperAD** → bản thân distance không novel; đóng góp = **hiểu biết (diagnosis) + few-shot
   nhắm trúng trần đã chẩn đoán**.
