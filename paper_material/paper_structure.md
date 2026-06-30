# Paper Structure — Few-Shot Low-FPR Anomaly Localization on MVTec AD 2

> Khung để **base theo viết paper**. Tiếng Anh = text tái dùng được; *(ghi chú tiếng Việt = nội bộ)*.
> Đánh dấu: **`[FIG: ...]`** chèn ảnh · **`[EQ: ...]`** chèn công thức · **`[TAB: ...]`** chèn bảng.
> Cập nhật 2026-06-27. Venue nhắm: ACCV.
>
> **Đã cắt cho gọn (so bản trước):** coarse-to-fine & Mahalanobis chỉ còn 1 dòng (off-thesis, không đổi
> kết luận); 7 negative gom thành **1 bảng debunk** trong Experiments (không tách mục); granularity demote
> thành **1 ablation**. Tổng còn **7 mục chính** (chuẩn 1 paper).

---

## Front-matter (không phải "section", chỉ ghi chú để viết)

**Title (chọn 1):**
1. *Closing the Rare-Normal Gap: Diagnosis-Driven Few-Shot Anomaly Localization at Low FPR*
2. *It's Not the Scorer, It's the Rare Normals: Few-Shot Low-FPR Anomaly Segmentation on Frozen Features*

**Abstract (1 đoạn, viết cuối):** MVTec AD 2 đo low-FPR (AU-PRO@0.05), defect nhỏ & in-distribution → reconstruction
SOTA (INP-Former) hỏng vì residual mù dù frozen feature tách được → distance thắng nhưng chạm trần → ta chẩn đoán
trần đó = **rare-normal** (unsup không gỡ được, oracle thì có) → few-shot head trên frozen feature: **~10 ảnh nhãn
đưa AU-PRO@0.05 0.585→0.718, SegF1 0.30→0.46**, thêm category-adaptive → [số]. *(thêm số private + dataset 2 nếu có).*

**Contributions (để cuối phần 1):**
- **C1 — Diagnosis.** Vì sao reconstruction AD hỏng ở low-FPR: residual mù (gap 0.001–0.008), prototype nhiễm +
  collapse, **nhưng** frozen feature tách tuyến tính (oracle LDA ~1.0).
- **C2 — Trần low-FPR = rare-normal.** FP ở FPR thấp là patch normal-nhưng-hiếm; mọi scorer unsup (kèm 6 enhancement)
  không tách nổi khỏi defect (HARD AUROC 0.45), **nhưng** supervised-recoverable (oracle 0.93) → là *supervision gap*.
- **C3 — Method.** Few-shot weakly-supervised head trên frozen feature nhắm đúng rare-normal gap; **~10 nhãn** →
  +0.13 AU-PRO@0.05, +0.16 SegF1; **category-adaptive operating point** đẩy thêm SegF1.
- **C4 — Generality.** Validate trên dataset thứ 2 *[VisA / MVTec AD]* + private server.

---

## 1. Introduction
- MVTec AD 2 khó: defect cực nhỏ (<0.2% diện tích), in-distribution, light-shift; SOTA <60% AU-PRO.
- Deployment quan tâm **low-FPR** → metric AU-PRO@0.05 (và SegF1).
- Kể ngắn arc: reconstruction hỏng → distance khá hơn nhưng chạm trần → trần đó là rare-normal cần ít giám sát.
- **[FIG: teaser — 1 ảnh defect + map RECON (mù) vs DIST (có FP rare-normal) vs FEW-SHOT (sạch), kèm 3 con số
  AUPRO@0.05]**.
- Contributions C1–C4.

## 2. Related Work
*(4 đoạn, mỗi đoạn chốt "khác ta ở đâu")*
- **Reconstruction AD:** INP-Former, Dinomaly, RD4AD → ta diagnose tại sao hỏng low-FPR.
- **Feature-distance AD:** PatchCore, PaDiM, AnomalyDINO, SuperAD, RoBiS → ta dùng làm nền, chỉ ra trần rare-normal họ không gỡ.
- **Few-shot / weakly-supervised AD:** PRN, DevNet, BGAD, DRA → **định vị then chốt:** họ cũng ít nhãn nhưng
  *train backbone / không nhắm low-FPR / không chẩn đoán rare-normal*. Ta = diagnosis-driven, **frozen-feature**,
  cực ít nhãn, nhắm thẳng trần low-FPR. *(viết kỹ để reviewer không nói "few-shot AD đã có".)*
- **Low-FPR metrics:** AU-PRO@0.05, AUPIMO, tapAUC → ta lấy low-FPR làm mục tiêu, không chỉ báo cáo.

## 3. Background & Setup
- MVTec AD 2 (8 cat, ảnh hi-res), tại sao khó.
- **[EQ: AU-PRO@0.05]** `= (1/0.05) ∫₀^0.05 PRO(FPR) dFPR`; giải thích vì sao low-FPR = deployment-relevant.
- Frozen encoder DINOv2-reg ViT-B/14, layer [2..9]; patch grid; tiling/res (nói luôn ta dùng eff-784 — granularity
  là setup, không phải đóng góp).

## 4. Diagnosis (C1 + C2) — *xương sống analysis*
*(Gộp toàn bộ phần "tại sao" vào 1 mục, ~4 hình. Đây là phần làm paper khác biệt.)*
- **4.1 Reconstruction mù.** **[EQ: residual]** `s_recon(p)=1−cos(en_p,de_p)`. INP nhiễm + collapse →
  **[FIG: prototype contamination + bảng cosine normal/defect ~0.99]**; residual gap **[FIG: per-layer gap 0.001–0.008]**.
- **4.2 Feature thì tách được.** **[FIG: scatter/t-SNE normal vs defect, centroid trùng nhưng LDA tách; bảng oracle
  LDA AUROC ~1.0 theo layer]** → failure ở SCORING, không ở features.
- **4.3 Distance thắng reconstruction.** **[EQ: distance]** `s_dist(p)=min_{b∈B}‖f_p−b‖`. **[TAB: RECON 0.353 vs
  DIST 0.436 per-cat, thắng 7/8]**. *(Mahalanobis/PaDiM = ngõ cụt vì vật xê dịch — 1 câu.)*
- **4.4 Trần low-FPR = rare-normal.** Định nghĩa hard-normal **[EQ: top-q% normal theo d1]**; **[TAB: FULL d1 AUROC
  0.882 vs HARD 0.453, relative-isolation 0.397]** + **[FIG: histogram d1 defect vs easy/hard-normal chồng nhau]**.
- **4.5 Nhưng gỡ được bằng giám sát.** **[TAB: oracle logistic ORACLE_FULL 0.955 / ORACLE_HARD 0.931 per-cat]**
  → đây là *supervision gap*, định lượng = khoảng cách 0.45 (unsup) ↔ 0.93 (oracle). **→ động lực trực tiếp cho few-shot.**

## 5. Method (C3)
- **Setup:** k ảnh defect có mask (k nhỏ), frozen encoder, **không train backbone**.
- **Head:** **[EQ: `g(f)=σ(wᵀφ(f)+b)`, `φ=PCA₁₂₈(standardize(f))`, LogReg class-balanced]** trên patch của k ảnh.
- **Fusion / operating points:** **[EQ: rank `r(p)=norm(d(p))`; FUSE `0.5r+0.5g`; HEAD `g`; FMULT `r·g`]**.
- **Category-adaptive operating point:** chọn branch/cat bằng **k-fold trên chính k shot** (out-of-fold P-F1max,
  không leak eval) → áp branch tốt nhất từng cat. *(động lực: rice cần UNSUP, fabric cần HEAD — heterogeneity.)*
- **[FIG: sơ đồ method — bank distance ⊕ few-shot head → (per-cat) fuse → map]**.
- **Vì sao gỡ đúng trần:** head học chiều phân biệt mà §4.5 chứng minh tồn tại, distance không với tới.

## 6. Experiments (C3 + C4)
- **6.1 Setup/protocol:** test_public eval-split (giữ k shot, eval phần còn lại + good); *[private server nếu có]*; metric AU-PRO@0.05 + SegF1.
- **6.2 Main — label efficiency.** **[TAB: k=0/1/5/10 × {UNSUP,HEAD,FUSE,FMULT}]** (0-UNSUP 0.585/0.300 →
  10-FUSE **0.7181/0.4389** → 10-HEAD 0.621/**0.4576**) + **[FIG: đường label-efficiency k vs AUPRO@0.05 & SegF1]**.
- **6.3 Category-adaptive.** **[TAB: fixed branches vs ADAPTIVE; bản đồ cat→branch]** (adaptive SegF1 ~0.51).
- **6.4 Unsupervised không gỡ được (debunk, gộp 7 negative thành 1).** **[TAB: 1 bảng — plain DIST 0.436 vs
  fusion/ViT-L/whitening/fg-mask/region/synthetic-head đều ≤0.436]** + 1 câu cơ chế (low-FPR bị chi phối bởi đỉnh
  sắc; can thiệp unsup đổi precision low-FPR lấy broad recall). *(coarse-to-fine: 1 dòng / đẩy xuống Appendix.)*
- **6.5 Granularity (ablation ngắn).** **[TAB: 392→multi672→eff784, AUPRO@0.05]** — res nâng nền nhưng vẫn còn trần.
- **6.6 So SOTA.** **[TAB: INP-Former/PatchCore/AnomalyDINO/SuperAD/RoBiS/PRN/BGAD vs ours, ghi rõ public/private]**.
- **6.7 Generality + ablations.** **[TAB: dataset 2]** + **[TAB: head_w, morph, layer nông/sâu, PCA dim, tiles]**.
- **[FIG: qualitative — gốc | GT | UNSUP | FEW-SHOT cho can/wallplugs/rice, thấy rare-normal FP bị head dập]**.

## 7. Conclusion & Limitations
- Chốt: low-FPR AD còn lại là **bài toán giám sát rare-normal, không phải kiến trúc**; ~10 nhãn phá trần.
- Limitations: few-shot = weakly-supervised (khác track unsup); cần nhãn (định vị vs PRN/BGAD); trade-off operating
  point (FUSE↔AUPRO, HEAD↔SegF1); *[caveat public vs private nếu chưa nộp server]*.

---

## Phụ lục A — checklist hình/bảng cần làm
FIG: teaser (§1) · prototype contamination (§4.1) · residual gap (§4.1) · feature separability+oracle (§4.2) ·
d1 histogram (§4.4) · method diagram (§5) · label-efficiency curve (§6.2) · qualitative maps (§6).
TAB: RECON vs DIST (§4.3) · diag10 FULL/HARD (§4.4) · oracle (§4.5) · label-efficiency (§6.2) · adaptive (§6.3) ·
negatives debunk (§6.4) · granularity (§6.5) · SOTA (§6.6) · dataset2+ablations (§6.7).

## Phụ lục B — script ↔ phần
| Phần | Script |
|---|---|
| §4.1–4.2 diagnosis | `diagnosis/diagnosis*` (D1–D9) |
| §4.3 distance | `eval_dist_only.py`, `eval_fusion_distance.py` |
| §4.4–4.5 rare-normal | `diagnosis10_bank_failure.py`, `diagnosis11_oracle_separability.py` |
| §5 method | `eval_fewshot.py`, `eval_fewshot_adaptive.py` |
| §6.2/6.3 results | `eval_fewshot.py`, `eval_fewshot_segf1.py`, `eval_fewshot_adaptive.py` |
| §6.4 negatives | `eval_dist_only.py` (whiten/fg/region), `eval_fusion_distance.py`, `train_lowfpr_head.py` |
| §6.5 granularity | `eval_multiscale_hires.py`, `eval_scalepyramid.py` |
| submit | `inference_fewshot_fuse.py`, `inference_fewshot_segf1.py` |
