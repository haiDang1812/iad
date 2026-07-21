# infer_submit_nrs3.py
# -----------------------------------------------------------------------------
# Submission v3 — 3 sửa so với infer_submit_nrs2.py, từ kết quả THẬT (server v2 +
# shotcurve 2026-07-19), thiết kế PHẪU THUẬT: chỉ đụng chỗ có bằng chứng.
#
#   1) ALL-BADS HEAD cho can/vial/wallplugs (3 cat kéo tụt private: AUPRO 61/62/69).
#      shotcurve (eval held-out cố định reserve=50, shots lồng nhau 10/25/50):
#        AUPRO tăng ĐƠN ĐIỆU 3/3 cat, vượt gate +0.02 @50:
#          can nrs   0.798 -> 0.890 -> 0.913 (+0.116)
#          wallplugs 0.867 -> 0.976 -> 0.981 (+0.114)
#          vial grid 0.804 -> 0.862 -> 0.886 (+0.082)
#      -> premise "defect-diversity transfer" SỐNG: tiff head 3 cat này train bằng
#      TOÀN BỘ bad public (hợp lệ — GT test_public công khai; paper vẫn báo 10-shot
#      là primary, all-bads là biến thể "full supervision").
#      png: F1@ksig KHÔNG tăng theo shots -> chỉ đổi nơi không tệ đi:
#          wallplugs nrs@all  (ksig 0.497 ~ 0.495 @10, trần +0.14)
#          vial      grid@all (0.409 vs 0.396 @10)
#          can       GIỮ 10-shot (all-bads đo được TỆ hơn: 0.178 vs 0.293)
#      5 cat còn lại (đang AUPRO 85-95) KHÔNG đụng — không có số shotcurve.
#
#   2) rice rule -> 'rate' (r_target = r_pub): v1 thực 71.61 > fbrate v2 68.77
#      (prevalence rice private ~ public nên rate không sập như fabric).
#
#   3) tiff a1b0 BỎ HẲN (server v2: can −0.51, fabric −0.05 AUPRO vs nrs a0b0)
#      -> tiff = NRS a0b0 mọi cat; toàn bộ máy móc tái tạo rng của ab10 xóa.
#
#   Mọi thứ khác GIỮ NGUYÊN v2: v3_large, enc_batch=16, max_train=60, seed=0,
#   head_w=0.6, canvas 256 + gaussian(5,4), softpro, per-cat rule ksig/fbrate.
#   Head 10-shot build theo ĐÚNG thứ tự seed của v1/v2 -> trùng số đã đo.
#
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs3.py \
#       --data_path ../data --out_dir ./submit_nrs3
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_nrs3
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, save_tiff_f16, save_png_binary,
    list_split_files, SPLITS, OBJECT_FILE_COUNTER, IMG_EXT,
)
from eval_nrs_head import build_nrs_head                                # noqa: E402
from infer_submit_nrs import native_map, rate_threshold                 # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')

# per-cat: config res + head png (+ shots) + RULE ngưỡng png + shots head tiff (tiff
# luôn là NRS a0b0). f1_shots/tiff_shots: 10 = few-shot như v2, 'all' = toàn bộ bad
# public (shotcurve winner). Comment = bằng chứng.
PER_CAT = {
    'can':         dict(tiles=3, grid_tile=48, f1_head='nrs',  f1_shots=10,    rule='fbrate', tiff_shots='all'),  # AUPRO +0.116@all; png @all TỆ (0.178<0.293) -> giữ 10
    'fabric':      dict(tiles=3, grid_tile=48, f1_head='grid', f1_shots=10,    rule='ksig',   tiff_shots=10),     # v2 F1 76.13, AUPRO 95.11 — không đụng; tiff về nrs (a1b0 −0.05)
    'fruit_jelly': dict(tiles=3, grid_tile=24, f1_head='grid', f1_shots=10,    rule='fbrate', tiff_shots=10),     # v2 64.23/85.6 — không đụng
    'rice':        dict(tiles=3, grid_tile=24, f1_head='grid', f1_shots=10,    rule='rate',   tiff_shots=10),     # rate v1 thực 71.61 > fbrate 68.77
    'sheet_metal': dict(tiles=3, grid_tile=48, f1_head='grid', f1_shots=10,    rule='ksig',   tiff_shots=10),     # v2 64.00/86.96 — không đụng
    'vial':        dict(tiles=3, grid_tile=24, f1_head='grid', f1_shots='all', rule='fbrate', tiff_shots='all'),  # AUPRO +0.082; png grid@all 0.409>0.396
    'wallplugs':   dict(tiles=3, grid_tile=48, f1_head='nrs',  f1_shots='all', rule='fbrate', tiff_shots='all'),  # AUPRO +0.114; png ksig ngang, trần +0.14
    'walnuts':     dict(tiles=3, grid_tile=48, f1_head='grid', f1_shots=10,    rule='ksig',   tiff_shots=10),     # v2 72.61/91.54 — không đụng
}


def gt_stats(ds):
    """(r_pub, a_cat) trên TOÀN test_public native GT:
    r_pub = pos / tổng pixel (rate rule + log), a_cat = pos / pixel-trong-ảnh-BAD
    (mật độ defect trong ảnh bad — thành phần cố định của fbrate)."""
    pos, tot, tot_bad = 0, 0, 0
    for i in range(len(ds.img_paths)):
        w, h = Image.open(ds.img_paths[i]).size          # đọc header, không decode
        tot += w * h
        if ds.labels[i] == 1:
            tot_bad += w * h
            if isinstance(ds.gt_paths[i], str) and os.path.exists(ds.gt_paths[i]):
                pos += int((np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).sum())
    return pos / max(tot, 1), pos / max(tot_bad, 1)


def score256(s_grid, gk, device):
    """Image score = max của map 256 đã smooth (== max map native vì upsample bilinear
    không vượt max). Cùng thống kê cho public lẫn private -> tau so sánh được."""
    with torch.no_grad():
        t = torch.tensor(s_grid, device=device)[None, None].float()
        t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
        return float(gk(t).max())


def ksig_threshold(maps_iter_factory, k):
    """mean + k·σ trên pooled native maps — streaming float64."""
    n, s1, s2 = 0, 0.0, 0.0
    for m in maps_iter_factory():
        md = m.double()
        n += m.numel()
        s1 += float(md.sum())
        s2 += float((md * md).sum())
    mu = s1 / max(n, 1)
    var = max(s2 / max(n, 1) - mu * mu, 0.0)
    return mu + k * var ** 0.5


def main():
    ap = argparse.ArgumentParser('submission v3: all-bads head can/vial/wallplugs + rice rate + tiff nrs thuần')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16, help='PHẢI = 16 (mọi eval đo với 16; đổi là đổi bank)')
    ap.add_argument('--max_train', type=int, default=60, help='PHẢI = 60 (bank của mọi eval)')
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--n_neg_img', type=int, default=8000)
    ap.add_argument('--pos_per_region', type=int, default=1500)
    ap.add_argument('--max_pos', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--k', type=float, default=4.5, help='k của ksig (khớp thrrules/eval)')
    ap.add_argument('--fb_scale', type=float, default=1.0,
                    help='nhân f_bad_est (an toàn: <1 = ngưỡng cao hơn nếu nghi classifier đếm thừa bad)')
    ap.add_argument('--tau_scale', type=float, default=1.0,
                    help='nhân tau ảnh-bad (>1 = khắt khe hơn khi phân loại ảnh private là bad)')
    ap.add_argument('--no_thresholded', action='store_true')
    ap.add_argument('--tiff_compression', type=str, default='zlib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--anomaly_dirname', type=str, default='anomaly_images')
    ap.add_argument('--thresh_dirname', type=str, default='anomaly_images_thresholded')
    ap.add_argument('--categories', type=str, nargs='+', default=list(PER_CAT.keys()))
    ap.add_argument('--out_dir', type=str, default='./submit_nrs3')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_nrs3', args.out_dir).info
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p('=' * 92)
    p(f'SUBMIT NRS v3 | model={args.model} layers={layers} loss={args.loss} shots={args.shots} '
      f'head_w={args.head_w} k={args.k} fb_scale={args.fb_scale} tau_scale={args.tau_scale} '
      f'| tiff=NRS a0b0 (all-bads @ can/vial/wallplugs)  png=per-cat head @ rule (ksig|fbrate|rate)')
    p('=' * 92)

    for cat in args.categories:
        cfg = PER_CAT[cat]
        args.tiles, args.grid_tile = cfg['tiles'], cfg['grid_tile']    # build_head/build_nrs_head đọc từ args
        T, gt_ = args.tiles, args.grid_tile
        R = gt_ * bb.patch
        hw = args.head_w

        tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
        if not tr:
            p(f'  [{cat}] không có train/good -> bỏ'); continue
        if args.max_train:
            tr = tr[:args.max_train]
        p(f'  [{cat}] cfg={T}:{gt_} f1={cfg["f1_head"]}@{cfg["f1_shots"]} rule={cfg["rule"]} '
          f'tiff=nrs@{cfg["tiff_shots"]} | bank từ {len(tr)} ảnh...')
        bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
        C = bank.shape[-1]

        # ---- shots + heads ----
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(bad)
        shots10 = bad[:args.shots]

        # 10-shot heads: thứ tự seed Y HỆT v1/v2 (manual_seed -> grid -> nrs) -> trùng số đã đo.
        torch.manual_seed(args.seed)
        head_grid10 = build_head(bb, ds, shots10, bank, args, layers, device)
        head_nrs10 = build_nrs_head(bb, ds, shots10, bank, args, layers, device, p)
        if head_nrs10 is None:
            p(f'  [{cat}] NRS 10-shot không build được -> bỏ (không nộp thiếu cat!)'); continue

        # all-bads heads (chỉ cat có bằng chứng shotcurve; bad = shuffled -> shots10 là prefix).
        head_grid_all = head_nrs_all = None
        if 'all' in (cfg['f1_shots'], cfg['tiff_shots']):
            p(f'  [{cat}] build all-bads heads ({len(bad)} ảnh bad public)...')
            torch.manual_seed(args.seed)
            if cfg['f1_head'] == 'grid' and cfg['f1_shots'] == 'all':
                head_grid_all = build_head(bb, ds, bad, bank, args, layers, device)
            head_nrs_all = build_nrs_head(bb, ds, bad, bank, args, layers, device, p)
            if head_nrs_all is None:
                p(f'  [{cat}] NRS all-bads không build được -> rơi về 10-shot')

        head_tiff = head_nrs_all if (cfg['tiff_shots'] == 'all' and head_nrs_all is not None) else head_nrs10
        if cfg['f1_head'] == 'nrs':
            head_f1 = head_nrs_all if (cfg['f1_shots'] == 'all' and head_nrs_all is not None) else head_nrs10
        else:
            head_f1 = head_grid_all if (cfg['f1_shots'] == 'all' and head_grid_all is not None) else head_grid10
            if head_f1 is None:
                head_f1 = head_tiff        # grid-GT rỗng -> fallback như v2
        shots_f1 = set(bad if cfg['f1_shots'] == 'all' else shots10)
        r_pub, a_cat = gt_stats(ds)
        p(f'  [{cat}] heads OK | r_pub={r_pub:.3e} a_cat={a_cat:.3e}')

        # ---- PASS 1a: encode PUBLIC test (chỉ cần cho fbrate: fit tau ảnh-bad) ----
        pub_recs = []
        if cfg['rule'] == 'fbrate' and not args.no_thresholded:
            with torch.no_grad():
                for i in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}/public(tau)'):
                    g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
                    G = g.shape[0]
                    d = np.asarray(nn_map(g, bank, device))
                    prF = torch.sigmoid(head_f1(g.reshape(-1, C))).reshape(G, G).cpu().numpy()
                    pub_recs.append((ds.labels[i], i in shots_f1, d, prF))

        # ---- PASS 1b: encode PRIVATE 1 lần, giữ (d, prF, prA) grid-level ----
        split_recs = {}
        pooled_dist = []
        with torch.no_grad():
            for split in SPLITS:
                files, _ = list_split_files(args.data_path, cat, split)
                if files is None:
                    p(f'  [{cat}/{split}] không tồn tại -> bỏ'); continue
                exp = OBJECT_FILE_COUNTER[cat]
                if len(files) != exp:
                    p(f'  [{cat}/{split}] CẢNH BÁO: {len(files)} ảnh (checker cần {exp})')
                recs = []
                for fp in tqdm(files, ncols=80, desc=f'  {cat}/{split}'):
                    pil = Image.open(fp)
                    W, H = pil.size
                    g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
                    G = g.shape[0]
                    d = np.asarray(nn_map(g, bank, device))
                    flat = g.reshape(-1, C)
                    prF = torch.sigmoid(head_f1(flat)).reshape(G, G).cpu().numpy()
                    prA = torch.sigmoid(head_tiff(flat)).reshape(G, G).cpu().numpy()
                    recs.append((fp, H, W, d, prF, prA))
                    pooled_dist.append(d.reshape(-1))
                split_recs[split] = recs
        if not pooled_dist:
            p(f'  [{cat}] không có ảnh private -> bỏ'); continue
        all_d = np.concatenate(pooled_dist)
        lo, hi = np.percentile(all_d, 1), np.percentile(all_d, 99)

        def fuse(d, pr):
            dr = (d - lo) / (hi - lo + 1e-8)
            return ((1 - hw) * dr + hw * pr).astype(np.float32)

        # ---- ngưỡng png theo rule per-cat ----
        thr = None
        if not args.no_thresholded:
            sF = [fuse(d, prF) for recs in split_recs.values() for _, _, _, d, prF, _ in recs]
            gmin = min(float(s.min()) for s in sF) - 0.05
            gmax = max(float(s.max()) for s in sF) + 0.05

            def maps_iter():
                for recs in split_recs.values():
                    for _, H, W, d, prF, _ in recs:
                        with torch.no_grad():
                            yield native_map(fuse(d, prF), gk, H, W, device)

            rule = cfg['rule']
            if rule == 'fbrate':
                # tau từ public GOOD (cùng lo/hi private -> so sánh được); recall báo trên
                # bad KHÔNG-shot (rỗng nếu f1_shots=all -> nan, tau vẫn hợp lệ).
                s_good = [score256(fuse(d, prF), gk, device) for lb, sh, d, prF in pub_recs if lb == 0]
                s_bad = [score256(fuse(d, prF), gk, device) for lb, sh, d, prF in pub_recs if lb == 1 and not sh]
                tau = args.tau_scale * max(s_good)
                rec_pub = float(np.mean([s > tau for s in s_bad])) if s_bad else float('nan')
                pix_bad, pix_all, n_bad, n_all = 0, 0, 0, 0
                for recs in split_recs.values():
                    for _, H, W, d, prF, _ in recs:
                        n_all += 1
                        pix_all += H * W
                        if score256(fuse(d, prF), gk, device) > tau:
                            n_bad += 1
                            pix_bad += H * W
                f_bad = pix_bad / max(pix_all, 1)
                p(f'  [{cat}] tau={tau:.4f} (max good pub) | recall bad pub={rec_pub:.2f} '
                  f'| private: {n_bad}/{n_all} ảnh bad -> f_bad={f_bad:.4f}')
                if f_bad <= 0:
                    p(f'  [{cat}] CẢNH BÁO: f_bad_est=0 -> rơi về ksig')
                    rule = 'ksig'
                else:
                    r_t = args.fb_scale * f_bad * a_cat
                    p(f'  [{cat}] r_target={r_t:.3e} (fbrate; r_pub={r_pub:.3e})')
                    thr = rate_threshold(maps_iter, r_t, gmin, gmax, device)
            if rule == 'rate':
                p(f'  [{cat}] r_target={r_pub:.3e} (rate = r_pub; v1 thực rice 71.61)')
                thr = rate_threshold(maps_iter, r_pub, gmin, gmax, device)
            if rule == 'ksig':
                thr = ksig_threshold(maps_iter, args.k)
            p(f'  [{cat}] dist_lo={lo:.3f} dist_hi={hi:.3f} thr({rule})={thr:.5f}')
        del pub_recs

        # ---- PASS 2: tiff = tiff-head fused, png = F1 fused > thr ----
        for split, recs in split_recs.items():
            for fp, H, W, d, prF, prA in recs:
                stem = os.path.splitext(os.path.basename(fp))[0]
                with torch.no_grad():
                    mA = native_map(fuse(d, prA), gk, H, W, device).cpu().numpy()
                save_tiff_f16(mA, os.path.join(args.out_dir, args.anomaly_dirname, cat, split, stem + '.tiff'),
                              args.tiff_compression)
                if thr is not None:
                    with torch.no_grad():
                        mF = native_map(fuse(d, prF), gk, H, W, device).cpu().numpy()
                    save_png_binary(mF > thr, os.path.join(args.out_dir, args.thresh_dirname, cat, split, stem + '.png'))
            p(f'  [{cat}/{split}] lưu {len(recs)} tiff' + ('' if thr is None else ' + png'))

    p('\n' + '=' * 92)
    p(f'XONG: {args.out_dir}')
    p('KIỂM TRA: cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py '
      f'{os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
