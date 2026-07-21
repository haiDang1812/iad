# infer_submit_nrs2.py
# -----------------------------------------------------------------------------
# Submission v2 — sửa 2 điểm so với infer_submit_nrs.py, theo đúng 3 run Pha 1:
#
#   1) NGƯỠNG PNG: bỏ rate rule (sập khi prevalence private != public — fabric −40
#      điểm trên server). thrrules (mô phỏng bad x1/0.5/0.5/0.25 trên public) chọn
#      per-cat rule theo worst-case F1:
#        ksig   (mean + k·σ pooled)             : fabric 0.7789 / sheet_metal 0.3526
#                                                 / walnuts 0.7611 (fbrate sập fabric
#                                                 x0.5 = 0.63)
#        fbrate (exceedance = f_bad_est · a_cat): can/fruit_jelly/rice/vial/wallplugs
#                                                 (ksig sập rice 0.48, vial 0.44)
#      mean worst-case per-cat mix = 0.6041 vs rate 0.4547 (rule đã nộp v1).
#      f_bad private KHÔNG có label -> ước lượng bằng phân loại ảnh: s_img = max của
#      map 256 đã smooth; tau = max s_img trên ảnh GOOD public (cùng chuẩn hóa lo/hi
#      với private). a_cat = pos_px / pixel-trong-ảnh-bad (public native GT).
#      Guard: f_bad_est = 0 -> tự rơi về ksig (in cảnh báo).
#
#   2) HEAD TIFF: nrsab48 đo a1b0 (W_r ∝ n_r, cân region theo size thật) thắng a0b0
#      về AUPRO trên fabric (0.9812 vs 0.9575) và can (0.7431 vs 0.7413), THUA trên
#      sheet_metal (0.7855 vs 0.7960) -> tiff dùng a1b0 chỉ cho fabric + can (đã đo),
#      a0b0 (build_nrs_head) cho 6 cat còn lại. Head a1b0 tái tạo ĐÚNG eval_nrs_alpha:
#      snapshot/restore torch-rng để subsample(bank) rút y hệt, manual_seed trước init.
#
#   ViT-H bị giết bởi nrsH48 (NRS head sập trên v3_huge, grid chỉ +0.03 can, fabric
#   AUPRO thua) -> giữ v3_large. Mọi thứ khác GIỮ NGUYÊN v1: enc_batch=16,
#   max_train=60, shots=10 seed=0, head_w=0.6, canvas 256 + gaussian(5,4), softpro.
#
#   CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python infer_submit_nrs2.py \
#       --data_path ../data --out_dir ./submit_nrs2
#   cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../submit_nrs2
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
    list_split_files, SPLITS, OBJECT_FILE_COUNTER, IMG_EXT, Head, subsample,
)
from eval_nrs_head import build_nrs_head                                # noqa: E402
from eval_nrs_alpha import collect_nrs_data, train_softpro_ab           # noqa: E402
from infer_submit_nrs import native_map, rate_threshold                 # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')

# per-cat: config res + head png + RULE ngưỡng png (thrrules worst-case winner) + head tiff.
# Số trong comment: worst-case F1 của rule thắng / rule thua gần nhất (public sim).
PER_CAT = {
    'can':         dict(tiles=3, grid_tile=48, f1_head='nrs',  rule='fbrate', tiff='ab10'),  # 0.3071 (ksig 0.3054)
    'fabric':      dict(tiles=3, grid_tile=48, f1_head='grid', rule='ksig',   tiff='ab10'),  # 0.7789 (fbrate 0.6316)
    'fruit_jelly': dict(tiles=3, grid_tile=24, f1_head='grid', rule='fbrate', tiff='nrs'),   # 0.8004 (ksig 0.7831)
    'rice':        dict(tiles=3, grid_tile=24, f1_head='grid', rule='fbrate', tiff='nrs'),   # 0.7402 (ksig 0.4759)
    'sheet_metal': dict(tiles=3, grid_tile=48, f1_head='grid', rule='ksig',   tiff='nrs'),   # 0.3526 (fbrate 0.3053)
    'vial':        dict(tiles=3, grid_tile=24, f1_head='grid', rule='fbrate', tiff='nrs'),   # 0.5631 (ksig 0.4372)
    'wallplugs':   dict(tiles=3, grid_tile=48, f1_head='nrs',  rule='fbrate', tiff='nrs'),   # 0.5296 (ksig 0.5271)
    'walnuts':     dict(tiles=3, grid_tile=48, f1_head='grid', rule='ksig',   tiff='nrs'),   # 0.7611 (fbrate 0.7229)
}


def gt_stats(ds):
    """(r_pub, a_cat) trên TOÀN test_public native GT:
    r_pub = pos / tổng pixel (log đối chiếu), a_cat = pos / pixel-trong-ảnh-BAD
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
    """mean + k·σ trên pooled native maps — streaming float64 (y hệt Hist.ksig của eval
    nhưng không cần giữ histogram)."""
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
    ap = argparse.ArgumentParser('submission v2: per-cat rule (ksig/fbrate) + tiff a1b0 fabric/can')
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
    ap.add_argument('--out_dir', type=str, default='./submit_nrs2')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA không khả dụng.')
    device = 'cuda:0'
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('submit_nrs2', args.out_dir).info
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p('=' * 92)
    p(f'SUBMIT NRS v2 | model={args.model} layers={layers} loss={args.loss} shots={args.shots} '
      f'head_w={args.head_w} k={args.k} fb_scale={args.fb_scale} tau_scale={args.tau_scale} '
      f'| tiff=NRS/a1b0  png=best-F1-head @ per-cat rule (ksig|fbrate)')
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
        p(f'  [{cat}] cfg={T}:{gt_} f1_head={cfg["f1_head"]} rule={cfg["rule"]} tiff={cfg["tiff"]} '
          f'| bank từ {len(tr)} ảnh...')
        bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
        C = bank.shape[-1]

        # ---- shots + heads (thứ tự seed Y HỆT eval -> head trùng số đã đo) ----
        ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
        bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(bad)
        shots = bad[:args.shots]
        torch.manual_seed(args.seed)
        head_grid = build_head(bb, ds, shots, bank, args, layers, device)
        cpu_state = torch.get_rng_state()                  # trạng thái sau build_head =
        cuda_state = torch.cuda.get_rng_state_all()        # điểm subsample của eval_nrs_alpha
        head_nrs = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
        if head_nrs is None:
            p(f'  [{cat}] NRS không build được -> bỏ (không nộp thiếu cat!)'); continue

        head_tiff = head_nrs
        if cfg['tiff'] == 'ab10':
            # tái tạo head a1b0 của eval_nrs_alpha: restore rng -> subsample rút y hệt,
            # manual_seed trước Head() -> init trùng.
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_state)
            Xneg_bank = subsample(bank, args.n_neg).detach().cpu()
            data = collect_nrs_data(bb, ds, shots, args, layers, device, p)
            if data is None:
                p(f'  [{cat}] không có data NRS-ab -> tiff rơi về a0b0')
            else:
                Xpos_c, rids, covs, nreg, neg_img = data
                Xpos = Xpos_c.to(device)
                Xneg = torch.cat([Xneg_bank, neg_img], 0).to(device)
                mu = bank.mean(0, keepdim=True)
                sd = bank.std(0, keepdim=True) + 1e-6
                torch.manual_seed(args.seed)
                head_tiff = train_softpro_ab(Head(C, mu, sd).to(device), Xpos, rids, covs, nreg,
                                             Xneg, 1.0, 0.0, args, device)
                del Xpos, Xneg, Xpos_c, neg_img
        head_f1 = head_nrs if (cfg['f1_head'] == 'nrs' or head_grid is None) else head_grid
        r_pub, a_cat = gt_stats(ds)
        p(f'  [{cat}] head_grid={"None" if head_grid is None else "OK"} head_nrs=OK '
          f'tiff_head={cfg["tiff"] if head_tiff is not head_nrs or cfg["tiff"] == "nrs" else "a0b0(fallback)"} '
          f'| r_pub={r_pub:.3e} a_cat={a_cat:.3e}')

        # ---- PASS 1a: encode PUBLIC test (chỉ cần cho fbrate: fit tau ảnh-bad) ----
        pub_recs = []
        if cfg['rule'] == 'fbrate' and not args.no_thresholded:
            with torch.no_grad():
                for i in tqdm(range(len(ds.img_paths)), ncols=80, desc=f'  {cat}/public(tau)'):
                    g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
                    G = g.shape[0]
                    d = np.asarray(nn_map(g, bank, device))
                    prF = torch.sigmoid(head_f1(g.reshape(-1, C))).reshape(G, G).cpu().numpy()
                    pub_recs.append((ds.labels[i], i in set(shots), d, prF))

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
                # bad KHÔNG-shot (shot đã được train nhìn thấy).
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
