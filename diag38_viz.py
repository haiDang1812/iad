# diag38_viz.py
# -----------------------------------------------------------------------------
# NHÌN TẬN MẮT các thủ phạm SegF1 (diag37 fabric verdict = bệnh C):
#   - defect LỚN (>=20k px) recall 0.157 trong khi nhỏ 0.87 → mù lệch diện rộng?
#   - FP dồn theo vật thể (000 ở MỌI lighting, 157-217k px/ảnh) → cháy sai chỗ so GT?
#     halo? defect thật lan rộng hơn GT? phải nhìn mới biết.
#
# Chạy lại chain fullscale (map gmaxz nhánh png), chọn tự động:
#   top-8 FP + mọi ảnh bad GT>=20k + top-3 FP trong ảnh good  (cap 18 ảnh/cat)
# và lưu panel JPG: [ảnh gốc | heatmap đỏ (chuẩn hóa quanh t*) | pred ĐỎ vs GT XANH
# (chồng = VÀNG)] vào out_dir/<cat>/. Đây là DIAG nhìn — không lever, không số mới.
#
#   python diag38_viz.py --data_path ../data --out_dir ./diag38 --categories fabric wallplugs
#   # xong: tar czf diag38.tgz diag38/ rồi tải về mở xem
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import IMG_EXT                                         # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_overlapmap import build_cand_overlap, overlap_score                      # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, fuse2, up_grid, guided1                         # noqa: E402
from diag37_segf1_gap import f1_argmax                                             # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')

PANEL_W = 900          # bề rộng mỗi ô panel (px)


def to_rgb(pil, w):
    im = pil.convert('RGB')
    return np.asarray(im.resize((w, round(im.size[1] * w / im.size[0])), Image.BILINEAR), np.float32)


def save_panel(path, pil, m, pred, gt, t_star, hi):
    W = PANEL_W
    img = to_rgb(pil, W)
    Hh = img.shape[0]
    mm = np.asarray(Image.fromarray(m).resize((W, Hh), Image.BILINEAR))
    pr = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize((W, Hh), Image.NEAREST)) > 127
    gg = np.asarray(Image.fromarray(gt * 255).resize((W, Hh), Image.NEAREST)) > 127
    # ô 2: heatmap — alpha từ (t* - 20% dải) tới max cat
    lo = t_star - 0.2 * (hi - t_star + 1e-6)
    a = np.clip((mm - lo) / (hi - lo + 1e-9), 0, 1)[..., None]
    heat = img * (1 - 0.7 * a) + np.array([255, 30, 30], np.float32) * 0.7 * a
    # ô 3: pred đỏ / GT xanh / chồng vàng trên nền mờ
    ov = img * 0.35
    ov[gg] += np.array([0, 190, 0], np.float32)
    ov[pr] += np.array([210, 0, 0], np.float32)
    panel = np.concatenate([img, heat, np.clip(ov, 0, 255)], axis=1).astype(np.uint8)
    Image.fromarray(panel).save(path, quality=88)


def run_cat(bb, cat, args, layers, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if args.max_train and len(tr) > args.max_train + 3:
        tr_use, va = tr[:args.max_train], tr[args.max_train:args.max_train + args.n_val]
    else:
        tr_use, va = tr[:-args.n_val], tr[-args.n_val:]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    idx = bad + good
    p(f'  [{cat}] bad={len(bad)} good={len(good)} | train_bank={len(tr_use)}')

    va_g, te_g = {}, {}
    for si, (T, gt) in enumerate(SCALES):
        R = gt * bb.patch
        cand = build_cand_overlap(bb, tr_use, T, R, gt, layers, args.enc_batch, args.cand_size, device).to(device)
        bank = coreset(cand, args.bank_size, device, seed=args.seed)
        del cand; torch.cuda.empty_cache()
        va_g[si] = [overlap_score(bb, Image.open(path), T, R, gt, layers, args.enc_batch, bank, device)
                    for path in tqdm(va, ncols=70, desc=f'    {cat} s{T}:{gt} heldout', leave=False)]
        te_g[si] = [overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch, bank, device)
                    for i in tqdm(idx, ncols=70, desc=f'    {cat} s{T}:{gt} test', leave=False)]
        del bank; torch.cuda.empty_cache()
    st = []
    for si in range(len(SCALES)):
        px = np.concatenate([g.ravel() for g in va_g[si]])
        st.append((float(px.mean()), float(px.std()) + 1e-6))
    del va_g

    def nat_map(k, pil, W, H):
        fused = fuse2(te_g[0][k], up_grid(te_g[1][k], G3, device), st)
        nat = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        return guided1(nat, load_gray(pil, device), max(1, round(min(H, W) / G3)))

    def load_gt(i, W, H):
        if ds.labels[i] == 0:
            return np.zeros((H, W), np.uint8)
        return (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)

    # pass 1: t* toàn cat
    vmin = min(min(float(te_g[0][k].min()), float(te_g[1][k].min())) for k in range(len(idx)))
    vmax = max(max(float(te_g[0][k].max()), float(te_g[1][k].max())) for k in range(len(idx)))
    h = Hist(vmin - 0.55, vmax + 0.55)
    for k, i in enumerate(tqdm(idx, ncols=70, desc=f'    {cat} pass1', leave=False)):
        pil = Image.open(ds.img_paths[i]); W, Hh = pil.size
        h.add(nat_map(k, pil, W, Hh).cpu().numpy().reshape(-1), load_gt(i, W, Hh).reshape(-1))
    f1, t_star = f1_argmax(h)
    p(f'    [{cat}] trần={f1:.4f} t*={t_star:.3f}')

    # pass 2: stats -> chọn thủ phạm
    rows = []
    for k, i in enumerate(tqdm(idx, ncols=70, desc=f'    {cat} pass2', leave=False)):
        pil = Image.open(ds.img_paths[i]); W, Hh = pil.size
        gt_ = load_gt(i, W, Hh)
        g_t = torch.from_numpy(gt_).to(device) > 0
        pred = nat_map(k, pil, W, Hh) > t_star
        rows.append((k, i, int(ds.labels[i]), int(gt_.sum()),
                     int((pred & g_t).sum()), int((pred & ~g_t).sum()), int(((~pred) & g_t).sum())))
        del pred, g_t
    by_fp = sorted(rows, key=lambda r: -r[5])
    sel = {r[0] for r in by_fp[:8]}                                  # top-8 FP
    sel |= {r[0] for r in rows if r[2] == 1 and r[3] >= 20000}        # mọi bad GT lớn
    sel |= {r[0] for r in [x for x in by_fp if x[2] == 0][:3]}        # top-3 FP good
    sel = set(list(sel)[:18])

    od = os.path.join(args.out_dir, cat)
    os.makedirs(od, exist_ok=True)
    for r in rows:
        if r[0] not in sel:
            continue
        k, i, lb, area, tp, fp, fn = r
        pil = Image.open(ds.img_paths[i]); W, Hh = pil.size
        m = nat_map(k, pil, W, Hh).cpu().numpy()
        pred = m > t_star
        name = os.path.splitext(os.path.basename(ds.img_paths[i]))[0]
        rec = tp / (tp + fn + 1e-9)
        save_panel(os.path.join(od, f'{name}_{"bad" if lb else "good"}_rec{rec:.2f}_fp{fp // 1000}k.jpg'),
                   pil, m, pred, load_gt(i, W, Hh), t_star, vmax)
        p(f'    [{cat}] lưu {name} ({"bad" if lb else "good"}) area={area} rec={rec:.2f} fp={fp}')
    del te_g


def main():
    ap = argparse.ArgumentParser('diag38: dump overlay các thủ phạm SegF1 (nhìn, không đo)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=['fabric'])
    ap.add_argument('--out_dir', type=str, default='./diag38')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('diag38', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} viz map gmaxz (chain fullscale). Panel: [gốc | heatmap | pred ĐỎ vs GT XANH].')
    for cat in args.categories:
        run_cat(bb, cat, args, layers, gk, device, p)


if __name__ == '__main__':
    main()
