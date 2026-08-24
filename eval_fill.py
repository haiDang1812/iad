# eval_fill.py
# -----------------------------------------------------------------------------
# LEVER: TOPOLOGICAL FILL (fill-holes) trên mask nhị phân — từ diag38 nhìn tận mắt:
#   fabric/000 = miếng vá cùng loại vải; map chỉ cháy VIỀN khép kín (ruột locally
#   normal → NN-distance mù ruột, recall 0.14 ~ chu-vi/diện-tích). Viền kín → điền
#   ruột = +920k px TP. SuperADD có fill-holes trong recipe (pub fabric 0.937) —
#   parity trả tiền, mình đã né vì chưa hiểu cơ chế, giờ cơ chế nhìn thấy rồi.
#
# THAY ĐỔI DUY NHẤT: pred = closing(...) -> binary_fill_holes(pred). Zero
#   hyperparameter mới. Chỉ đụng png/F1@rule; map liên tục (tiff/AUPRO/trần) giữ
#   nguyên gmaxz. FAIR, không per-cat, không GT.
#
# HẠ TẦNG (quan trọng ngang lever): dump GRID score (fine+ctx, test+heldout) ra
#   npz NGAY sau khi score từng cat -> mọi thí nghiệm post-processing sau này chạy
#   offline miễn phí, không đốt GPU lại; crash giữa chừng không mất grid đã xong.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy — không sửa sau khi thấy số):
#   1) Cơ chế: fabric F1@rule(gfill - gmaxz) kỳ vọng >= +0.30 (viền kín ở ngưỡng rule).
#      Nếu fabric không nhảy -> viền KHÔNG kín ở ngưỡng rule -> đọc lại, không tune.
#   2) FILL VÀO NỀN nếu: Δmean F1@rule >= +0.010 VÀ không cat nào tụt > 0.02
#      (fill chỉ thêm pixel dương: ruột vòng FP kín cũng bị điền -> có thể tụt cat khác).
#   3) Số F1@rule của biến thể thắng = số png chính thức -> script submit.
#
#   python eval_fill.py --data_path ../data --out_dir ./fill \
#       --categories fabric can sheet_metal fruit_jelly vial rice wallplugs walnuts
#   (fabric ĐẦU TIÊN để thấy tín hiệu cơ chế sớm nhất)
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
from scipy.ndimage import binary_fill_holes

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import IMG_EXT                                         # noqa: E402
from eval_bankmap import coreset                                                   # noqa: E402
from eval_overlapmap import build_cand_overlap, overlap_score                      # noqa: E402
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, RULE_P, RULE_G, fuse2, up_grid, guided1         # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402
from backbones_ext import load_backbone                                            # noqa: E402

warnings.filterwarnings('ignore')


def run_cat(bb, cat, args, layers, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    rng = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    if args.max_train and len(tr) > args.max_train + 3:
        tr_use, va = tr[:args.max_train], tr[args.max_train:args.max_train + args.n_val]
    else:
        tr_use, va = tr[:-args.n_val], tr[-args.n_val:]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    idx = bad + good
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | train_bank={len(tr_use)} heldout={len(va)}')

    gz = os.path.join(args.out_dir, f'grids_{cat}.npz')
    if os.path.exists(gz):
        z = np.load(gz, allow_pickle=True)
        va_g = {0: list(z['va_fine']), 1: list(z['va_ctx'])}
        te_g = {0: list(z['te_fine']), 1: list(z['te_ctx'])}
        p(f'    [{cat}] grid cache HIT: {gz}')
    else:
        va_g, te_g = {}, {}
        for si, (T, gt) in enumerate(SCALES):
            R = gt * bb.patch
            cand = build_cand_overlap(bb, tr_use, T, R, gt, layers, args.enc_batch, args.cand_size, device).to(device)
            bank = coreset(cand, args.bank_size, device, seed=args.seed)
            del cand; torch.cuda.empty_cache()
            p(f'    [{cat}] scale {T}:{gt} bank(coreset,overlap)={bank.shape[0]} tu cand={args.cand_size}')
            va_g[si] = [overlap_score(bb, Image.open(path), T, R, gt, layers, args.enc_batch, bank, device)
                        for path in tqdm(va, ncols=70, desc=f'    {cat} s{T}:{gt} heldout', leave=False)]
            te_g[si] = [overlap_score(bb, Image.open(ds.img_paths[i]), T, R, gt, layers, args.enc_batch, bank, device)
                        for i in tqdm(idx, ncols=70, desc=f'    {cat} s{T}:{gt} test', leave=False)]
            del bank; torch.cuda.empty_cache()
        np.savez_compressed(gz, va_fine=np.stack(va_g[0]), va_ctx=np.stack(va_g[1]),
                            te_fine=np.stack(te_g[0]), te_ctx=np.stack(te_g[1]),
                            paths=np.array([ds.img_paths[i] for i in idx]),
                            labels=np.array([ds.labels[i] for i in idx]),
                            va_paths=np.array(va))
        p(f'    [{cat}] grid cache SAVED: {gz}')

    st = []
    for si in range(len(SCALES)):
        px = np.concatenate([g.ravel() for g in va_g[si]])
        st.append((float(px.mean()), float(px.std()) + 1e-6))

    def gmaxz_nat(g0, g1, pil, W, H):
        fused = fuse2(g0, up_grid(g1, G3, device), st)
        nat = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        return guided1(nat, load_gray(pil, device), max(1, round(min(H, W) / G3)))

    # ---- heldout -> ngưỡng rule (map gmaxz) ----
    tr_px = []
    for k, path in enumerate(tqdm(va, ncols=70, desc=f'    {cat} heldout thr', leave=False)):
        pil = Image.open(path)
        W, H = pil.size
        tr_px.append(gmaxz_nat(va_g[0][k], va_g[1][k], pil, W, H).cpu().numpy().ravel()[::4])
    thr = float(np.percentile(np.concatenate(tr_px), RULE_P)) * RULE_G
    del tr_px, va_g
    np.savez(os.path.join(args.out_dir, f'meta_{cat}.npz'), st=np.array(st), thr=thr)

    # ---- test: trần (tham khảo) + F1@rule gmaxz vs gfill ----
    vmin = min(min(float(te_g[0][k].min()), float(te_g[1][k].min())) for k in range(len(idx)))
    vmax = max(max(float(te_g[0][k].max()), float(te_g[1][k].max())) for k in range(len(idx)))
    h = Hist(vmin - 0.55, vmax + 0.55)
    mst = {v: np.zeros(3, np.float64) for v in ('gmaxz', 'gfill')}
    for k, i in enumerate(tqdm(idx, ncols=70, desc=f'    {cat} maps', leave=False)):
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        if ds.labels[i] == 0:
            g_nat = np.zeros((H, W), bool)
        else:
            g_nat = np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127
        nat = gmaxz_nat(te_g[0][k], te_g[1][k], pil, W, H)
        m_np = nat.cpu().numpy()
        h.add(m_np.reshape(-1), g_nat.astype(np.uint8).reshape(-1))
        r = max(1, round(min(H, W) / G3))
        pred = closing(nat > thr, 2 * r + 1).cpu().numpy().astype(bool)
        for v, pd in (('gmaxz', pred), ('gfill', binary_fill_holes(pred))):
            mst[v] += ((pd & g_nat).sum(), (pd & ~g_nat).sum(), ((~pd) & g_nat).sum())
        del nat, pred
    del te_g

    out = {'f1_max': h.f1_max()}
    for v in ('gmaxz', 'gfill'):
        tp, fp, fn = mst[v]
        out[v] = float(2 * tp / (2 * tp + fp + fn + 1e-9))
    p(f'    [{cat}] trần={out["f1_max"]:.4f} | F1@rule gmaxz={out["gmaxz"]:.4f}  gfill={out["gfill"]:.4f}  '
      f'Δ={out["gfill"] - out["gmaxz"]:+.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_fill: fill-holes trên mask nhị phân (nhánh png) + dump grid cache')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--cand_size', type=int, default=200000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=200)
    ap.add_argument('--max_eval', type=int, default=0)
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--n_val', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['fabric', 'can', 'sheet_metal', 'fruit_jelly', 'vial', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./fill')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('fill', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} FILL-HOLES trên png (map gmaxz, chain fullscale nguyên vẹn) + dump grid cache. '
      f'RULE p={RULE_P} g={RULE_G} morph=CÓ. Không đụng tiff/AUPRO.')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (FULL test_public) =====')
    m_g = float(np.mean([res[c]['gmaxz'] for c in res]))
    m_f = float(np.mean([res[c]['gfill'] for c in res]))
    p(f'  F1@rule: gmaxz={m_g:.4f}  gfill={m_f:.4f}  Δ={m_f - m_g:+.4f}')
    drops = [(c, res[c]['gfill'] - res[c]['gmaxz']) for c in res if res[c]['gfill'] - res[c]['gmaxz'] < -0.02]
    if 'fabric' in res:
        p(f'  fabric: gmaxz={res["fabric"]["gmaxz"]:.4f} -> gfill={res["fabric"]["gfill"]:.4f} '
          f'(cơ chế: kỳ vọng Δ >= +0.30)')
    p(f'  cat tụt >0.02: {drops if drops else "KHÔNG"}')
    p('\nĐỌC (pre-registered): FILL VÀO nếu Δmean >= +0.010 VÀ không cat tụt >0.02. '
      'fabric không nhảy -> viền không kín ở ngưỡng rule, đọc lại, KHÔNG tune.')


if __name__ == '__main__':
    main()
