# eval_correspond.py
# -----------------------------------------------------------------------------
# PROBE cơ chế MỚI (khác hẳn INP/cvar/subspace): GEOMETRY-AWARE thay cho BAG model.
#
#   Mọi method AD2 (PatchCore/INP-Former/bank) là BAG: patch test so với patch normal
#   GẦN NHẤT Ở BẤT KỲ ĐÂU trong kho. -> rare-normal (normal thật nhưng hiếm) không có
#   hàng xóm gần -> score cao -> FP ở FPR thấp -> dập AUPRO0.05. AD2 CỐ TÌNH bẫy bag.
#
#   Câu hỏi đổi: KHÔNG hỏi "giống normal ở đâu đó?" mà "giống cái ĐÁNG LẼ ở ĐÚNG VỊ TRÍ NÀY?".
#   rare-normal có tương ứng tại đúng chỗ trong ảnh good -> khớp -> HẾT FP by construction.
#   defect không có tương ứng hợp lệ -> residual cao -> vẫn bắt.
#
#   Cô lập cơ chế bằng 1 NÚM = bán kính cửa sổ w (sau khi coi ảnh good ~ căn sẵn theo camera):
#     bag       = min NN toàn ảnh (vị trí tự do)                 = baseline leaderboard
#     corr{w}   = min NN CHỈ trong cửa sổ ±w quanh cùng vị trí   = position-aware
#   w→lớn  => corr thoái hoá về bag. w nhỏ => "so đúng chỗ".  refs của bag & corr GIỐNG HỆT
#   -> biến duy nhất = ràng buộc vị trí. Controlled sạch.
#
#   FAIR: refs chỉ train/good; GT chỉ để chấm. KHÔNG head/nhãn/shot/train.
#
#   ĐỌC: corr{w nhỏ} > bag RÕ ở AUPRO0.05 -> hình học quan trọng -> cơ chế geometry-aware
#        THẬT, đáng dồn lực (dense correspondence). ~bag / thua -> geometry không cứu -> kill.
#        Kỳ vọng: MẠNH trên nhóm cấu trúc (can/sheet_metal/vial), YẾU trên texture (fabric/rice)
#        và vật thể nằm lộn xộn (walnuts/wallplugs) -> per-cat cho biết cần dense/per-instance.
#
#   python eval_correspond.py --data_path ../data --out_dir ./corr --max_eval 30 \
#       --categories can sheet_metal vial wallplugs walnuts fabric rice fruit_jelly
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
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import img_featgrid, IMG_EXT                                 # noqa: E402
from diag30_thin_premise import eval_sgrids, norm01                                      # noqa: E402
from dataset import MVTecAD2Dataset                                                       # noqa: E402
from utils import get_gaussian_kernel, get_logger                                         # noqa: E402
from backbones_ext import load_backbone                                                   # noqa: E402

warnings.filterwarnings('ignore')
WLIST = [1, 2, 4, 8]
VARIANTS = ['bag'] + [f'corr{w}' for w in WLIST]


@torch.no_grad()
def _normgrid(grid, device):
    """(G,G,C) -> L2-normalize theo C, đưa lên device, float."""
    G, _, C = grid.shape
    g = grid.reshape(-1, C).float().to(device)
    g = F.normalize(g, dim=1)
    return g.reshape(G, G, C)


@torch.no_grad()
def bag_score(Ft, Fr, chunk=4096):
    """min NN toàn ảnh: mỗi patch test so với MỌI patch của MỌI ref (vị trí tự do).
    Ft (G,G,C) đã chuẩn hoá; Fr (K,G,G,C) đã chuẩn hoá.  score = 1 - max cos."""
    G, _, C = Ft.shape
    q = Ft.reshape(-1, C)                                  # (G²,C)
    r = Fr.reshape(-1, C)                                  # (K G²,C)
    out = torch.empty(q.shape[0], device=q.device)
    for s in range(0, q.shape[0], chunk):
        out[s:s + chunk] = (q[s:s + chunk] @ r.T).max(dim=1).values
    return (1.0 - out).reshape(G, G)


@torch.no_grad()
def corr_scores(Ft, Fr, wlist):
    """position-aware: mỗi patch test (i,j) chỉ khớp ref trong cửa sổ ±w quanh (i,j),
    lấy max cos qua K refs & qua các offset trong bán kính. Trả 1 map / mỗi w.
    Tận dụng lồng nhau: cur = max sim luỹ tích theo bán kính tăng dần -> snapshot ở mỗi w."""
    G, _, C = Ft.shape
    Wmax = max(wlist)
    by_r = {}
    for di in range(-Wmax, Wmax + 1):
        for dj in range(-Wmax, Wmax + 1):
            by_r.setdefault(max(abs(di), abs(dj)), []).append((di, dj))
    cur = torch.full((G, G), -1e4, device=Ft.device)      # r=0 phủ toàn bộ -> không còn -inf
    snaps = {}
    for r in range(0, Wmax + 1):
        for (di, dj) in by_r.get(r, []):
            ti0, ti1 = max(0, -di), G - max(0, di)         # vùng test hợp lệ
            tj0, tj1 = max(0, -dj), G - max(0, dj)
            t = Ft[ti0:ti1, tj0:tj1, :]                    # (h,w,C)
            rr = Fr[:, ti0 + di:ti1 + di, tj0 + dj:tj1 + dj, :]   # (K,h,w,C) ref lệch (di,dj)
            sim = (rr * t.unsqueeze(0)).sum(-1).max(dim=0).values  # (h,w) best qua K refs
            cur[ti0:ti1, tj0:tj1] = torch.maximum(cur[ti0:ti1, tj0:tj1], sim)
        if r in wlist:
            snaps[r] = cur.clone()
    return {f'corr{w}': (1.0 - snaps[w]) for w in wlist}


@torch.no_grad()
def score_variants(grid, refs, device):
    Ft = _normgrid(grid, device)
    Fr = torch.stack([_normgrid(r, device) for r in refs])            # (K,G,G,C)
    out = {'bag': bag_score(Ft, Fr).cpu().numpy()}
    for k, v in corr_scores(Ft, Fr, WLIST).items():
        out[k] = v.cpu().numpy()
    return out


def run_cat(bb, cat, args, layers, gk, device, p, rng):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if len(tr) < args.n_ref:
        p(f'  [{cat}] train/good < n_ref={args.n_ref} -> bỏ'); return None
    ref_paths = tr[:args.n_ref]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    sizes = [(Image.open(ds.img_paths[i]).size[1], Image.open(ds.img_paths[i]).size[0]) for i in idx]
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={T * gt_} n_ref={args.n_ref} W={WLIST}')

    refs = [img_featgrid(bb, Image.open(pth), T, R, gt_, layers, args.enc_batch) for pth in ref_paths]

    raws = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
            g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
            sv = score_variants(g, refs, device)
            for v in VARIANTS:
                raws[v].append(sv[v])
            del g
    del refs; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        sg, _ = norm01(raws[v])
        m = eval_sgrids(sg, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
        out[v] = m
        db = '' if v == 'bag' else f'   Δaupro={m["aupro"] - out["bag"]["aupro"]:+.4f}'
        p(f'    [{cat}] {v:7s}: AUPRO0.05={m["aupro"]:.4f}  SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}{db}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_correspond: geometry-aware vs bag (fair, unsup) PROBE')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--n_ref', type=int, default=4, help='số ảnh good làm tham chiếu (bag & corr DÙNG CHUNG)')
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split (fair). Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./corr')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('correspond', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} W={WLIST} '
      f'n_ref={args.n_ref} aupro_res={args.aupro_res} k={args.thr_sigma}')
    p('  FAIR: refs CHỈ train/good; GT chỉ để chấm. bag=baseline (NN toàn ảnh = leaderboard AD2).')
    p('  bag & corr DÙNG CHUNG refs -> biến duy nhất = ràng buộc vị trí (controlled).')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    p('\n' + '=' * 84 + '\n===== MEAN (AUPRO0.05 / SegF1 / trần) — bag=baseline =====')
    base = float(np.mean([res[c]['bag']['aupro'] for c in res]))
    for v in VARIANTS:
        au = float(np.mean([res[c][v]['aupro'] for c in res]))
        f1 = float(np.mean([res[c][v]['segf1'] for c in res]))
        fm = float(np.mean([res[c][v]['segf1_max'] for c in res]))
        db = '' if v == 'bag' else f'   Δ={au - base:+.4f}'
        p(f'  {v:7s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{db}')
    p('\n  Per-cat (biến thể tốt nhất theo AUPRO0.05):')
    for c in res:
        bv = max(VARIANTS, key=lambda vv: res[c][vv]['aupro'])
        r = res[c][bv]
        p(f'    [{c:11s}] best={bv:7s} AUPRO0.05={r["aupro"]:.4f} (bag={res[c]["bag"]["aupro"]:.3f})  SegF1={r["segf1"]:.4f}')
    p('\nĐỌC: corr{w nhỏ}>bag rõ -> geometry-aware ĐÚNG = cơ chế mới đáng dồn lực (dense corr).')
    p('     ~bag/thua khắp nơi -> geometry không cứu -> kill. Xem per-cat: cấu trúc thắng, texture thua?')


if __name__ == '__main__':
    main()
