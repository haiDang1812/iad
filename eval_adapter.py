# eval_adapter.py
# -----------------------------------------------------------------------------
# METHOD (fair, unsup, non-huge): THÀNH PHẦN HỌC ĐƯỢC trên feature DINOv3 đông cứng.
#   Không phải trick scoring — học một adapter z = f + MLP(f) (per-patch) rồi NN trên z.
#
#   Chẩn đoán: rare-normal = ĐUÔI xa của phân bố NN-distance TRONG normal -> FP low-FPR.
#   Objective (chỉ good, KHÔNG aug, KHÔNG nhãn) = nén riêng cái đuôi về manifold, có hãm sụp đổ:
#     L_tail = mean( top-alpha% NN-distance lớn nhất trong batch normal )   (CVaR nén đuôi)
#     L_var  = mean relu(1 - std(z_d))    (VICReg variance: chống sụp về điểm)
#     L_cov  = ||off-diag Cov(z)||² / d   (khử tương quan chiều: chống sụp thông tin)
#     L_sep  = hinge( margin·mean(d_nn) - d(neg->normal) )  (ĐẨY negative off-manifold = x+eps·u)
#   -> co BÁN KÍNH trên manifold (rare-normal) nhưng GIỮ hướng thoát manifold (defect) -> không nuốt defect.
#
#   FAIR: bank & adapter CHỈ train/good; GT chỉ để chấm.
#
#   Biến thể (per-cat):
#     raw    = NN feature gốc (= pipeline hiện tại)                         [BASELINE]
#     noneg  = adapter nén đuôi KHÔNG negative (bản cũ: sụp vial/wallplugs) [CONTROL]
#     method = nén đuôi + đẩy negative off-manifold                        [METHOD]
#     select = nhánh do SYNTH-VAL chọn per-cat (FAIR)                      [METHOD v2]
#
#   SYNTH-VAL (selector fair, KHÔNG GT, KHÔNG test):
#     Kết quả adv-run: adapter thắng raw 6/8 + SegF1 x2, nhưng sụp vial/wallplugs
#     (hướng defect nằm TRONG span bị nén -> không objective nào cứu per-cat đồng loạt).
#     -> chọn nhánh per-cat bằng bộ val TỰ SINH: defect giả (blob/scratch, cut-paste/
#     noise/jitter MỨC ẢNH — họ khác hẳn negative feature-level lúc train, không tự chấm
#     mình) trên ảnh train/good NGOÀI bank; đo recall@FPR5 patch-level (threshold =
#     q95 điểm patch CLEAN -> đuôi rare-normal tính vào đúng như AUPRO0.05).
#
#   ĐỌC: select TRÚNG oracle-best >= 6/8 cat VÀ mean(select) ~ mean(best-per-cat) >> raw
#        -> method = adapter + synth-select SỐNG. select TRẬT vial/wallplugs (vẫn chọn
#        nhánh sụp) -> synth mức ảnh không phản ánh defect thật -> kill selector.
#
#   python eval_adapter.py --data_path ../data --out_dir ./adapter --max_eval 30 \
#       --categories can sheet_metal fruit_jelly vial fabric rice wallplugs walnuts
# -----------------------------------------------------------------------------
import os
import sys
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from tqdm import tqdm

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from infer_submit_mvtec_ad2 import img_featgrid, build_bank, IMG_EXT                          # noqa: E402
from diag30_thin_premise import eval_sgrids, norm01                                           # noqa: E402
from dataset import MVTecAD2Dataset                                                            # noqa: E402
from utils import get_gaussian_kernel, get_logger                                              # noqa: E402
from backbones_ext import load_backbone                                                        # noqa: E402

warnings.filterwarnings('ignore')


class Adapter(nn.Module):
    """z = f + MLP(f), per-patch. Residual -> khởi tạo ≈ identity."""
    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)          # bắt đầu = identity (MLP=0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def _adv_delta(ad, x, z, eps):
    """GLASS-style cho metric: tìm hướng adapter đang CO mạnh nhất tại x.
    1 bước gradient GIẢM output-displacement ||ad(x+δ)-ad(x)|| ở biên độ ||δ||=eps cố định,
    rồi chiếu lại biên độ (truncated projection). Negative đặt ở đó -> hinge bịt đúng hướng co."""
    u = torch.randn_like(x)
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
    d0 = (eps * u).detach().requires_grad_(True)
    out = (ad(x + d0) - z.detach()).norm(dim=1).sum()
    g = torch.autograd.grad(out, d0)[0]
    with torch.no_grad():
        d1 = d0 - 0.5 * eps * g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)
        d1 = eps * d1 / d1.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return d1.detach()


def adapter_loss(ad, x, alpha, lam_v, lam_c, lam_s, beta_lo, beta_hi, margin, neg_mode='rand'):
    """Nén đuôi normal (ON-manifold) + ĐẨY negative tổng hợp (OFF-manifold) + chống sụp.
    Negative = x + eps*u (u hướng ngẫu nhiên ~ vuông góc manifold trong high-D; eps tự-hiệu-chuẩn
    theo NN-distance batch). Adapter phải co BÁN KÍNH trên manifold nhưng KHÔNG co hướng thoát ->
    rare-normal bị kéo về, defect (khác HƯỚNG dù cùng bán kính) không bị nuốt."""
    z = ad(x)
    B, d = z.shape
    D = torch.cdist(z, z)
    eye = torch.eye(B, device=z.device, dtype=z.dtype)
    D = D + eye * 1e9                                 # loại đường chéo (không inplace -> backward ok)
    d_nn = D.min(dim=1).values                       # (B,) NN-distance trong batch
    k = max(1, int(alpha * B))
    L_tail = torch.topk(d_nn, k, largest=True).values.mean()   # nén đuôi (alpha=1 -> nén chung)
    L_sep = z.new_zeros(())
    contrast = 0.0
    if lam_s > 0:
        with torch.no_grad():
            beta = torch.empty(B, 1, device=x.device).uniform_(beta_lo, beta_hi)
            eps = beta * d_nn.mean()                  # biên độ tự-hiệu-chuẩn theo scale NN hiện tại
        if neg_mode == 'adv':
            x_neg = x + _adv_delta(ad, x, z, eps)     # hướng đối kháng: đúng hướng adapter đang co
        else:
            with torch.no_grad():
                u = torch.randn_like(x)
                u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
                x_neg = x + eps * u
        z_neg = ad(x_neg)
        d_neg = torch.cdist(z_neg, z.detach()).min(dim=1).values
        m = margin * d_nn.mean().detach()             # margin TƯƠNG ĐỐI (bất biến scale -> không lách bằng co toàn cục)
        L_sep = torch.relu(m - d_neg).mean()
        contrast = (d_neg.mean() / d_nn.mean().clamp_min(1e-8)).item()
    std = z.std(dim=0)
    L_var = torch.relu(1.0 - std).mean()
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.t() @ zc) / (B - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    L_cov = (off ** 2).sum() / d
    loss = L_tail + lam_v * L_var + lam_c * L_cov + lam_s * L_sep
    return loss, (L_tail.item(), L_sep.item(), L_var.item(), std.mean().item(), contrast)


def train_adapter(pool, alpha, lam_s, args, device, p, tag):
    """pool: (N,C) feature normal ĐÃ standardize. Trả adapter đã train."""
    N, C = pool.shape
    ad = Adapter(C, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(ad.parameters(), lr=args.lr)
    bs = min(args.batch, N)
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=device)
        logs = []
        for s in range(0, N - bs + 1, bs):
            idx = perm[s:s + bs]
            x = pool[idx]
            loss, comps = adapter_loss(ad, x, alpha, args.lam_v, args.lam_c, lam_s,
                                       args.beta_lo, args.beta_hi, args.margin, args.neg_mode)
            opt.zero_grad(); loss.backward(); opt.step()
            logs.append(comps)
        if ep == 0 or ep == args.epochs - 1:
            lt, ls, lv, st, ct = np.mean(logs, axis=0)
            p(f'      [{tag} a={alpha} lam_s={lam_s}] ep{ep}: tail={lt:.3f} sep={ls:.3f} var={lv:.3f} std(z)={st:.3f} contrast={ct:.2f}')
    ad.eval()
    return ad


@torch.no_grad()
def nn_map(q, bank, G, chunk=2048):
    Np = q.shape[0]
    d = torch.empty(Np, device=q.device)
    for s in range(0, Np, chunk):
        d[s:s + chunk] = torch.cdist(q[s:s + chunk], bank).min(dim=1).values
    return d.reshape(G, G).cpu().numpy()


def make_mask(H, W, rng):
    """Mask anomaly giả: blob (noise mượt ngưỡng) HOẶC vệt mảnh (scratch) — khớp defect AD2."""
    if rng.random() < 0.5:                                             # blob
        s = int(rng.integers(6, 24))
        small = (rng.random((s, s)) * 255).astype(np.uint8)
        m = np.asarray(Image.fromarray(small).resize((W, H), Image.BILINEAR), np.float32) / 255.
        thr = np.quantile(m, rng.uniform(0.75, 0.93))
        mask = m > thr
    else:                                                             # vệt mảnh (scratch)
        img = Image.new('L', (W, H), 0)
        d = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(1, 4))):
            pts = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(int(rng.integers(2, 5)))]
            d.line(pts, fill=255, width=int(rng.integers(1, 4)))
        mask = np.asarray(img) > 127
    return mask


def synth_anomaly(pil, src_pil, rng):
    """(anom_pil, mask uint8). blend cut-paste / noise / jitter trong mask. MỨC ẢNH ->
    họ synthesis KHÁC negative feature-level của L_sep -> selector không tự chấm mình."""
    im = np.asarray(pil.convert('RGB'), np.float32)
    H, W = im.shape[:2]
    mask = make_mask(H, W, rng)
    if mask.sum() < 3:
        mask[H // 2, W // 2] = True
    mode = int(rng.integers(0, 3))
    if mode == 0:                                                     # cut-paste ảnh khác
        content = np.asarray(src_pil.convert('RGB').resize((W, H)), np.float32)
    elif mode == 1:                                                   # noise
        content = rng.integers(0, 256, (H, W, 3)).astype(np.float32)
    else:                                                             # jitter cường độ/màu
        content = np.clip(im * rng.uniform(0.3, 1.7) + rng.uniform(-50, 50), 0, 255)
    beta = rng.uniform(0.2, 0.9)
    out = im.copy()
    out[mask] = (1 - beta) * im[mask] + beta * content[mask]
    return Image.fromarray(out.astype(np.uint8)), mask.astype(np.uint8)


@torch.no_grad()
def _branch_maps(bb, pil, T, R, gt_, layers, eb, mu, sg, ad_n, ad_m, banks):
    g = img_featgrid(bb, pil, T, R, gt_, layers, eb)
    G = g.shape[0]; C = g.shape[-1]
    q_raw = g.reshape(-1, C).to(mu.device)
    q_std = (q_raw - mu) / sg
    qs = {'raw': q_raw, 'noneg': ad_n(q_std), 'method': ad_m(q_std)}
    del g
    return {v: nn_map(qs[v], banks[v], G) for v in qs}, G


def _auroc(pos, neg):
    a = np.concatenate([pos, neg])
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    return float((ranks[:len(pos)].mean() - (len(pos) + 1) / 2) / max(1, len(neg)))


def synth_select(bb, cat, va, args, layers, mu, sg, ad_n, ad_m, banks, p, rng_s, overlap):
    """Selector FAIR: chấm 3 nhánh trên val TỰ SINH, chọn theo recall@FPR5 (tie: auroc).
    neg = MỌI patch ảnh clean + patch ngoài mask ảnh synth -> threshold q95 ăn cả đuôi rare-normal."""
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    pos = {v: [] for v in banks}; neg = {v: [] for v in banks}
    for path in tqdm(va, ncols=70, desc=f'    {cat} synthval', leave=False):
        base = Image.open(path).convert('RGB')
        mp, G = _branch_maps(bb, base, T, R, gt_, layers, args.enc_batch, mu, sg, ad_n, ad_m, banks)
        for v in banks:
            neg[v].append(mp[v].ravel())
        for _ in range(args.n_per):
            src = Image.open(va[int(rng_s.integers(0, len(va)))]).convert('RGB')
            anom, gm = synth_anomaly(base, src, rng_s)
            mg = np.asarray(Image.fromarray(gm * 255).resize((G, G), Image.BILINEAR), np.float32) / 255.
            pm, nm = mg > 0.3, mg == 0.                               # patch lửng (0..0.3] bỏ: nhãn nhiễu
            mp2, _ = _branch_maps(bb, anom, T, R, gt_, layers, args.enc_batch, mu, sg, ad_n, ad_m, banks)
            for v in banks:
                pos[v].append(mp2[v][pm]); neg[v].append(mp2[v][nm])
    stats = {}
    for v in banks:
        pp, nn_ = np.concatenate(pos[v]), np.concatenate(neg[v])
        rec = float((pp > np.quantile(nn_, 0.95)).mean())
        stats[v] = (rec, _auroc(pp, nn_))
        p(f'      [synthval {v:6s}] rec@FPR5={rec:.3f} auroc={stats[v][1]:.3f} (pos={len(pp)} neg={len(nn_)}){" [VAL TRÙNG BANK]" if overlap else ""}')
    sel = max(banks, key=lambda v: stats[v])
    return sel, stats


def run_cat(bb, cat, args, layers, gk, device, p, rng):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không train/good -> bỏ'); return None
    tr_use = tr[:args.max_train] if args.max_train else tr

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    rng.shuffle(bad); rng.shuffle(good)
    if args.max_eval:
        bad = bad[:args.max_eval]; good = good[:args.max_eval]
    idx = bad + good
    sizes = [(Image.open(ds.img_paths[i]).size[1], Image.open(ds.img_paths[i]).size[0]) for i in idx]
    p(f'  [{cat}] eval bad={len(bad)} good={len(good)} | eff_grid={T * gt_} alpha={args.alpha}')

    bank_raw = build_bank(bb, tr_use, T, R, gt_, layers, args.enc_batch, args.bank_size, device)   # (N,C)
    mu = bank_raw.mean(0, keepdim=True)
    sg = bank_raw.std(0, keepdim=True).clamp_min(1e-6)
    bank_std = (bank_raw - mu) / sg
    p(f'    [{cat}] bank={bank_raw.shape[0]} C={bank_raw.shape[1]} -> train adapter...')

    ad_n = train_adapter(bank_std, args.alpha, 0.0, args, device, p, cat)         # control: KHÔNG negative (bản sụp vial/wallplugs)
    ad_m = train_adapter(bank_std, args.alpha, args.lam_s, args, device, p, cat)  # METHOD: nén đuôi + đẩy off-manifold
    with torch.no_grad():
        bank_n = ad_n(bank_std)
        bank_m = ad_m(bank_std)

    banks = {'raw': bank_raw, 'noneg': bank_n, 'method': bank_m}
    VARIANTS = ['raw', 'noneg', 'method']

    # --- SYNTH-VAL selector (fair): val = train/good NGOÀI bank; rng RIÊNG để không lệch shuffle eval ---
    rng_s = np.random.default_rng(args.seed * 1009 + sum(map(ord, cat)))
    va = tr[args.max_train:args.max_train + args.n_val] if args.max_train else []
    overlap = False
    if len(va) < 3:
        va, overlap = tr_use[-args.n_val:], True                       # fallback hiếm: cat ít ảnh train
    sel, sstats = synth_select(bb, cat, va, args, layers, mu, sg, ad_n, ad_m, banks, p, rng_s, overlap)
    raws = {v: [] for v in VARIANTS}
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat} score', leave=False):
            g = img_featgrid(bb, Image.open(ds.img_paths[i]), T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]; C = g.shape[-1]
            q_raw = g.reshape(-1, C).to(device)
            q_std = (q_raw - mu) / sg
            qs = {'raw': q_raw, 'noneg': ad_n(q_std), 'method': ad_m(q_std)}
            for v in VARIANTS:
                raws[v].append(nn_map(qs[v], banks[v], G))
            del g
    del bank_raw, bank_std, bank_n, bank_m; torch.cuda.empty_cache()

    out = {}
    for v in VARIANTS:
        sgn, _ = norm01(raws[v])
        m = eval_sgrids(sgn, sizes, idx, ds, args.canvas, gk, args.aupro_res, args.thr_sigma, device, rng)
        out[v] = m
        db = '' if v == 'raw' else f'   Δ={m["aupro"] - out["raw"]["aupro"]:+.4f}'
        p(f'    [{cat}] {v:6s}: AUPRO0.05={m["aupro"]:.4f}  SegF1={m["segf1"]:.4f}  trần={m["segf1_max"]:.4f}{db}')
    oracle = max(VARIANTS, key=lambda vv: out[vv]['aupro'])
    out['select'] = dict(out[sel]); out['select']['pick'] = sel
    p(f'    [{cat}] select: pick={sel} (oracle={oracle} -> {"TRÚNG" if sel == oracle else "TRẬT"})  '
      f'AUPRO0.05={out[sel]["aupro"]:.4f}  SegF1={out[sel]["segf1"]:.4f}  Δ={out[sel]["aupro"] - out["raw"]["aupro"]:+.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('eval_adapter: learned manifold-tail-compaction adapter (fair, unsup)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split (fair). Thử nhanh: 30')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    # adapter
    ap.add_argument('--alpha', type=float, default=0.1, help='tỉ lệ đuôi nén (adaptA). 1.0 = nén chung')
    ap.add_argument('--hidden', type=int, default=0, help='0 = dim')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam_v', type=float, default=1.0)
    ap.add_argument('--lam_c', type=float, default=1.0)
    ap.add_argument('--lam_s', type=float, default=1.0, help='trọng số đẩy negative off-manifold (0 = tắt)')
    ap.add_argument('--beta_lo', type=float, default=1.0, help='biên độ noise negative (x mean NN-distance)')
    ap.add_argument('--beta_hi', type=float, default=3.0)
    ap.add_argument('--margin', type=float, default=2.0, help='margin tương đối: d(neg) >= margin x mean d_nn')
    ap.add_argument('--neg_mode', type=str, default='rand', choices=['rand', 'adv'],
                    help='rand = hướng ngẫu nhiên; adv = hướng adapter đang co mạnh nhất (GLASS-style)')
    ap.add_argument('--n_val', type=int, default=12, help='số ảnh train/good NGOÀI bank làm synth-val')
    ap.add_argument('--n_per', type=int, default=2, help='số bản synth mỗi ảnh val')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['can', 'sheet_metal', 'fruit_jelly', 'vial', 'fabric', 'rice', 'wallplugs', 'walnuts'])
    ap.add_argument('--out_dir', type=str, default='./adapter')
    args = ap.parse_args()
    if args.hidden == 0:
        args.hidden = None

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('adapter', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(ll / 12 * bb.n_layers))) for ll in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} eff_grid={args.tiles * args.grid_tile} layers={layers} bank={args.bank_size} '
      f'alpha={args.alpha} epochs={args.epochs} lam_s={args.lam_s} beta=[{args.beta_lo},{args.beta_hi}] margin={args.margin} k={args.thr_sigma}')
    p('  FAIR: bank & adapter CHỈ train/good; GT chỉ để chấm. raw=baseline (NN feature gốc).')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p, rng)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được.'); return

    VARIANTS = ['raw', 'noneg', 'method', 'select']
    p('\n' + '=' * 84 + '\n===== MEAN (AUPRO0.05 / SegF1 / trần) — raw=baseline =====')
    base = float(np.mean([res[c]['raw']['aupro'] for c in res]))
    for v in VARIANTS:
        au = float(np.mean([res[c][v]['aupro'] for c in res]))
        f1 = float(np.mean([res[c][v]['segf1'] for c in res]))
        fm = float(np.mean([res[c][v]['segf1_max'] for c in res]))
        db = '' if v == 'raw' else f'   Δ={au - base:+.4f}'
        p(f'  {v:6s}: AUPRO0.05={au:.4f}  SegF1={f1:.4f}  trần={fm:.4f}{db}')
    p('\n  Per-cat (oracle-best theo AUPRO0.05 vs select):')
    hit = 0
    for c in res:
        bv = max(['raw', 'noneg', 'method'], key=lambda vv: res[c][vv]['aupro'])
        r = res[c][bv]
        pk = res[c]['select']['pick']
        hit += int(pk == bv)
        p(f'    [{c:11s}] oracle={bv:6s} AUPRO0.05={r["aupro"]:.4f} (raw={res[c]["raw"]["aupro"]:.3f})  '
          f'SegF1={r["segf1"]:.4f}  | pick={pk:6s} {"TRÚNG" if pk == bv else "TRẬT"}')
    p(f'\n  selector TRÚNG oracle {hit}/{len(res)} cat')
    p('\nĐỌC: select TRÚNG >=6/8 + mean(select) ~ mean(oracle-best) >> raw -> adapter+synth-select SỐNG. '
      'TRẬT vial/wallplugs (chọn nhánh sụp) -> synth mức ảnh không phản ánh defect thật -> kill selector.')


if __name__ == '__main__':
    main()
