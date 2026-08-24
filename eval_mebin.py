# eval_mebin.py
# -----------------------------------------------------------------------------
# OFFLINE (grid cache ./fill) -- NGUONG THICH UNG: MEBin (literature) vs SSE (cua ta).
#
# DONG CO (do duoc, khong doan): tren chinh map hien tai, khoang cach
#   F1@rule -> TRAN (nguong oracle toan cat) = +0.150 mean, don vao
#   wallplugs (+0.389) va fabric (+0.317) -- hai cat ganh 59% khoang thua voi top1.
#   => tang chon nguong dang vut di nhieu hon moi lever ta tung thu.
#
# V0 (control, nen dong bang) = closing(k=2r+1) -> fill_holes, nguong = thr(rule).
#     PHAI tai tao 0.3748. Lech > 0.001 la cache/chain sai, DUNG.
#
# V1 (MEBin, AnomalyNCD/RoBiS): MOT nguong cho ca anh, chon bang "cao nguyen" cua
#     SO LUONG vung lien thong khi quet nguong. Day la baseline literature NGUYEN BAN.
#     Luu y do duoc tren test tong hop: MEBin khong co trang thai "khong co loi" --
#     tren anh good sach no to kin ~100% anh. Nen bao cao no de trung thuc, KHONG
#     dung no lam moc so sanh cua novelty.
#
# V1g (BASELINE CONG BANG cua ablation) = MEBin + DUNG cong neo-dinh cua V2.
#     Moi thu giong V2 tru mot diem: V1g chon MOT muc cho ca anh (theo so vung),
#     V2 chon MOT muc cho TUNG vung (theo on dinh dien tich).
#     => d(V2 - V1g) la dong gop rieng cua novelty, khong lan thu gi khac.
#
# V2 (SSE = Stable-Extent Selection, DE XUAT CUA TA):
#     tach doi quyet dinh nhi phan hoa --
#       (a) CO loi hay khong: van do luat fair cu (thr = p95(heldout) x 1.15) quyet.
#           Chi nhung vung co DINH >= thr moi duoc xet. => KHONG sinh detection moi,
#           precision cua nhanh cu duoc bao toan theo thiet ke.
#       (b) Loi RONG toi dau: moi vung tu chon muc cua minh theo DO ON DINH dien
#           tich (tieu chi MSER): s(z) = (A(z-d) - A(z+d)) / A(z), chon z* = argmin s.
#           Vung lon-yeu (mieng va fabric, con no) no xuong duoi thr; vung nho-manh
#           giu nguyen hoac co len. MOT nguong chung khong lam duoc ca hai.
#     Khac MEBin o cho: MEBin chon 1 nguong/ANH theo SO LUONG vung; SSE chon
#     1 nguong/VUNG theo DO ON DINH cua chinh vung do, va neo vao quyet dinh
#     phat hien cua luat fair.
#
# FAIR: moi nguong suy tu map + z-stats heldout train/good. Khong GT, khong nhan cat,
#   mot bo hang so cho ca 8 cat (va cho dataset khac). Hang so dong bang TRUOC khi chay:
#     Z_UP=2.0  Z_DOWN=4.0  Z_STEP=0.25  (luoi muc, don vi sigma quanh thr)
#     D_STAB=2 muc (=0.5 sigma)  MIN_AREA=100px  MAX_FRAC=0.35
#   KHONG sweep hang so nao sau khi thay so.
#
# DOC (pre-register, chot TRUOC khi chay):
#   0) V0 == 0.3748 (+-0.001). Khong khop -> dung, khong doc gi them.
#   1) Bao cao per-cat + %(khoang rule->tran) lay lai duoc. TRAN do ngay trong run nay.
#   2) V2 VAO NEN neu: mean F1 >= V0 + 0.060 (=40% cua khoang 0.150 da do)
#      VA tong phan TUT < 1/2 tong phan TANG (luat khoi luong -- KHONG dung veto
#      per-cat kieu cu, vi khoi luong metric don vao vai chuc component lon).
#   3) Ablation: d(V1g - V0) = loi cua "nguong thich ung" noi chung;
#      d(V2 - V1g) = loi RIENG cua novelty (per-VUNG thay vi per-anh). Bao cao ca hai.
#      Neu d(V2 - V1g) <= +0.010 thi novelty KHONG dung duoc: MEBin+cong da du,
#      phai noi thang the trong paper thay vi ban SSE.
#   4) Truot -> KHONG tune hang so. Doc lai co che tu per-cat roi moi de xuat tiep.
#
#   python eval_mebin.py --data_path ../data --cache_dir ./fill --out_dir ./mebin
# -----------------------------------------------------------------------------
import os
import sys
import argparse
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes, label as cc_label, maximum_position

_D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _D)
from eval_fairthr import closing                                                   # noqa: E402
from eval_native import Hist, make_map                                             # noqa: E402
from eval_guidedup import load_gray                                                # noqa: E402
from eval_fullscale import SCALES, fuse2, up_grid, guided1                         # noqa: E402
from dataset import MVTecAD2Dataset                                                # noqa: E402
from utils import get_gaussian_kernel, get_logger                                  # noqa: E402

warnings.filterwarnings('ignore')

Z_UP, Z_DOWN, Z_STEP = 2.0, 4.0, 0.25     # luoi muc quanh thr, don vi sigma. DONG BANG.
D_STAB = 2                                # nua cua so on dinh = 2 muc = 0.5 sigma
MIN_AREA = 100                            # px native
MAX_FRAC = 0.35                           # 1 vung khong duoc chiem > 35% anh
VARIANTS = ['V0', 'V1', 'V1g', 'V2']


def gate_peak(nat, mask, thr):
    """Giu lai chi cac vung lien thong co DINH >= thr (cong neo-dinh, dung chung
       cho V1g va V2). Khong co no, MEBin nguyen ban to kin ca anh good."""
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    mx = np.array(maximum_position(nat, lab, index=list(range(1, n + 1))), ndmin=2)
    peak = nat[mx[:, 0], mx[:, 1]]
    want = np.nonzero(peak >= thr)[0] + 1
    return np.isin(lab, want) if want.size else np.zeros_like(mask)


def levels_of(thr, sd):
    """Luoi muc giam dan quanh thr (don vi sigma cua scale fine)."""
    ks = np.arange(Z_UP, -Z_DOWN - 1e-9, -Z_STEP)
    return thr + ks * sd


def scan_levels(nat, lvls):
    """Quet muc MOT lan. Tra:
       chains: dict seed_pos -> list (li, area)  (li = chi so muc, giam dan)
       counts: so vung >= MIN_AREA theo tung muc (cho MEBin).
    Danh tinh mot vung = vi tri pixel cuc dai cua no (ngu nghia max-tree: khi hai vung
    nhap lai, vung manh hon giu chuoi, vung yeu dung chuoi)."""
    chains, counts = {}, []
    for li, lv in enumerate(lvls):
        lab, n = cc_label(nat >= lv)
        if n == 0:
            counts.append(0)
            continue
        areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        keep = np.nonzero(areas >= MIN_AREA)[0] + 1        # nhan label 1-based
        counts.append(int(keep.size))
        if keep.size == 0:
            continue
        pos = maximum_position(nat, lab, index=list(keep))
        if not isinstance(pos, list):        # scipy tra tuple don khi chi 1 index
            pos = [pos]
        for lb, pp in zip(keep, pos):
            chains.setdefault(tuple(int(x) for x in pp), []).append((li, int(areas[lb - 1])))
    return chains, counts


def mebin_level(counts):
    """MEBin: doan chay dai nhat ma SO vung khong doi (>=1); lay muc THAP NHAT cua doan
       (bao trum nhat). Khong co doan nao -> None."""
    best_len, best_end, i = 0, None, 0
    while i < len(counts):
        j = i
        while j + 1 < len(counts) and counts[j + 1] == counts[i]:
            j += 1
        if counts[i] >= 1 and (j - i + 1) > best_len:
            best_len, best_end = j - i + 1, j
        i = j + 1
    return best_end


def sse_select(chains, i_thr, max_area):
    """Moi chuoi (vung) chon muc rieng theo do on dinh dien tich (MSER).
       Chi xet chuoi co DINH >= thr (xuat hien o muc chi so <= i_thr).
       Tra dict li -> set(seed_pos)."""
    sel = {}
    for seed, ent in chains.items():
        if ent[0][0] > i_thr:                 # dinh nam duoi thr -> KHONG phai detection
            continue
        li = np.array([e[0] for e in ent])
        ar = np.array([e[1] for e in ent], np.float64)
        ok = ar <= max_area
        if not ok.any():
            continue
        if len(ent) >= 2 * D_STAB + 1:
            s = np.full(len(ent), np.inf)
            for i in range(D_STAB, len(ent) - D_STAB):
                s[i] = (ar[i + D_STAB] - ar[i - D_STAB]) / max(ar[i], 1.0)
            s[~ok] = np.inf
            # HOA -> lay muc THAP nhat (extent LON nhat): "maximally stable" hieu theo
            # nghia cuc dai. Quy uoc dong bang truoc khi cham du lieu that; khop chan
            # doan do duoc (precision thua, recall thieu tren vung lon).
            k = len(s) - 1 - int(np.argmin(s[::-1]))
            if not np.isfinite(s[k]):
                k = int(np.nonzero(ok)[0][-1])
        else:
            k = int(np.nonzero(ok)[0][-1])    # chuoi ngan -> muc thap nhat hop le
        if ar[k] < MIN_AREA:
            continue
        sel.setdefault(int(li[k]), set()).add(seed)
    return sel


def build_mask(nat, lvls, sel):
    """Hop cac vung da chon, moi vung o muc rieng cua no."""
    out = np.zeros(nat.shape, bool)
    for li, seeds in sel.items():
        lab, n = cc_label(nat >= lvls[li])
        if n == 0:
            continue
        want = sorted({int(lab[s]) for s in seeds if lab[s] > 0})
        if want:
            out |= np.isin(lab, want)
    return out


def run_cat(cat, args, gk, device, p):
    G3 = SCALES[0][0] * SCALES[0][1]
    z = np.load(os.path.join(args.cache_dir, f'grids_{cat}.npz'), allow_pickle=True)
    meta = np.load(os.path.join(args.cache_dir, f'meta_{cat}.npz'))
    st = [(float(meta['st'][i][0]), float(meta['st'][i][1])) for i in range(2)]
    thr = float(meta['thr']); sd = st[0][1]
    te_f, te_c = z['te_fine'], z['te_ctx']
    paths = [str(x) for x in z['paths']]
    labels = [int(x) for x in z['labels']]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    gt_of = dict(zip(ds.img_paths, ds.gt_paths))
    lvls = levels_of(thr, sd)
    i_thr = int(np.argmin(np.abs(lvls - thr)))
    p(f'  [{cat}] n={len(paths)} thr={thr:.3f} sd={sd:.3f} | {len(lvls)} muc, i_thr={i_thr}')

    mst = {v: np.zeros(3, np.float64) for v in VARIANTS}
    h = None
    for k in tqdm(range(len(paths)), ncols=70, desc=f'    {cat}', leave=False):
        pil = Image.open(paths[k])
        W, H = pil.size
        fused = fuse2(te_f[k], up_grid(te_c[k], G3, device), st)
        nat_t = make_map(fused['maxz'], args.canvas, gk, (H, W), device)
        nat_t = guided1(nat_t, load_gray(pil, device), max(1, round(min(H, W) / G3)))
        r = max(1, round(min(H, W) / G3))
        m0 = closing(nat_t > thr, 2 * r + 1).cpu().numpy().astype(bool)
        nat = nat_t.cpu().numpy().astype(np.float32)
        del nat_t
        gt = (np.zeros((H, W), bool) if labels[k] == 0
              else np.asarray(Image.open(gt_of[paths[k]]).convert('L')) > 127)
        if h is None:
            h = Hist(float(nat.min()) - 0.5, thr + (Z_UP + 4.0) * sd)
        h.add(nat.reshape(-1), gt.reshape(-1).astype(np.uint8))

        chains, counts = scan_levels(nat, lvls)
        li1 = mebin_level(counts)
        m1 = np.zeros((H, W), bool) if li1 is None else (nat >= lvls[li1])
        m1g = gate_peak(nat, m1, thr) if m1.any() else m1
        sel = sse_select(chains, i_thr, MAX_FRAC * H * W)
        m2 = build_mask(nat, lvls, sel)
        del nat, chains

        for v, m in (('V0', m0), ('V1', m1), ('V1g', m1g), ('V2', m2)):
            if v != 'V0':
                m = closing(torch.from_numpy(m).to(device), 2 * r + 1).cpu().numpy().astype(bool)
            pd = binary_fill_holes(m)
            mst[v] += ((pd & gt).sum(), (pd & ~gt).sum(), ((~pd) & gt).sum())
        del m0, m1, m1g, m2

    out = {v: float(2 * mst[v][0] / (2 * mst[v][0] + mst[v][1] + mst[v][2] + 1e-9)) for v in VARIANTS}
    out['tran'] = float(h.f1_max())
    gap = out['tran'] - out['V0']
    p(f"    [{cat}] V0={out['V0']:.4f}  V1={out['V1']:.4f}  V1g={out['V1g']:.4f}  V2(SSE)={out['V2']:.4f}"
      f"  | tran={out['tran']:.4f} khoang={gap:+.4f}"
      f"  lay lai V2={100 * (out['V2'] - out['V0']) / (gap + 1e-9):.0f}%")
    return out


def main():
    ap = argparse.ArgumentParser('eval_mebin: nguong thich ung -- MEBin (baseline) vs SSE (de xuat)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--cache_dir', type=str, default='./fill')
    ap.add_argument('--canvas', type=int, default=256)
    ap.add_argument('--categories', type=str, nargs='+',
                    default=['fabric', 'wallplugs', 'vial', 'sheet_metal',
                             'fruit_jelly', 'rice', 'walnuts', 'can'])
    ap.add_argument('--out_dir', type=str, default='./mebin')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('mebin', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} | V0 control(=0.3748) | V1 MEBin 1-nguong/anh theo so vung | '
      f'V2 SSE 1-nguong/VUNG theo on dinh dien tich, neo dinh >= thr. '
      f'Hang so dong bang: Z_UP={Z_UP} Z_DOWN={Z_DOWN} Z_STEP={Z_STEP} D={D_STAB} '
      f'MIN_AREA={MIN_AREA} MAX_FRAC={MAX_FRAC}. Khong GT, khong per-cat.')

    res = {}
    for cat in args.categories:
        if not os.path.exists(os.path.join(args.cache_dir, f'grids_{cat}.npz')):
            p(f'  [{cat}] KHONG co cache -> bo'); continue
        res[cat] = run_cat(cat, args, gk, device, p)
    if not res:
        p('khong cache nao.'); return

    p('\n' + '=' * 88 + '\n===== MEAN (FULL test_public, offline cache) =====')
    m = {v: float(np.mean([res[c][v] for c in res])) for v in VARIANTS + ['tran']}
    p(f"  V0={m['V0']:.4f}  V1(MEBin nguyen ban)={m['V1']:.4f}  V1g(MEBin+cong)={m['V1g']:.4f}  "
      f"V2(SSE)={m['V2']:.4f}  tran={m['tran']:.4f}")
    p(f"  d(V1g-V0)={m['V1g'] - m['V0']:+.4f}  d(V2-V0)={m['V2'] - m['V0']:+.4f}  "
      f"NOVELTY d(V2-V1g)={m['V2'] - m['V1g']:+.4f}  (moi thu khac giong het nhau)")
    d = [(c, res[c]['V2'] - res[c]['V0']) for c in res]
    up = sum(x for _, x in d if x > 0); dn = -sum(x for _, x in d if x < 0)
    p(f"  tong tang={up:+.4f}  tong tut={-dn:+.4f}  (luat khoi luong: tut phai < {up / 2:.4f})")
    p(f"  per-cat d(V2-V0): {[(c, round(x, 4)) for c, x in sorted(d, key=lambda t: -t[1])]}")
    ok = (m['V2'] >= m['V0'] + 0.060) and (dn < up / 2)
    p("\nDOC (pre-registered): V0 phai=0.3748. V2 VAO NEN neu mean >= V0+0.060 VA tut < 1/2 tang."
      f"  => {'VAO NEN' if ok else 'TRUOT -- khong tune hang so, doc lai co che per-cat'}")


if __name__ == '__main__':
    main()
