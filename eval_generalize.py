# eval_generalize.py
# -----------------------------------------------------------------------------
# TỔNG QUÁT HOÁ NRS: chạy Y HỆT ablation của AD2 (eval_nrs_head) trên VisA và
#   MVTec-AD (bản cũ), CÙNG BỘ SIÊU THAM SỐ, KHÔNG tinh chỉnh gì cho dataset mới.
#   Mọi default ở đây = config uniform đã chốt trên AD2 (grid_tile 48, tiles 3,
#   head_w 0.6, shots 10, seed 0, max_train 60, k 4.5, softpro, layers 2..9).
#   Đổi bất kỳ cái nào cho riêng VisA/MVTec = mất luôn giá trị của thí nghiệm này.
#
# ĐO 2 THỨ TRONG 1 LẦN CHẠY:
#   (1) Đối đầu head grid-GT (NEAREST) vs head NRS (GT native) — như trên AD2.
#   (2) THỐNG KÊ CƠ CHẾ: với lưới hiệu dụng G = tiles*grid_tile, mỗi ô phủ
#       (H/G x W/G) pixel native. Đếm tỉ lệ region defect có diện tích < 1 ô
#       ("dưới-ô") và tỉ lệ region BIẾN MẤT hoàn toàn sau NEAREST về lưới.
#       Đây là biến giải thích: NRS chỉ nên thắng ở đâu tỉ lệ này cao.
#
# ĐỌC (pre-register, chốt TRƯỚC khi chạy):
#   - NRS thắng AUPRO0.05 ở >=70% category của MỖI dataset  -> NRS tổng quát,
#     không phải hiện tượng riêng của AD2. Đây là bằng chứng chính chống lại
#     phản biện "single benchmark".
#   - NRS thắng nhưng chỉ ở nhóm category có tỉ lệ dưới-ô cao (tương quan dương
#     rõ giữa %dưới-ô và ΔAUPRO)                            -> CÒN MẠNH HƠN: cơ chế
#     được xác nhận, không chỉ là con số. Khai đúng như vậy, kèm biểu đồ tán xạ.
#   - NRS ~ ngang (|Δ| < 0.01) trên cả 2 dataset             -> NRS chỉ là hiện tượng
#     của ảnh siêu phân giải AD2 (2448x2048). PHẢI khai giới hạn đó trong paper,
#     đóng khung lại contribution là "cho ảnh phân giải cao", không tổng quát.
#   - NRS THUA rõ ở dataset mới                              -> supervision native
#     hại khi defect vốn đã lớn hơn ô; contribution có điều kiện, phải nói rõ.
#   MỌI KẾT QUẢ ĐỀU DÙNG ĐƯỢC. Không có nhánh nào là "chạy lại cho đẹp".
#
# LƯU Ý PROTOCOL (phải khai trong paper): 10 shot lấy từ test split có GT, giống
#   hệt protocol đã dùng trên AD2 -> số báo cáo là trên test\shots (held-out).
#
#   python eval_generalize.py --dataset visa  --data_path ../data_visa  --out_dir ./gen_visa
#   python eval_generalize.py --dataset mvtec --data_path ../data_mvtec --out_dir ./gen_mvtec
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
from scipy.ndimage import label as cc_label

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D):
    if os.path.exists(os.path.join(_D, 'infer_submit_mvtec_ad2.py')):
        sys.path.insert(0, _D)
        break
    _D = os.path.dirname(_D)
from infer_submit_mvtec_ad2 import (                                   # noqa: E402
    build_bank, build_head, img_featgrid, nn_map, gt_grid, IMG_EXT,
)
from eval_nrs_head import build_nrs_head                                # noqa: E402
from eval_native import Hist                                            # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
# MVTec-AD cũ + VisA báo cáo AUPRO@0.30 (PRO gốc 2019) + P-AUROC + I-AUROC.
# AUPRO@0.05 giữ lại để nhất quán nội bộ với AD2. Tính TẤT CẢ trong 1 lần gọi.
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def eval_metrics(preds, gts):
    """Trả dict thước pixel/ảnh trên tập ảnh AUPRO (paired). Thước baseline chính = aupro30."""
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    r = ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp, use_metrics=METRIC_NAMES)
    g = lambda n: float(r[METRIC_NAMES.index(n)])
    return {'aupro30': g('AUPRO0.30'), 'aupro05': g('AUPRO0.05'),
            'p_auroc': g('P-AUROC'), 'p_ap': g('P-AP'), 'i_auroc': g('I-AUROC')}

CATS = {
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
              'wood', 'zipper'],
}
GT_EXT = ('.png', '.PNG', '.jpg', '.JPG', '.bmp')


def find_gt(gt_dir, stem):
    """Khớp mask: MVTec-AD cũ = <stem>_mask.png, VisA = <stem>.png. Thử cả hai."""
    if not os.path.isdir(gt_dir):
        return None
    for base in (stem + '_mask', stem):
        for e in GT_EXT:
            pth = os.path.join(gt_dir, base + e)
            if os.path.exists(pth):
                return pth
    return None


class SimpleADDataset:
    """Layout chuẩn MVTec-AD cũ / VisA_pytorch: root/test/<type>/ + root/ground_truth/<type>/.
    'good' -> label 0. Chỉ phơi ra đúng 3 field mà build_head / build_nrs_head cần."""

    def __init__(self, root):
        self.root = root
        img_paths, gt_paths, labels, types = [], [], [], []
        test_dir = os.path.join(root, 'test')
        gt_root = os.path.join(root, 'ground_truth')
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f'không thấy {test_dir}')
        for dtype in sorted(os.listdir(test_dir)):
            d = os.path.join(test_dir, dtype)
            if not os.path.isdir(d):
                continue
            files = sorted(sum([glob.glob(os.path.join(d, e)) for e in IMG_EXT], []))
            if dtype == 'good':
                img_paths += files
                gt_paths += [None] * len(files)
                labels += [0] * len(files)
                types += ['good'] * len(files)
                continue
            gt_dir = os.path.join(gt_root, dtype)
            matched = [find_gt(gt_dir, os.path.splitext(os.path.basename(f))[0]) for f in files]
            if any(m is None for m in matched):                          # dự phòng: ghép theo thứ tự
                gtf = sorted(sum([glob.glob(os.path.join(gt_dir, '*' + e)) for e in GT_EXT], []))
                if len(gtf) == len(files):
                    matched = gtf
            keep = [(f, m) for f, m in zip(files, matched) if m is not None]
            img_paths += [f for f, _ in keep]
            gt_paths += [m for _, m in keep]
            labels += [1] * len(keep)
            types += [dtype] * len(keep)
        self.img_paths = np.array(img_paths, dtype=object)
        self.gt_paths = np.array(gt_paths, dtype=object)
        self.labels = np.array(labels)
        self.types = np.array(types)

    def __len__(self):
        return len(self.img_paths)


def subcell_stats(ds, bad_idx, G, max_img=60):
    """Với lưới G: %region defect nhỏ hơn 1 ô, và %region BIẾN MẤT sau NEAREST.
    Đây là biến giải thích cho ΔNRS — đo trên GT gốc, không cần model."""
    n_reg = n_sub = n_gone = 0
    areas_cell = []
    for i in bad_idx[:max_img]:
        gp = ds.gt_paths[i]
        if not (isinstance(gp, str) and os.path.exists(gp)):
            continue
        gpil = Image.open(gp).convert('L')
        W, H = gpil.size
        gm = np.asarray(gpil) > 127
        if not gm.any():
            continue
        cell = (H / G) * (W / G)                                          # px native mỗi ô lưới
        lab, n = cc_label(gm)
        gnear = np.asarray(gpil.resize((G, G), Image.NEAREST)) > 127      # đúng phép của gt_grid
        for rid in range(1, n + 1):
            m = lab == rid
            a = int(m.sum())
            n_reg += 1
            areas_cell.append(a / cell)
            if a < cell:
                n_sub += 1
            ys, xs = np.nonzero(m)                                        # region còn sống trên lưới?
            gy = np.clip((ys / H * G).astype(int), 0, G - 1)
            gx = np.clip((xs / W * G).astype(int), 0, G - 1)
            if not gnear[gy, gx].any():
                n_gone += 1
    if n_reg == 0:
        return {'n_reg': 0, 'p_sub': 0.0, 'p_gone': 0.0, 'med_cell': 0.0}
    return {'n_reg': n_reg, 'p_sub': n_sub / n_reg, 'p_gone': n_gone / n_reg,
            'med_cell': float(np.median(areas_cell))}


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    root = os.path.join(args.data_path, cat)
    tr = sorted(sum([glob.glob(os.path.join(root, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        p(f'  [{cat}] không có train/good -> bỏ')
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    try:
        ds = SimpleADDataset(root)
    except FileNotFoundError as e:
        p(f'  [{cat}] {e} -> bỏ')
        return None
    bad = [i for i in range(len(ds)) if ds.labels[i] == 1]
    good = [i for i in range(len(ds)) if ds.labels[i] == 0]
    if len(bad) < args.shots + 5:
        p(f'  [{cat}] chỉ {len(bad)} ảnh bad -> không đủ {args.shots} shot + eval, bỏ')
        return None
    rng.shuffle(bad)
    shots = bad[:args.shots]

    Gg = T * gt_
    st = subcell_stats(ds, bad, Gg)
    p(f'  [{cat}] {len(tr)} train | test bad={len(bad)} good={len(good)} | lưới {Gg} '
      f'| region={st["n_reg"]} dưới-ô={st["p_sub"]:.1%} mất-sau-NEAREST={st["p_gone"]:.1%} '
      f'| median area={st["med_cell"]:.2f} ô')

    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    grid_px = sum(int(gt_grid(ds.gt_paths[i], 1, Gg).sum()) for i in shots)
    # THỨ TỰ SEED Y HỆT eval_nrs_head (manual_seed -> grid -> nrs)
    torch.manual_seed(args.seed)
    headA = build_head(bb, ds, shots, bank, args, layers, device)
    headB = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
    p(f'    supervision grid-GT: {grid_px} px | head grid: '
      f'{"None -> dist-only" if headA is None else "OK"} | NRS: {"None" if headB is None else "OK"}')
    if headB is None:
        p(f'  [{cat}] NRS head None -> bỏ')
        return None

    idx_bad = bad[args.shots:]
    idx_good = good
    if args.max_eval:
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good
    m = max(2, args.aupro_max) // 2
    ap_ids = set(idx_bad[:m] + idx_good[:m]) if args.aupro_max else set(idx)

    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))
            g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
            G = g.shape[0]
            d = np.asarray(nn_map(g, bank, device))
            flat = g.reshape(-1, C)
            prA = torch.sigmoid(headA(flat)).reshape(G, G).cpu().numpy() if headA is not None else None
            prB = torch.sigmoid(headB(flat)).reshape(G, G).cpu().numpy()
            grids.append((d, prA, prB))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _, _ in grids]), [1, 99])

    def fuse(d, pr):
        dr = (d - lo) / (hi - lo + 1e-8)
        return dr.astype(np.float32) if pr is None else ((1 - hw) * dr + hw * pr).astype(np.float32)

    out = {'stats': st}
    for name, k in [('grid', 1), ('nrs', 2)]:
        s_grids = [fuse(gg[0], gg[k]) for gg in grids]
        gmin = min(float(s.min()) for s in s_grids)
        gmax = max(float(s.max()) for s in s_grids)
        h = Hist(gmin - 0.05, gmax + 0.05)
        ap_preds, ap_gts = [], []
        for s, (H, W), i in zip(s_grids, sizes, idx):
            with torch.no_grad():
                t = torch.tensor(s, device=device)[None, None].float()
                t = F.interpolate(t, size=256, mode='bilinear', align_corners=False)
                t = gk(t)
                mnat = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                if i in ap_ids:
                    m512 = F.interpolate(t, size=(args.aupro_res, args.aupro_res),
                                         mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            if ds.labels[i] == 0:
                gnat = np.zeros(mnat.size, np.uint8)
                g512 = np.zeros((args.aupro_res, args.aupro_res), np.uint8)
            else:
                gpil = Image.open(ds.gt_paths[i]).convert('L')
                gnat = (np.asarray(gpil) > 127).astype(np.uint8).reshape(-1)
                g512 = (np.asarray(gpil.resize((args.aupro_res, args.aupro_res), Image.BOX)) > 0).astype(np.uint8)
            h.add(mnat.reshape(-1), gnat)
            if i in ap_ids:
                ap_preds.append(m512)
                ap_gts.append(g512)
        tag = ' (dist-only)' if (name == 'grid' and headA is None) else ''
        mt = eval_metrics(ap_preds, ap_gts)
        out[name] = {'f1': h.f1_at(h.ksig(args.thr_sigma)), 'f1max': h.f1_max(), 'tag': tag, **mt}
        p(f'    [{cat}] head={name:4s}{tag}: F1@ksig={out[name]["f1"]:.4f} trần={out[name]["f1max"]:.4f} '
          f'| AUPRO0.30={mt["aupro30"]:.4f} AUPRO0.05={mt["aupro05"]:.4f} '
          f'P-AUROC={mt["p_auroc"]:.4f} I-AUROC={mt["i_auroc"]:.4f} ({len(ap_preds)}img)')
    return out


def main():
    ap = argparse.ArgumentParser('NRS tổng quát hoá: VisA / MVTec-AD, CÙNG hyperparam với AD2')
    ap.add_argument('--dataset', type=str, required=True, choices=['visa', 'mvtec'])
    ap.add_argument('--data_path', type=str, required=True, help='thư mục chứa các category')
    # ==== TỪ ĐÂY XUỐNG: default = config uniform đã chốt trên AD2. ĐỪNG ĐỔI. ====
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=48)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
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
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    # ===========================================================================
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split')
    ap.add_argument('--aupro_max', type=int, default=120)
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=None)
    ap.add_argument('--out_dir', type=str, default='./gen')
    args = ap.parse_args()

    cats = args.categories or CATS[args.dataset]
    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('gen_' + args.dataset, args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p('=' * 100)
    p(f'TỔNG QUÁT HOÁ NRS | dataset={args.dataset} ({len(cats)} cat) | CONFIG AD2 NGUYÊN SI: '
      f'grid={args.tiles}:{args.grid_tile} shots={args.shots} head_w={args.head_w} k={args.thr_sigma} '
      f'loss={args.loss} seed={args.seed}')
    p('=' * 100)

    res = {}
    for cat in cats:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        p('không category nào chạy được — kiểm tra --data_path/layout')
        return

    p('\n' + '=' * 100)
    p(f'===== TỔNG {args.dataset} ({len(res)} cat) | thước native, full test trừ shots =====')
    p(f'  (AUPRO0.30 = thước baseline MVTec/VisA; AUPRO0.05 = nhất quán với AD2)')
    p(f'{"cat":13s} {"dưới-ô":>7s} {"mất":>6s} | {"AUPRO30 g":>9s} {"nrs":>7s} {"Δ30":>7s} | '
      f'{"AUPRO05 g":>9s} {"nrs":>7s} | {"P-AUR g":>8s} {"nrs":>7s} | {"F1 g":>6s} {"nrs":>6s}')
    for cat, r in res.items():
        g, nr = r['grid'], r['nrs']
        p(f'{cat:13s} {r["stats"]["p_sub"]:7.1%} {r["stats"]["p_gone"]:6.1%} | '
          f'{g["aupro30"]:9.4f} {nr["aupro30"]:7.4f} {nr["aupro30"] - g["aupro30"]:+7.4f} | '
          f'{g["aupro05"]:9.4f} {nr["aupro05"]:7.4f} | '
          f'{g["p_auroc"]:8.4f} {nr["p_auroc"]:7.4f} | '
          f'{g["f1"]:6.3f} {nr["f1"]:6.3f}{g["tag"]}')
    for key, lab in [('aupro30', 'AUPRO0.30'), ('aupro05', 'AUPRO0.05'), ('p_auroc', 'P-AUROC'),
                     ('i_auroc', 'I-AUROC'), ('f1', 'F1@ksig'), ('f1max', 'trần')]:
        a = float(np.mean([res[c]['grid'][key] for c in res]))
        b = float(np.mean([res[c]['nrs'][key] for c in res]))
        w = sum(1 for c in res if res[c]['nrs'][key] > res[c]['grid'][key])
        p(f'  MEAN {lab:9s}: grid={a:.4f}  nrs={b:.4f}  Δ={b - a:+.4f}  | NRS thắng {w}/{len(res)}')

    # tương quan cơ chế: NRS có nên chỉ thắng ở nơi defect dưới-ô nhiều không? (dùng thước báo cáo @0.30)
    xs = np.array([res[c]['stats']['p_sub'] for c in res])
    ys = np.array([res[c]['nrs']['aupro30'] - res[c]['grid']['aupro30'] for c in res])
    if len(res) >= 3 and xs.std() > 1e-6 and ys.std() > 1e-6:
        p(f'  TƯƠNG QUAN %dưới-ô vs ΔAUPRO0.30: pearson r={float(np.corrcoef(xs, ys)[0, 1]):+.3f} (n={len(res)})')

    p('\nĐỌC (pre-registered, KHÔNG sửa sau khi thấy số):')
    p('  - NRS thắng AUPRO >=70% cat  -> tổng quát hoá, dùng chống "single benchmark".')
    p('  - r dương rõ (>=+0.4)        -> cơ chế dưới-ô được xác nhận, mạnh hơn cả con số.')
    p('  - |Δ| < 0.01 cả 2 dataset    -> NRS chỉ đúng cho ảnh siêu phân giải; KHAI giới hạn.')
    p('  - NRS thua rõ                -> contribution có điều kiện; khai rõ điều kiện đó.')


if __name__ == '__main__':
    main()
