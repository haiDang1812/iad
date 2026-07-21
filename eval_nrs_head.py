# eval_nrs_head.py
# -----------------------------------------------------------------------------
# NRS — Native-Resolution Supervision cho few-shot head. PREMISE TEST (đối đầu).
#
# Cơ chế giết can (đã đo): supervision hiện tại đi qua gt_grid = GT native ->
#   NEAREST về lưới G. Defect nhỏ hơn 1 ô (can: min 5px, >20% <17px linear) BIẾN MẤT
#   khỏi nhãn -> <3 px defect -> build_head trả None -> head không tồn tại.
#   Grid 144 cứu một phần (native348: can F1 0.20) nhưng cây thước train vẫn là
#   lưới — mọi defect < 17px vẫn bị NEAREST nuốt hoặc gán trọng số sai.
#
# NRS: GIỮ GT Ở NATIVE. Về nguyên tắc: upsample logit head (bilinear, khả vi) lên
#   native rồi tính SoftPRO/BCE trên pixel native -> defect dưới-ô đóng góp gradient
#   qua trọng số nội suy của 4 ô lân cận. Vì head TUYẾN TÍNH và trọng số bilinear
#   cộng bằng 1: logit-nội-suy(pixel) == head(feature-nội-suy(pixel)). Nên cài đặt
#   = grid_sample feature tại đúng tọa độ pixel native, KHÔNG cần backward qua map
#   2448x2048. Region của SoftPRO = connected-component trên GT NATIVE (region nhỏ
#   giờ TỒN TẠI và được cân bằng trọng số như region to — đúng hình dáng AUPRO).
#
# Đối đầu CÙNG bank / CÙNG shots / CÙNG pipeline eval (chỉ khác supervision):
#   A = head hiện tại (build_head, gt_grid NEAREST)   [can@3:24: None -> dist-only]
#   B = head NRS      (build_nrs_head, GT native)
#   Đo full split test_public, thước native như server (eval_native):
#     SegF1@ksig(native), SegF1_max(native), AUPRO0.05@512 (subset PAIRED).
#
# ĐỌC (pre-register): NRS thắng head cũ >= +0.03 F1 native trên can -> premise SỐNG,
#   nấu vào pipeline + thành novelty chính của paper. NRS ~ head cũ (±0.01) -> GIẾT
#   trong buổi này, không tốn tuần.
#
#   python eval_nrs_head.py --data_path ../data --out_dir ./nrs48 \
#       --tiles 3 --grid_tile 48 --categories can wallplugs fruit_jelly
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
    build_bank, build_head, img_featgrid, nn_map, gt_grid,
    train_softpro, train_bce, Head, subsample, VALID, IMG_EXT,
)
from eval_native import Hist                                            # noqa: E402
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger, ader_evaluator       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
METRIC_NAMES = ['I-AUROC', 'I-AP', 'I-F1_max', 'P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO', 'AUPRO0.05', 'AUPRO0.30']


def aupro05(preds, gts):
    sp = np.array([float(p.max()) for p in preds])
    if float(sp.max() - sp.min()) < 1e-9:
        sp = sp + np.random.default_rng(0).normal(0, 1e-6, sp.shape)
    gt_sp = np.array([1 if g.sum() > 0 else 0 for g in gts])
    return ader_evaluator(np.stack(preds), sp, np.stack(gts), gt_sp,
                          use_metrics=METRIC_NAMES)[METRIC_NAMES.index('AUPRO0.05')]


def feats_at(inp, ys, xs, H, W):
    """Feature nội suy bilinear tại pixel native (ys, xs). inp: (1,C,G,G) trên device.

    Trùng khớp toán học với F.interpolate(align_corners=False) của lưới lên (H,W):
    normalized = 2*(p+0.5)/S - 1. Head tuyến tính => head(feats_at(...)) chính là
    logit-map-upsample đọc tại pixel đó — NRS không cần map native nào trong train."""
    device = inp.device
    gx = torch.as_tensor((xs + 0.5) / W * 2.0 - 1.0, dtype=torch.float32, device=device)
    gy = torch.as_tensor((ys + 0.5) / H * 2.0 - 1.0, dtype=torch.float32, device=device)
    grid = torch.stack([gx, gy], -1)[None, None]                       # (1,1,N,2) thứ tự (x,y)
    out = F.grid_sample(inp, grid, mode='bilinear', align_corners=False, padding_mode='border')
    return out[0, :, 0, :].T                                           # (N,C)


def build_nrs_head(bb, ds, shot_pool, bank, args, layers, device, p):
    """Bản sao build_head nhưng supervision Ở NATIVE. Khác biệt DUY NHẤT so với head cũ:
    (1) pos = feature nội suy tại pixel defect NATIVE (không phải ô lưới NEAREST);
    (2) region id = cc trên GT NATIVE; (3) thêm neg trong-ảnh (pixel normal native của
    chính shot — cho head thấy đúng phân bố feature-nội-suy ở cả hai phía biên)."""
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    C = bank.shape[-1]
    rng = np.random.default_rng(args.seed + 1)
    per_img_neg = max(1, args.n_neg_img // max(1, len(shot_pool)))
    pos_f, pos_r, neg_f = [], [], []
    rbase = 0
    tot_px = 0
    for i in shot_pool:
        if not (isinstance(ds.gt_paths[i], str) and os.path.exists(ds.gt_paths[i])):
            continue
        pil = Image.open(ds.img_paths[i])
        W, H = pil.size
        gm = np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127
        g = img_featgrid(bb, pil, T, R, gt_, layers, args.enc_batch)
        inp = g.permute(2, 0, 1).contiguous()[None]
        del g
        lab, n = cc_label(gm)
        for rid in range(1, n + 1):
            ys, xs = np.nonzero(lab == rid)
            if len(ys) > args.pos_per_region:
                s = rng.choice(len(ys), args.pos_per_region, replace=False)
                ys, xs = ys[s], xs[s]
            with torch.no_grad():
                pos_f.append(feats_at(inp, ys, xs, H, W).cpu())
            pos_r.append(np.full(len(ys), rbase))
            rbase += 1
            tot_px += len(ys)
        nys, nxs = np.nonzero(~gm)
        s = rng.choice(len(nys), min(per_img_neg, len(nys)), replace=False)
        with torch.no_grad():
            neg_f.append(feats_at(inp, nys[s], nxs[s], H, W).cpu())
        del inp
    if rbase < 1 or tot_px < 1:
        return None
    Xpos = torch.cat(pos_f, 0)
    rids = np.concatenate(pos_r)
    # cap tổng pos (VRAM) — subsample nhưng GIỮ >=5 px/region: region nhỏ (chính đối
    # tượng NRS cứu) không bao giờ bị cap nuốt.
    if len(Xpos) > args.max_pos:
        keep = []
        for r in np.unique(rids):
            ii = np.nonzero(rids == r)[0]
            keep.append(ii if len(ii) <= 5 else rng.choice(ii, 5, replace=False))
        keep = np.unique(np.concatenate(keep))
        rest = np.setdiff1d(np.arange(len(Xpos)), keep)
        extra = rng.choice(rest, min(len(rest), max(0, args.max_pos - len(keep))), replace=False)
        sel = np.sort(np.concatenate([keep, extra]).astype(np.int64))
        Xpos, rids = Xpos[sel], rids[sel]
    Xpos = Xpos.to(device)
    Xneg = torch.cat([subsample(bank, args.n_neg).detach().cpu()] + neg_f, 0).to(device)
    mu = bank.mean(0, keepdim=True)
    sd = bank.std(0, keepdim=True) + 1e-6
    p(f'      NRS: {rbase} region NATIVE / {tot_px} px defect gốc -> train {len(Xpos)} pos, '
      f'{len(Xneg)} neg (bank {args.n_neg} + in-image)')
    torch.manual_seed(args.seed)
    head = Head(C, mu, sd).to(device)
    if args.loss == 'softpro':
        head = train_softpro(head, Xpos, rids, Xneg, args.steps, args.lr,
                             args.q, args.temp, args.w_bce, args.w_fp, device)
    else:
        head = train_bce(head, Xpos, Xneg, args.steps, args.lr, device)
    head.eval()
    return head


def run_cat(bb, cat, args, layers, gk, device, p):
    T, gt_ = args.tiles, args.grid_tile
    R = gt_ * bb.patch
    hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    if args.max_train:
        tr = tr[:args.max_train]
    p(f'  [{cat}] build bank từ {len(tr)} ảnh train...')
    bank = build_bank(bb, tr, T, R, gt_, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]

    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    shots = bad[:args.shots]

    # chẩn đoán cơ chế: cùng 10 shot, hai cây thước thấy bao nhiêu supervision?
    Gg = T * gt_
    grid_px = sum(int(gt_grid(ds.gt_paths[i], 1, Gg).sum()) for i in shots)
    torch.manual_seed(args.seed)
    headA = build_head(bb, ds, shots, bank, args, layers, device)      # head HIỆN TẠI
    headB = build_nrs_head(bb, ds, shots, bank, args, layers, device, p)
    p(f'    supervision grid-GT: {grid_px} px trên lưới {Gg} '
      f'| head cũ: {"None -> dist-only" if headA is None else "OK"} | NRS: {"None" if headB is None else "OK"}')
    if headB is None:
        p(f'  [{cat}] NRS không build được (shot không có GT?) -> bỏ')
        return None

    reserve = max(args.shots, args.eval_reserve)
    idx_bad = bad[reserve:]
    idx_good = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0]
    p(f'    shots={len(shots)} | eval held-out: bad={len(idx_bad)} good={len(idx_good)} (reserve={reserve})')
    if args.max_eval:                                                   # 0 = FULL split
        idx_bad = idx_bad[:args.max_eval]
        idx_good = idx_good[:args.max_eval]
    idx = idx_bad + idx_good
    m = max(2, args.aupro_max) // 2
    ap_ids = set(idx_bad[:m] + idx_good[:m]) if args.aupro_max else set(idx)

    # PASS 1: encode 1 lần, áp CẢ HAI head lên cùng feature grid
    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))                    # (H, W) native
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

    # PASS 2: mỗi head -> map native (canvas 256, y hệt eval_native) -> Hist + AUPRO@512
    out = {}
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
                t = gk(t)                                              # gaussian @256 như production
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
        tag = ' (dist-only: head cũ None)' if (name == 'grid' and headA is None) else ''
        out[name] = {'f1': h.f1_at(h.ksig(args.thr_sigma)), 'f1max': h.f1_max(),
                     'aupro': aupro05(ap_preds, ap_gts), 'n_ap': len(ap_preds), 'tag': tag}
        p(f'    [{cat}] head={name:4s}{tag}: F1@ksig={out[name]["f1"]:.4f} '
          f'trần={out[name]["f1max"]:.4f} AUPRO0.05@{args.aupro_res}({len(ap_preds)}img)={out[name]["aupro"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser('NRS premise: head supervision native vs grid, đối đầu cùng pipeline')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--eval_reserve', type=int, default=0,
                    help='cố định tập eval = bad[reserve:] để so sánh giữa các shots; 0 = như cũ (bad[shots:])')
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--n_neg_img', type=int, default=8000, help='neg NATIVE trong-ảnh (tổng qua các shot)')
    ap.add_argument('--pos_per_region', type=int, default=1500, help='cap px/region native')
    ap.add_argument('--max_pos', type=int, default=20000, help='cap tổng pos px (giữ >=5/region)')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=0, help='0 = FULL split (không có bẫy [:0])')
    ap.add_argument('--aupro_max', type=int, default=120, help='số ảnh cho AUPRO@512 (PAIRED giữa 2 head); 0 = tất cả')
    ap.add_argument('--aupro_res', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=['can', 'wallplugs', 'fruit_jelly'])
    ap.add_argument('--out_dir', type=str, default='./nrs')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('nrs', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    gk = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    p(f'device={device} model={args.model} eff_grid={args.tiles * args.grid_tile} '
      f'layers={args.layers}->{layers} loss={args.loss} shots={args.shots} head_w={args.head_w} '
      f'k={args.thr_sigma} | A=head grid-GT (production)  B=head NRS (GT native)')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, gk, device, p)
        if r is not None:
            res[cat] = r
    if not res:
        return

    p('\n' + '=' * 92 + '\n===== TỔNG (thước native, full split trừ shots) =====')
    for cat, r in res.items():
        d_f1 = r['nrs']['f1'] - r['grid']['f1']
        d_fm = r['nrs']['f1max'] - r['grid']['f1max']
        d_ap = r['nrs']['aupro'] - r['grid']['aupro']
        p(f"  [{cat:11s}] grid{r['grid']['tag']}: F1={r['grid']['f1']:.4f} trần={r['grid']['f1max']:.4f} "
          f"AUPRO={r['grid']['aupro']:.4f}")
        p(f"  [{cat:11s}] nrs : F1={r['nrs']['f1']:.4f} trần={r['nrs']['f1max']:.4f} "
          f"AUPRO={r['nrs']['aupro']:.4f}   Δ=({d_f1:+.4f}, {d_fm:+.4f}, {d_ap:+.4f})")
    for key, lab in [('f1', 'F1@ksig'), ('f1max', 'trần'), ('aupro', 'AUPRO0.05')]:
        a = float(np.mean([res[c]['grid'][key] for c in res]))
        b = float(np.mean([res[c]['nrs'][key] for c in res]))
        p(f'  MEAN {lab:9s}: grid={a:.4f}  nrs={b:.4f}  Δ={b - a:+.4f}')

    p('\nĐỌC (pre-registered):')
    p('  - NRS >= grid +0.03 F1 native trên can  -> premise SỐNG: supervision đúng thước cứu')
    p('    defect dưới-ô. Nấu vào pipeline (thay build_head) + đo full 8 cat -> novelty paper.')
    p('  - NRS ~ grid (±0.01)                    -> GIẾT trong buổi này. Bottleneck không phải')
    p('    thước supervision mà là bản thân feature của ô (defect 5px không làm đổi feature ô).')
    p('  - NRS < grid                            -> pos nội-suy (90% feature normal) kéo head lệch;')
    p('    thử tăng w_bce / giảm pos biên trước khi kết luận.')


if __name__ == '__main__':
    main()
