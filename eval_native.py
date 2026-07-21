# eval_native.py
# -----------------------------------------------------------------------------
# SỬA CÂY THƯỚC. Mọi SegF1 "public" ta báo cáo từ trước tới giờ đều đo ở 256x256:
#   infer_submit.up_to():  grid -> [256x256 VUÔNG] -> gaussian -> resize (H,W)
#   infer_submit.gt_grid(): GT (2448x2048) -> resize 256 NEAREST
# Nhưng SERVER chấm SegF1 ở ĐỘ PHÂN GIẢI GỐC 2448x2048.
#
# Vì sao chí mạng (số từ paper MVTec AD 2 / RoBiS): ">20% anomaly < 281 pixel, nhỏ nhất 5 pixel".
#   Ở lưới 256, một ô = 9.6 x 8.0 px gốc:
#     - defect 5 px   -> NEAREST làm nó BIẾN MẤT khỏi GT của ta  => ta không hề chấm nó
#     - defect 281 px -> còn ~3 ô                                 => gaussian nuốt nốt
#   => ta đang tự chấm mình trên một bài toán DỄ HƠN bài thật, đúng ở nhóm defect nhỏ mà
#      benchmark này dùng để phân thứ hạng. Public đẹp mà leaderboard thì không.
#
# Và cổ chai 256 còn bóp PRECISION thật: map buộc phải qua 256 rồi bung lên 2448 => mọi blob
#   dự đoán rộng tối thiểu ~10px gốc. Với defect 17x17px thì blob phình gấp mấy lần => SegF1 sụp.
#   Ở 256 thì mất mát này VÔ HÌNH (map và GT cùng bị làm mờ như nhau) => nên ta chưa từng thấy.
#   (Điều này cũng VÔ HIỆU HÓA kết luận cũ "canvas 256->512 không giúp": đo canvas 512 bằng
#    thước 256 thì không có nghĩa gì.)
#
# Script này đo Ở NATIVE (GT gốc, không resize), quét CANVAS = độ phân giải cổ chai:
#   - SegF1@ksig(native)  : con số thật, so được với server
#   - SegF1_max(native)   : TRẦN thật của map (bỏ biến quy tắc cắt)
#   - SegF1@ksig(256)     : cách đo CŨ, để thấy chính xác cây thước đã nói dối bao nhiêu
# Gaussian được scale theo canvas => giữ NGUYÊN độ mờ vật lý => cô lập đúng biến "độ phân giải".
#
#   python eval_native.py --data_path ../data --out_dir ./native --tiles 3 --grid_tile 24 \
#       --canvases 256 512 1024 --categories sheet_metal rice vial
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
    build_bank, build_head, img_featgrid, nn_map, VALID, IMG_EXT,
)
from dataset import MVTecAD2Dataset                                     # noqa: E402
from utils import get_gaussian_kernel, get_logger                       # noqa: E402
from backbones_ext import load_backbone                                 # noqa: E402

warnings.filterwarnings('ignore')
NB = 4096                     # số bin histogram: đủ để dựng PR-curve chính xác trên ~200M pixel


class Hist:
    """Thống kê streaming trên hàng trăm triệu pixel native (không thể giữ mảng trong RAM).
    Histogram score tách theo GT => dựng PR-curve => F1_max; kèm mean/std => ngưỡng ksig."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.pos = np.zeros(NB, np.int64)     # histogram score của pixel defect
        self.neg = np.zeros(NB, np.int64)     # ... của pixel normal
        self.n = 0
        self.s1 = 0.0
        self.s2 = 0.0

    def add(self, s, g):
        b = np.clip(((s - self.lo) / (self.hi - self.lo + 1e-12) * NB).astype(np.int32), 0, NB - 1)
        gb = g.astype(bool)
        self.pos += np.bincount(b[gb], minlength=NB)
        self.neg += np.bincount(b[~gb], minlength=NB)
        self.n += s.size
        self.s1 += float(s.sum(dtype=np.float64))
        self.s2 += float(np.square(s, dtype=np.float64).sum())

    def f1_at(self, thr):
        """F1 khi cắt tại thr (pooled toàn cat, đúng như server: 1 ngưỡng cho cả category)."""
        b = int(np.clip((thr - self.lo) / (self.hi - self.lo + 1e-12) * NB, 0, NB))
        TP = float(self.pos[b:].sum()); FP = float(self.neg[b:].sum()); FN = float(self.pos[:b].sum())
        return 2 * TP / (2 * TP + FP + FN + 1e-9)

    def f1_max(self):
        """TRẦN: quét mọi ngưỡng bin, lấy F1 tốt nhất."""
        tp = np.cumsum(self.pos[::-1])[::-1].astype(np.float64)      # TP nếu cắt tại bin i
        fp = np.cumsum(self.neg[::-1])[::-1].astype(np.float64)
        fn = float(self.pos.sum()) - tp
        return float(np.max(2 * tp / (2 * tp + fp + fn + 1e-9)))

    def ksig(self, k):
        mu = self.s1 / max(self.n, 1)
        var = max(self.s2 / max(self.n, 1) - mu * mu, 0.0)
        return mu + k * float(np.sqrt(var))


def make_map(s_grid, canvas, gk, size_hw, device):
    """grid -> [canvas x canvas] -> gaussian (đã scale) -> (H,W) native. canvas=256 = pipeline HIỆN TẠI."""
    t = torch.tensor(s_grid, device=device)[None, None].float()
    t = F.interpolate(t, size=canvas, mode='bilinear', align_corners=False)
    t = gk(t)
    t = F.interpolate(t, size=size_hw, mode='bilinear', align_corners=False)
    return t[0, 0]


def run_cat(bb, cat, args, layers, kernels, device):
    T, gt = args.tiles, args.grid_tile
    R = gt * bb.patch; hw = args.head_w
    rng = np.random.default_rng(args.seed)
    tr = sorted(sum([glob.glob(os.path.join(args.data_path, cat, 'train', 'good', e)) for e in IMG_EXT], []))
    if not tr:
        return None
    bank = build_bank(bb, tr[:args.max_train] if args.max_train else tr,
                      T, R, gt, layers, args.enc_batch, args.bank_size, device)
    C = bank.shape[-1]
    ds = MVTecAD2Dataset(root=os.path.join(args.data_path, cat), transform=None, gt_transform=None, phase='test')
    bad = [i for i in range(len(ds.img_paths)) if ds.labels[i] == 1]
    rng.shuffle(bad)
    head = build_head(bb, ds, bad[:args.shots], bank, args, layers, device)
    if head is None:
        return None

    idx = [i for i in bad if i not in set(bad[:args.shots])][:args.max_eval]
    idx += [i for i in range(len(ds.img_paths)) if ds.labels[i] == 0][:args.max_eval]

    # PASS 1: grid map (nhẹ, giữ được trong RAM) + lo/hi pooled cho fuse
    grids, sizes = [], []
    with torch.no_grad():
        for i in tqdm(idx, ncols=70, desc=f'    {cat}', leave=False):
            pil = Image.open(ds.img_paths[i])
            sizes.append((pil.size[1], pil.size[0]))                  # (H, W) native
            g = img_featgrid(bb, pil, T, R, gt, layers, args.enc_batch)
            d = np.asarray(nn_map(g, bank, device))
            pr = torch.sigmoid(head(g.reshape(-1, C))).reshape(g.shape[0], g.shape[0]).cpu().numpy()
            grids.append((d, pr))
    lo, hi = np.percentile(np.concatenate([d.reshape(-1) for d, _ in grids]), [1, 99])
    s_grids = [((1 - hw) * ((d - lo) / (hi - lo + 1e-8)) + hw * pr).astype(np.float32) for d, pr in grids]

    # PASS 2: với mỗi canvas -> map native -> nạp histogram (2 vòng: lấy range, rồi nạp)
    gmin = min(float(s.min()) for s in s_grids); gmax = max(float(s.max()) for s in s_grids)
    out = {}
    for cv in args.canvases + ['old256']:
        h = Hist(gmin - 0.05, gmax + 0.05)
        for (s, (H, W), i) in zip(s_grids, sizes, idx):
            if cv == 'old256':                                        # CÁCH ĐO CŨ: map 256 vs GT 256 NEAREST
                m = make_map(s, 256, kernels[256], (256, 256), device).cpu().numpy()
                if ds.labels[i] == 0:
                    g = np.zeros((256, 256), np.uint8)
                else:
                    g = (np.asarray(Image.open(ds.gt_paths[i]).convert('L').resize((256, 256), Image.NEAREST))
                         > 127).astype(np.uint8)
            else:                                                     # NATIVE: GT gốc, không đụng vào
                m = make_map(s, cv, kernels[cv], (H, W), device).cpu().numpy()
                if ds.labels[i] == 0:
                    g = np.zeros((H, W), np.uint8)
                else:
                    g = (np.asarray(Image.open(ds.gt_paths[i]).convert('L')) > 127).astype(np.uint8)
            h.add(m.reshape(-1), g.reshape(-1))
        out[cv] = (h.f1_at(h.ksig(args.thr_sigma)), h.f1_max())
    return out


def main():
    ap = argparse.ArgumentParser('eval_native: SegF1 ở ĐỘ PHÂN GIẢI GỐC (như server), không phải 256')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--model', type=str, default='v3_large')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--grid_tile', type=int, default=24)
    ap.add_argument('--canvases', type=int, nargs='+', default=[256, 512, 1024],
                    help='độ phân giải CỔ CHAI trước khi bung lên native. 256 = pipeline hiện tại.')
    ap.add_argument('--bank_size', type=int, default=50000)
    ap.add_argument('--enc_batch', type=int, default=16)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--head_w', type=float, default=0.6)
    ap.add_argument('--loss', type=str, default='softpro', choices=['bce', 'softpro'])
    ap.add_argument('--n_neg', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--q', type=float, default=0.95)
    ap.add_argument('--temp', type=float, default=0.5)
    ap.add_argument('--w_bce', type=float, default=0.3)
    ap.add_argument('--w_fp', type=float, default=1.0)
    ap.add_argument('--thr_sigma', type=float, default=4.5)
    ap.add_argument('--max_train', type=int, default=60)
    ap.add_argument('--max_eval', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--categories', type=str, nargs='+', default=VALID)
    ap.add_argument('--out_dir', type=str, default='./native')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    p = get_logger('native', args.out_dir).info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    bb = load_backbone(args.model, device)
    layers = sorted(set(max(1, min(bb.n_layers - 1, round(l / 12 * bb.n_layers))) for l in args.layers))
    # gaussian scale theo canvas => ĐỘ MỜ VẬT LÝ không đổi => cô lập đúng biến độ phân giải
    kernels = {cv: get_gaussian_kernel(kernel_size=max(3, int(5 * cv / 256) | 1),
                                       sigma=4.0 * cv / 256).to(device)
               for cv in set(args.canvases) | {256}}
    p(f'device={device} model={args.model} eff_grid={args.tiles*args.grid_tile} canvases={args.canvases} '
      f'k={args.thr_sigma} | native = GT GỐC (như server); old256 = cách đo CŨ')

    res = {}
    for cat in args.categories:
        r = run_cat(bb, cat, args, layers, kernels, device)
        if r is None:
            p(f'  [{cat}] bỏ'); continue
        res[cat] = r
        p(f'  [{cat:11s}] ' + '  '.join(
            f'{("CŨ@256" if cv == "old256" else f"native@cv{cv}")}: F1={r[cv][0]:.4f} trần={r[cv][1]:.4f}'
            for cv in args.canvases + ['old256']))
    if not res:
        return

    p('\n' + '=' * 92 + '\n===== MEAN =====')
    for cv in args.canvases + ['old256']:
        f1 = float(np.mean([res[c][cv][0] for c in res]))
        fm = float(np.mean([res[c][cv][1] for c in res]))
        tag = 'CÁCH ĐO CŨ (GT resize 256)' if cv == 'old256' else f'NATIVE, canvas={cv}'
        p(f'  {tag:30s}: SegF1={f1:.4f}   TRẦN={fm:.4f}')

    p('\nĐỌC:')
    p('  1) native@cv256 vs CŨ@256: chênh bao nhiêu = cây thước cũ đã nói dối bấy nhiêu.')
    p('     Nếu native THẤP hơn nhiều => mọi con số public từ trước tới giờ đều bị thổi phồng,')
    p('     và ta đã tối ưu sai hàm mục tiêu (bỏ quên nhóm defect nhỏ - chính nhóm phân thứ hạng).')
    p('  2) canvas 256 -> 512 -> 1024 (đo ở native): cổ chai 256 có đang bóp SegF1 không?')
    p('     TRẦN lên theo canvas => bỏ cổ chai là LỢI MIỄN PHÍ, không cần ý tưởng mới.')
    p('     TRẦN đứng yên => cổ chai vô can, bottleneck nằm ở feature grid (eff_grid), đi tăng res.')


if __name__ == '__main__':
    main()
