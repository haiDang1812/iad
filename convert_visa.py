# convert_visa.py
# -----------------------------------------------------------------------------
# VisA thô (giải nén phẳng ra <data_root>/<category>/Data/Images/...) + split_csv/1cls.csv
#   -> layout MVTec-style mà eval_generalize.SimpleADDataset cần:
#     <out>/<cat>/train/good/<file>
#     <out>/<cat>/test/good/<file>
#     <out>/<cat>/test/bad/<file>
#     <out>/<cat>/ground_truth/bad/<stem>.png   (mask, khớp find_gt kiểu VisA)
#   Dùng SYMLINK (không copy) — không tốn dung lượng, không đụng data gốc.
#
#   python convert_visa.py --data_root /workspace --out /workspace/VisA_pytorch/1cls
# -----------------------------------------------------------------------------
import os
import csv
import argparse
import numpy as np
from PIL import Image


def write_mask_255(src, dst):
    """VisA mask lưu giá trị 1 (hoặc mã màu tối) -> mọi reader downstream ngưỡng >127 đọc RỖNG.
    Vật chất hoá thành PNG 0/255 nhị phân (>0) để khớp chuẩn MVTec. Ảnh vẫn symlink, chỉ mask ghi thật."""
    m = (np.asarray(Image.open(src).convert('L')) > 0).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    Image.fromarray(m).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True, help='thư mục chứa các category VisA + split_csv/')
    ap.add_argument('--csv', default=None, help='mặc định = <data_root>/split_csv/1cls.csv')
    ap.add_argument('--out', required=True, help='thư mục layout đầu ra (…/VisA_pytorch/1cls)')
    ap.add_argument('--copy', action='store_true', help='copy thay vì symlink')
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(args.data_root, 'split_csv', '1cls.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'không thấy csv: {csv_path}')

    link = (__import__('shutil').copy2) if args.copy else os.symlink

    def put(src_rel, dst):
        src = os.path.join(args.data_root, src_rel)
        if not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        link(os.path.abspath(src), dst)
        return True

    n_img = n_mask = n_miss = 0
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        cols = {c.lower(): c for c in rd.fieldnames}
        c_obj = cols.get('object')
        c_split = cols.get('split')
        c_label = cols.get('label')
        c_img = cols.get('image')
        c_mask = cols.get('mask')
        for row in rd:
            cat = row[c_obj].strip()
            split = row[c_split].strip().lower()          # train / test
            label = row[c_label].strip().lower()          # normal / anomaly
            img_rel = row[c_img].strip()
            base = os.path.basename(img_rel)
            stem = os.path.splitext(base)[0]
            is_good = 'normal' in label
            sub = 'good' if is_good else 'bad'
            dst_img = os.path.join(args.out, cat, split, sub, base)
            if put(img_rel, dst_img):
                n_img += 1
            else:
                n_miss += 1
                print(f'  [thiếu ảnh] {img_rel}')
            if not is_good and c_mask and row.get(c_mask, '').strip():
                mask_rel = row[c_mask].strip()
                src_m = os.path.join(args.data_root, mask_rel)
                dst_m = os.path.join(args.out, cat, 'ground_truth', 'bad', stem + '.png')
                if os.path.exists(src_m):
                    write_mask_255(src_m, dst_m)                            # binarize 0/255, không symlink
                    n_mask += 1
    print(f'\nXONG: {n_img} ảnh, {n_mask} mask, {n_miss} thiếu -> {args.out}')
    # tóm tắt theo category
    for cat in sorted(os.listdir(args.out)):
        d = os.path.join(args.out, cat)
        if not os.path.isdir(d):
            continue
        def cnt(*p):
            q = os.path.join(d, *p)
            return len(os.listdir(q)) if os.path.isdir(q) else 0
        print(f'  {cat:12s} train/good={cnt("train","good"):4d}  '
              f'test/good={cnt("test","good"):4d}  test/bad={cnt("test","bad"):4d}  '
              f'gt/bad={cnt("ground_truth","bad"):4d}')


if __name__ == '__main__':
    main()
