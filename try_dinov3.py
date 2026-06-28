# try_dinov3.py — TẢI HẲN weight DINOv3 về cache + verify forward.
# Sau khi chạy OK, diag13 chỉ việc đọc cache (chạy với HF_HUB_OFFLINE=1).
#
# Chạy:
#   export HF_TOKEN=hf_xxx                 # token đã được duyệt quyền (từng size 1 repo gated)
#   python try_dinov3.py                   # mặc định tải base + large + huge
#   python try_dinov3.py facebook/dinov3-vitb16-pretrain-lvd1689m   # chỉ 1 cái
import os
import sys
import torch

DEFAULT = [
    'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'facebook/dinov3-vith16plus-pretrain-lvd1689m',
]
ids = sys.argv[1:] or DEFAULT

try:
    import transformers
    from transformers import AutoModel
except Exception as e:
    raise SystemExit(f'Thiếu transformers -> uv pip install -U transformers  ({e})')
print('transformers', transformers.__version__)
try:
    from huggingface_hub import constants
    print('HF cache:', constants.HF_HUB_CACHE)
except Exception:
    pass

ok, fail = [], []
for mid in ids:
    print('\n' + '=' * 70)
    print('TẢI:', mid)
    model = None
    for trc in (False, True):              # thử thường, rồi trust_remote_code
        try:
            model = AutoModel.from_pretrained(mid, trust_remote_code=trc)
            break
        except Exception as e:
            last = e
    if model is None:
        print('  FAIL:', last)
        msg = str(last)
        if '401' in msg or 'gated' in msg.lower() or 'restricted' in msg.lower():
            print('  -> chưa có quyền repo NÀY (mỗi size gated riêng): mở trang model, bấm Agree + HF_TOKEN.')
        elif 'safetensors' in msg or 'pytorch_model' in msg or 'recognize' in msg.lower():
            print('  -> transformers cũ: uv pip install -U transformers')
        fail.append(mid)
        continue

    cfg = model.config
    patch = getattr(cfg, 'patch_size', 16)
    nreg = getattr(cfg, 'num_register_tokens', 0) or 0
    print(f'  LOAD OK | patch={patch} reg={nreg} layers={getattr(cfg,"num_hidden_layers","?")} '
          f'hidden={getattr(cfg,"hidden_size","?")}')
    # forward thử
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    R = 28 * patch
    x = torch.randn(1, 3, R, R)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        out = model(pixel_values=x, output_hidden_states=True)
    ntok = out.hidden_states[-1].shape[1] - 1 - nreg
    print(f'  FORWARD OK | input={R} | #hidden_states={len(out.hidden_states)} '
          f'| patch_tokens={ntok} (grid~{int(ntok**0.5)})')
    ok.append(mid)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('\n' + '=' * 70)
print('CACHED OK :', ok if ok else '(none)')
print('FAIL      :', fail if fail else '(none)')
if ok:
    print('\n=> Đã cache. Chạy diag13 với cache (khỏi đụng mạng/gate):')
    print('   HF_HUB_OFFLINE=1 python diagnosis_scripts/diagnosis13_backbone_sweep.py ...')
