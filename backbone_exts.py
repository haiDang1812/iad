# backbones_ext.py
# -----------------------------------------------------------------------------
# Loader THỐNG NHẤT cho DINOv2 + DINOv3, mọi size (base/large/huge|giant), qua HuggingFace
# transformers. KHÔNG đụng vit_encoder.py gốc.
#
# Trả về object Backbone với:
#   .extract(imgs[B,3,R,R], layers) -> [B, N, C]  (mean các layer; đã bỏ CLS + register tokens)
#   .patch (14 cho v2, 16 cho v3) · .n_reg · .n_layers · .family · .name
#
# Tên gọi -> HF id (mặc định, có thể override bằng "name=hf_id"):
#   v2_base/large/giant  = facebook/dinov2-with-registers-{base,large,giant}   (patch 14)
#   v3_base/large/huge   = facebook/dinov3-vit{b,l}16 / vith16plus -pretrain-lvd1689m (patch 16)
#
# Lưu ý: weight DINOv3 GATED -> cần `huggingface-cli login` + được duyệt quyền truy cập.
# -----------------------------------------------------------------------------

import torch

_HF = {
    # DINOv2 (with registers, patch 14)
    'v2_base':  'facebook/dinov2-with-registers-base',
    'v2_large': 'facebook/dinov2-with-registers-large',
    'v2_giant': 'facebook/dinov2-with-registers-giant',
    'v2_huge':  'facebook/dinov2-with-registers-giant',   # v2 không có "huge"; giant là lớn nhất
    # DINOv3 (patch 16)
    'v3_base':  'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'v3_large': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'v3_huge':  'facebook/dinov3-vith16plus-pretrain-lvd1689m',
    'v3_7b':    'facebook/dinov3-vit7b16-pretrain-lvd1689m',
}


def resolve_hf_id(name):
    if '=' in name:                      # "myname=facebook/xxx"
        return name.split('=', 1)
    if name in _HF:
        return name, _HF[name]
    return name, name                    # coi như đã là HF id


class Backbone:
    def __init__(self, model, patch, n_reg, n_layers, family, name, device):
        self.model = model.to(device).eval()
        self.patch = patch
        self.n_reg = n_reg
        self.n_layers = n_layers
        self.family = family
        self.name = name
        self.device = device

    @torch.no_grad()
    def extract(self, imgs, layers):
        out = self.model(pixel_values=imgs.to(self.device), output_hidden_states=True)
        hs = out.hidden_states                 # tuple len = n_layers+1 (0 = embeddings)
        skip = 1 + self.n_reg                   # bỏ CLS + register tokens
        feats = []
        for l in layers:
            li = min(l, len(hs) - 1)
            feats.append(hs[li][:, skip:, :])
        return torch.stack(feats, dim=1).mean(dim=1)   # [B, N, C]


def load_backbone(name, device='cuda:0'):
    try:
        from transformers import AutoModel
    except Exception as e:
        raise SystemExit(f'Cần `transformers`: pip install transformers. ({e})')

    key, hf_id = resolve_hf_id(name)
    try:
        model = AutoModel.from_pretrained(hf_id)
    except Exception as e:
        raise SystemExit(
            f'Không load được "{hf_id}". Nếu là DINOv3 -> weight GATED: '
            f'`huggingface-cli login` + xin quyền truy cập model trên HuggingFace.\n  Lỗi: {e}')

    cfg = model.config
    patch = getattr(cfg, 'patch_size', 16)
    n_reg = getattr(cfg, 'num_register_tokens', 0) or 0
    n_layers = getattr(cfg, 'num_hidden_layers', None)
    family = 'v3' if 'dinov3' in hf_id.lower() else ('v2' if 'dinov2' in hf_id.lower() else '?')
    print(f'[backbone] {key} <- {hf_id} | patch={patch} reg={n_reg} layers={n_layers} family={family}')
    return Backbone(model, patch, n_reg, n_layers, family, key, device)
