import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualCalibration(nn.Module):
    """Dự đoán residual NORMAL kỳ vọng mu(p) từ feature encoder.

    Grounded vào chẩn đoán rare-normal-FP: vùng bình-thường-nhưng-HIẾM sinh residual
    tái tạo cao -> false positive ở FPR thấp -> dập AUPRO0.05. Vì train chỉ có normal,
    MỌI residual lúc train là normal -> mu học "trường residual bình thường". Lúc chấm:
    score = relu(r - lam*mu) -> giữ residual thật sự BẤT NGỜ (defect), chiết khấu phần
    residual rare-normal đoán-trước-được. (Frozen rarecal chết vì không học được mu; ở đây
    model HỌC nên có cửa.)"""

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or max(64, dim // 4)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Softplus(),                      # mu >= 0
        )

    def forward(self, feat):                    # feat: [B, C, H, W] (encoder fused)
        return self.net(feat)                   # [B, 1, H, W]


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            use_calibration=False,
            cvar_alpha=1.0,
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        self.cvar_alpha = cvar_alpha
        self.use_calibration = use_calibration
        if use_calibration:
            dim = self.prototype_token.shape[-1]        # embed_dim
            self.residual_calibration = ResidualCalibration(dim)

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        alpha = self.cvar_alpha
        if alpha >= 1.0:
            # k-means style: mean over all patches (INP-Former gốc)
            gather_loss = self.distance.mean()
        else:
            # tail-coverage (CVaR_alpha): mean over the worst-covered alpha fraction.
            # Ep prototype phu vung rare-normal te nhat -> giam FP low-FPR tu goc.
            d = self.distance.reshape(-1)
            k = max(1, int(math.ceil(alpha * d.numel())))
            topk, _ = torch.topk(d, k, largest=True, sorted=False)
            gather_loss = topk.mean()
        return gather_loss

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(x, agg_prototype)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        if self.use_calibration:
            cal_r, cal_mu, cal_loss = [], [], 0.
            for e, d in zip(en, de):
                r = (1. - F.cosine_similarity(e, d, dim=1)).unsqueeze(1)   # [B,1,H,W] residual
                mu = self.residual_calibration(e)                          # [B,1,H,W] normal-residual dự đoán
                cal_loss = cal_loss + F.smooth_l1_loss(mu, r.detach())     # detach: KHÔNG hại nhánh recon
                cal_r.append(r); cal_mu.append(mu)
            cal_loss = cal_loss / len(en)
            self._cal_r, self._cal_mu = cal_r, cal_mu                      # để eval calibrated đọc
            return en, de, g_loss, cal_loss

        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)









































