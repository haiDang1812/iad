import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.vision_transformer import ResidualCalibrationModule


class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            # ── NEW ──
            use_calibration=True,
            calibration_heads=8,
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
        self.use_calibration = use_calibration

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        # ── NEW: Residual Calibration Module ──
        if use_calibration:
            # embed_dim inferred from prototype_token shape
            embed_dim   = prototype_token[0].shape[-1]
            inp_num     = prototype_token[0].shape[0]
            self.residual_calibration = ResidualCalibrationModule(
                dim=embed_dim,
                num_prototypes=inp_num,
                num_heads=calibration_heads
            )

    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(
            query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    # ── NEW: Normality Contrastive Loss ──────────────────────────────────────
    def normality_contrastive_loss(self, en_fused, de_fused, prototype, margin=0.2):
        """
        Push prototype away from anomaly feature space.
        Uses residual map as self-supervised signal — no GT needed.

        en_fused: [B, C, H, W]
        de_fused: [B, C, H, W]
        prototype: [B, INP_num, C]
        """
        B, C, H, W = en_fused.shape

        residual = 1.0 - F.cosine_similarity(en_fused, de_fused, dim=1)  # [B, H, W]
        r_flat   = residual.flatten(1)  # [B, N]

        # Top-k anomaly tokens and bottom-k normal tokens
        k = max(1, int(r_flat.shape[1] * 0.05))
        anomaly_idx = torch.topk(r_flat,  k, dim=1)[1]  # [B, k]
        normal_idx  = torch.topk(r_flat, k, dim=1, largest=False)[1]

        en_flat = en_fused.flatten(2).permute(0, 2, 1)  # [B, N, C]

        # Gather anomaly and normal features
        anomaly_feat = en_flat.gather(
            1, anomaly_idx.unsqueeze(-1).expand(-1, -1, C)).mean(dim=1)  # [B, C]
        normal_feat  = en_flat.gather(
            1, normal_idx.unsqueeze(-1).expand(-1, -1, C)).mean(dim=1)   # [B, C]

        proto_mean = prototype.mean(dim=1)  # [B, C]

        sim_normal  = F.cosine_similarity(proto_mean, normal_feat,  dim=-1)  # [B]
        sim_anomaly = F.cosine_similarity(proto_mean, anomaly_feat, dim=-1)  # [B]

        # Prototype should be closer to normal than anomaly
        loss = F.relu(sim_anomaly - sim_normal + margin).mean()
        return loss
    # ─────────────────────────────────────────────────────────────────────────

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

        side = int(math.sqrt(
            en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)

        # ── INP Extractor (unchanged) ──
        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)

        g_loss = self.gather_loss(x, agg_prototype)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # ── Decoder Pass 1 — get rough residual for calibration ──
        de_list = []
        x_de = x
        for i, blk in enumerate(self.decoder):
            x_de = blk(x_de, agg_prototype)
            de_list.append(x_de)
        de_list_1 = de_list[::-1]

        en_fused_spatial, de_fused_spatial = None, None
        cal_loss = torch.tensor(0., device=x.device)

        if self.use_calibration:
            # Fuse for calibration (dùng cùng fuse_layer config)
            en_cal = [self.fuse_feature([en_list[idx] for idx in idxs])
                      for idxs in self.fuse_layer_encoder]
            de_cal = [self.fuse_feature([de_list_1[idx] for idx in idxs])
                      for idxs in self.fuse_layer_decoder]

            # Remove class token nếu chưa remove
            if not self.remove_class_token:
                en_cal = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_cal]
                de_cal = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de_cal]

            # Reshape về spatial
            en_cal_sp = [e.permute(0, 2, 1).reshape(B, -1, side, side).contiguous()
                         for e in en_cal]
            de_cal_sp = [d.permute(0, 2, 1).reshape(B, -1, side, side).contiguous()
                         for d in de_cal]

            # Dùng fused feature đầu tiên để calibrate
            en_fused_spatial = en_cal_sp[0]
            de_fused_spatial = de_cal_sp[0]

            # Calibrate prototype
            calibrated_prototype, cal_weights = self.residual_calibration(
                en_fused_spatial, de_fused_spatial, agg_prototype)

            # Normality contrastive loss
            cal_loss = self.normality_contrastive_loss(
                en_fused_spatial, de_fused_spatial, calibrated_prototype)

            # ── Decoder Pass 2 — với calibrated prototype ──
            de_list = []
            x_de2 = x  # reset về bottleneck output
            for i, blk in enumerate(self.decoder):
                x_de2 = blk(x_de2, calibrated_prototype)
                de_list.append(x_de2)
            de_list = de_list[::-1]
        else:
            de_list = de_list_1

        # ── Final output ──
        en = [self.fuse_feature([en_list[idx] for idx in idxs])
              for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs])
              for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape(B, -1, side, side).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape(B, -1, side, side).contiguous() for d in de]

        return en, de, g_loss, cal_loss  # ← thêm cal_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)