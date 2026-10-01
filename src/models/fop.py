import torch
import torch.nn as nn
import torch.nn.functional as F


def make_fc_1d(f_in: int, f_out: int, n_layers: int = 1, dropout: float = 0.5):
    layers = [
        nn.Linear(f_in, f_out),
        nn.BatchNorm1d(f_out),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
    ]

    # For any intermediate layers
    for _ in range(n_layers - 2):
        layers.extend([
            nn.Linear(f_out, f_out),
            nn.BatchNorm1d(f_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ])

    # Final projection layer
    if n_layers > 1:
        layers.append(nn.Linear(f_out, f_out))

    return nn.Sequential(*layers)
class ForwardBlock(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, p_val=0.0):
        super(ForwardBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(p=p_val),
        )

    def forward(self, x):
        return self.block(x)


class LinearWeightedAvg(nn.Module):
    def __init__(self):
        super(LinearWeightedAvg, self).__init__()
        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))

    def forward(self, face_feat, voice_feat):
        fused = self.w1 * face_feat + self.w2 * voice_feat
        return fused, face_feat, voice_feat


class GatedFusion(nn.Module):
    def __init__(self, embed_dim=128, mid_att_dim=128):
        super(GatedFusion, self).__init__()
        self.attention = nn.Sequential(
            ForwardBlock(embed_dim * 2, mid_att_dim),
            nn.Linear(mid_att_dim, embed_dim),
        )

    def forward(self, face_embed, voice_embed):
        concat = torch.cat((face_embed, voice_embed), dim=1)
        att = torch.sigmoid(self.attention(concat))
        face_trans = torch.tanh(face_embed)
        voice_trans = torch.tanh(voice_embed)
        fused = face_trans * att + (1.0 - att) * voice_trans
        return fused, face_embed, voice_embed
class EmbedBranch(nn.Module):
    def __init__(self, feat_dim: int, dim_embed: int, n_layers: int = 1, dropout: float = 0.5):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, dim_embed, n_layers=n_layers, dropout=dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class EnhancedGatedFusion(nn.Module):
    """
    Enhanced Gate Feature Fusion (EGFF) from PAEFF.
    """
    def __init__(self, embed_dim: int=128, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.res_mix = nn.Linear(embed_dim, embed_dim)

    def forward(self, face_embed, voice_embed):
        # [1] Enhanced non-linear representation
        face_features = F.gelu(face_embed)
        voice_features = F.gelu(voice_embed)

        # [2] 1D Convolution over cross-modal interaction
        attn = face_features * voice_features
        attn = self.conv(attn.unsqueeze(1)).squeeze(1)
        attn = torch.sigmoid(attn)

        # [3] Gated fusion & Post-fusion Projection
        fused = attn * face_embed + (1.0 - attn) * voice_embed
        fused = torch.tanh(self.res_mix(fused))

        return fused, face_features, voice_features

class FOP(nn.Module):
    def __init__(self, model_cfg, face_feat_dim, voice_feat_dim, n_class):
        super(FOP, self).__init__()
        self.dim_embed = model_cfg.dim_embed
        self.fusion = model_cfg.fusion

        n_layers = getattr(model_cfg, "n_layers", 1)
        dropout = getattr(model_cfg, "dropout", 0.5)

        self.voice_branch = EmbedBranch(voice_feat_dim, self.dim_embed, n_layers=n_layers, dropout=dropout)
        self.face_branch = EmbedBranch(face_feat_dim, self.dim_embed, n_layers=n_layers, dropout=dropout)

        if self.fusion == "linear":
            self.fusion_layer = LinearWeightedAvg()
        elif self.fusion == "gated":
            self.fusion_layer = GatedFusion(self.dim_embed, 128)
        elif self.fusion == 'egff':
            self.fusion_layer = EnhancedGatedFusion(self.dim_embed)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")

        self.logits_layer = nn.Linear(self.dim_embed, n_class)

    def forward(self, faces, voices):
        voice_embeds = self.voice_branch(voices)
        face_embeds = self.face_branch(faces)
        fused, face_embeds, voice_embeds = self.fusion_layer(
            face_embeds, voice_embeds
        )
        return fused, face_embeds, voice_embeds

    def train_forward(self, faces, voices, labels):
        voice_embeds = self.voice_branch(voices)
        face_embeds = self.face_branch(faces)
        fused, _, _ = self.fusion_layer(face_embeds, voice_embeds)
        logits = self.logits_layer(fused)
        comb = [fused, logits]
        return comb, face_embeds, voice_embeds
