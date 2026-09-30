import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, dim_embed):
        super(EmbedBranch, self).__init__()
        self.fc1 = nn.Linear(feat_dim, dim_embed)
        self.bn1 = nn.BatchNorm1d(dim_embed)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        return x


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs, n_features):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(n_inputs)]
        )
        self.bn = nn.BatchNorm1d(n_features)

    def forward(self, input_list):
        res = 0
        for i, weight in enumerate(self.weights):
            res += input_list[i] * weight
        return self.bn(res)


class GatedFusion(nn.Module):
    def __init__(
        self,
        dim_face,
        dim_voice,
        dim_face_embed,
        dim_voice_embed,
        dim_fused_embed,
    ):
        super(GatedFusion, self).__init__()
        self.fc1 = nn.Linear(dim_face + dim_voice, 2)
        self.bn1 = nn.BatchNorm1d(dim_fused_embed)

    def forward(self, face, voice, face_embed, voice_embed):
        features = torch.cat((face, voice), dim=1)
        z = F.sigmoid(self.fc1(features))
        fused = (z[:, 0].unsqueeze(1) * face_embed) + (
            z[:, 1].unsqueeze(1) * voice_embed
        )
        return self.bn1(fused)


class FOP(nn.Module):
    def __init__(self, model_cfg, face_feat_dim, voice_feat_dim, n_class):
        super(FOP, self).__init__()
        self.dim_embed = model_cfg.dim_embed
        self.fusion = model_cfg.fusion

        self.voice_branch = EmbedBranch(voice_feat_dim, self.dim_embed)
        self.face_branch = EmbedBranch(face_feat_dim, self.dim_embed)

        if self.fusion == "linear":
            self.fusion_layer = LinearWeightedAvg(self.dim_embed, self.dim_embed)
        elif self.fusion == "gated":
            self.fusion_layer = GatedFusion(
                face_feat_dim,
                voice_feat_dim,
                self.dim_embed,
                128,
                self.dim_embed,
            )

        self.logits_layer = nn.Linear(self.dim_embed, n_class)

    def forward(self, faces, voices):
        face_embeds = self.face_branch(faces)
        voice_embeds = self.voice_branch(voices)

        if self.fusion == "linear":
            fused_embeds = self.fusion_layer([face_embeds, voice_embeds])
        elif self.fusion == "gated":
            fused_embeds = self.fusion_layer(
                faces, voices, face_embeds, voice_embeds
            )

        return fused_embeds, face_embeds, voice_embeds

    def train_forward(self, faces, voices, labels):
        fused_embeds, face_embeds, voice_embeds = self.forward(faces, voices)
        logits = self.logits_layer(fused_embeds)
        comb = [fused_embeds, logits]
        return comb, face_embeds, voice_embeds
