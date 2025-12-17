import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class BatchedLoRALayer:
    def __init__(
        self,
        M: int,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.M = M
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Mark the weight as unmerged
        self.merge_weights = merge_weights
        self.merged = False


class BatchedEmbedding(nn.Module, BatchedLoRALayer):
    def __init__(
        self,
        base_embedding: nn.Embedding,
        M: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        merge_weights: bool = False,
    ):
        nn.Module.__init__(self)
        BatchedLoRALayer.__init__(self, M, r, lora_alpha, 0.0, merge_weights)

        self.weight = base_embedding.weight
        self.weight.requires_grad = False
        self.num_embeddings, self.embedding_dim = self.weight.shape

        if r > 0:
            self.lora_A = nn.Parameter(
                torch.zeros(M, r, self.num_embeddings)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(M, self.embedding_dim, r)
            )
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def forward(self, x):
        """
        x: [B, T]
        returns: [M, B, T, d]
        """
        base = F.embedding(x, self.weight)  # [B,T,d]
        base = base.unsqueeze(0).expand(self.M, -1, -1, -1)

        if self.r == 0:
            return base
        
        def embed_one(w):
            return F.embedding(x, w)

        # vectorize over lora particles dimension
        after_A = torch.vmap(embed_one)(self.lora_A.transpose(1, 2))

        delta = torch.einsum("mbtr,mdr->mbtd", after_A, self.lora_B)

        return base + self.scaling * delta


class BatchedMergedLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        M: int,
        r: int,
        enable_lora: list[bool],
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
    ):
        super().__init__()

        assert len(enable_lora) > 0
        assert sum(enable_lora) > 0

        self.M = M
        self.enable_lora = enable_lora
        self.G = sum(enable_lora)
        self.r = r
        self.scaling = lora_alpha / r
        self.fan_in_fan_out = fan_in_fan_out

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        assert self.out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        
        self.block = self.out_features // len(enable_lora)

        # Shared frozen base weights
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # Batched grouped LoRA parameters
        # A: [M, G * r, in]
        # B: [M, G * block, r]
        self.lora_A = nn.Parameter(
            torch.zeros(M, self.G * r, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(M, self.G * self.block, r)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(lora_dropout)

        # Mask with output indices for enabled blocks 
        lora_out_indices = []
        for i, enabled in enumerate(enable_lora):
            if enabled:
                lora_out_indices.append(
                    torch.arange(i * self.block, (i + 1) * self.block)
                )

        self.register_buffer(
            "lora_out_indices",
            torch.cat(lora_out_indices)
        )

    def forward(self, x):
        """
        x: [M * B, T, d] or [M, B, T, d]
        returns: [M * B, T, d_out]
        """
        if x.dim() == 3:
            x = x.reshape(self.M, -1, *x.shape[1:])
            # x = x.unsqueeze(0).expand(self.M, -1, -1)

        def T(w):
            return w.T if self.fan_in_fan_out else w

        # Base output
        base = torch.einsum("mbtd,od->mbto", x, T(self.weight))
        if self.bias is not None:
            base = base + self.bias.view(1, 1, -1)

        # LoRA update
        x_d = self.dropout(x)

        # Merge A and B using grouped conv1d
        lora_A_reshaped = self.lora_A.reshape(1, self.M * self.G * self.r, self.in_features)
        lora_B_reshaped = self.lora_B.reshape(self.M * self.G * self.block, self.r, 1)

        delta_w_reshaped = F.conv1d(
            lora_A_reshaped,                 # [1, M*G*r, d]
            lora_B_reshaped,                 # [M*G*block, r, 1]
            groups = self.M *self.G,
        ).squeeze(0)                         # [M*G*block, d]
        delta_w = delta_w_reshaped.reshape(self.M, self.G * self.block, self.in_features)
        
        # Apply ΔW in one GEMM        
        delta_out = torch.einsum("mbtd,mod->mbto", x_d, T(delta_w))
        base[:, :, :, self.lora_out_indices] += self.scaling * delta_out

        return base.reshape(-1, *x.shape[2:])