"""
The Fidelity-Uncertainty Frontier in LoRA Fine-Tuning:
Post-Bayesian Posterior Flows

Notes:
- Base model is a HuggingFace-style causal LM
- LoRA layers are replaced by BatchedLinear / BatchedEmbedding
- M particles are represented by the batch dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def log_likelihood_from_logits(logits, targets, ignore_index=-100):
    """
    logits: [M, B, T, V]   T: sequence length, V: vocabulary
    targets: [B, T]
    returns: [M, B]
    """
    M, B, T, V = logits.shape

    logits = logits.view(M * B * T, V)
    targets = targets.unsqueeze(0).expand(M, -1, -1)
    targets = targets.reshape(M * B * T)

    loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore_index,
    )

    loss = loss.view(M, B, T)
    logp = -loss.sum(dim=-1)  # sum over tokens to compute joing log prob
    return logp  # [M, B]


# -----------------------------------------------------------------------------
# Rényi responsabilities (stable computation)
# -----------------------------------------------------------------------------

def renyi_responsibilities(logp: torch.Tensor, alpha: float):
    """
    logp: [M, B]
    returns: w_alpha [M, B]
    """
    scores = alpha * logp
    w = torch.softmax(scores, dim=0)  # softmax across lora particles dimension
    return w


# -----------------------------------------------------------------------------
# Particle trainer
# -----------------------------------------------------------------------------

class RenyiLoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        M: int,
        alpha: float,
        lr: float,
        lambda_kl: float,
        prior_std: float = 1.0,
    ):
        self.model = model
        self.M = M
        self.alpha = alpha
        self.lambda_kl = lambda_kl
        self.prior_std = prior_std

        # Only LoRA params require grad
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = AdamW(params, lr=lr)

    def kl_grad(self, param):
        """Gaussian prior gradient"""
        return -param / (self.prior_std ** 2)

    def training_step(self, batch: Dict[str, torch.Tensor]):
        """
        batch: {input_ids, labels}
        """
        input_ids = batch["input_ids"]  # [B, T]
        batch_size = input_ids.shape[0]
        labels = batch["labels"]

        # Forward: logits [M, B, T, V]
        logits = self.model(input_ids)

        # Log-likelihoods
        logp = log_likelihood_from_logits(logits, labels)

        # Responsibilities
        w_alpha = renyi_responsibilities(logp, self.alpha).detach()  # [M, B]
        # detached from the computational graph to not take gradients across it so that we can compute the optimiser update easily

        # Weighted objective 
        # L = - (1/B) * sum_i sum_b w_i,b log p_i,b
        loss = -(w_alpha * logp).sum() / batch_size

        # KL regularisation
        kl_term = 0.0
        for p in self.model.parameters():
            if p.requires_grad:
                kl_term += (p ** 2).sum() / (2 * self.prior_std ** 2)

        loss = loss + self.lambda_kl / self.M * kl_term

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.item(),
            "nll": (-logp.mean()).item(),
        }


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------

class BatchedCausalLM(nn.Module):
    """
    Wraps a base LM whose Linear/Embedding layers
    have already been replaced by BatchedLoRA variants.
    """

    def __init__(self, base_model, M):
        super().__init__()
        self.base_model = base_model
        self.M = M

    def forward(self, input_ids):
        # base_model expected to return logits [M,B,T,V]
        out = self.base_model(input_ids)
        return out.logits if hasattr(out, "logits") else out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train(
    model,
    dataloader,
    M=8,
    alpha=0.5,
    lr=1e-4,
    lambda_kl=1e-4,
    steps=10_000,
):
    trainer = RenyiLoRATrainer(
        model=model,
        M=M,
        alpha=alpha,
        lr=lr,
        lambda_kl=lambda_kl,
    )

    model.train()
    for step, batch in enumerate(dataloader):
        metrics = trainer.training_step(batch)

        if step % 100 == 0:
            print(step, metrics)

        if step >= steps:
            break

