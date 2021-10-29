import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        total_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_hat[n], y)).mean()
            total_loss = total_loss + loss

        return total_loss


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p: float, max_steps: int = 1_000, device=None):
        super().__init__()

        p_g = torch.zeros((max_steps,), device=device)
        not_halted = 1.

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, p: torch.Tensor):
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)
