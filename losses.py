import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    '''
        Computes the weighted average of the given loss across steps according to
        the probability of stopping at each step.

        Parameters
        ----------
        loss_func : callable
            Loss function accepting true and predicted labels. It should output
            a loss item for each element in the input batch.
    '''

    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, p: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor):
        '''
            Compute the loss.

            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, of shape `(max_steps, batch_size)`.

            y_pred : torch.Tensor
                Predicted outputs, of shape `(max_steps, batch_size)`.

            y : torch.Tensor
                True targets, of shape `(batch_size)`.

            Returns
            -------
            total_loss : torch.Tensor
                Scalar representing the reconstruction loss.
        '''
        total_loss = p.new_tensor(0.)

        for n in range(p.shape[0]):
            loss = (p[n] * self.loss_func(y_pred[n], y)).mean()
            total_loss = total_loss + loss

        return total_loss


class RegularizationLoss(nn.Module):
    '''
        Computes the KL-divergence between the halting distribution generated
        by the network and a geometric distribution with parameter `lambda_p`.

        Parameters
        ----------
        lambda_p : float
            Parameter determining our prior geometric distribution.

        max_steps : int
            Maximum number of allowed pondering steps.
    '''

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
        '''
            Compute the loss.

            Parameters
            ----------
            p : torch.Tensor
                Probability of halting at each step, representing our
                halting distribution.

            Returns
            -------
            loss : torch.Tensor
                Scalar representing the regularization loss.
        '''
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)
        return self.kl_div(p.log(), p_g)


class Loss:
    '''
        Class to group the losses together and calculate the total loss.

        Parameters
        ----------
        rec_loss : torch.Tensor
            Reconstruction loss obtained from running the network.

        reg_loss : torch.Tensor
            Regularization loss obtained from running the network.

        beta : float
            Hyperparameter to calculate the total loss.
    '''

    def __init__(self, rec_loss, reg_loss, beta):
        self.rec_loss = rec_loss
        self.reg_loss = reg_loss
        self.beta = beta

    def get_rec_loss(self):
        '''returns the reconstruciton loss'''
        return self.rec_loss

    def get_reg_loss(self):
        '''returns the regularization loss'''
        return self.reg_loss

    def get_total_loss(self):
        '''returns the total loss'''
        return self.rec_loss + self.beta * self.reg_loss
