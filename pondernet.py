import wandb
from math import floor

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from losses import ReconstructionLoss, RegularizationLoss


class PonderCNN(pl.LightningModule):
    def __init__(self, n_classes, n_input, n_hidden, n_hidden_cnn, n_hidden_lin, kernel_size, max_steps, lambda_p, beta, lr):
        super().__init__()

        # attributes
        self.n_classes = n_classes
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.beta = beta
        self.n_hidden = n_hidden
        self.lr = lr

        # modules
        self.cnn = CNN(n_input=n_input, kernel_size=kernel_size, n_hidden=n_hidden_cnn)
        self.mlp = MLP(n_input=n_hidden_cnn + n_hidden, n_hidden=n_hidden_lin, n_output=n_hidden)
        self.outpt_layer = nn.Linear(n_hidden, n_classes)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        # losses
        self.loss_rec = ReconstructionLoss(nn.CrossEntropyLoss())
        self.loss_reg = RegularizationLoss(self.lambda_p, max_steps=self.max_steps, device=self.device)

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hparams on W&B
        self.save_hyperparameters()

    def forward(self, x):
        # extract batch size for QoL
        batch_size = x.shape[0]

        # propagate to get h_1
        h = x.new_zeros((batch_size, self.n_hidden))
        embedding = self.cnn(x)
        concat = torch.cat([embedding, h], 1)
        h = self.mlp(concat)

        # lists to save p_n, y_n
        p = []
        y = []

        # vectors to save intermediate values
        un_halted_prob = h.new_ones((batch_size,))  # unhalted probability till step n
        halted = h.new_zeros((batch_size,))  # stopping step

        # vector to save the outputs at stopping time for inference
        y_m = h.new_zeros((batch_size, self.n_classes))
        step_halted = h.new_zeros((batch_size,))

        # main loop
        for n in range(1, self.max_steps + 1):
            # obtain lambda_n
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h))[:, 0]

            # obtain output and p_n
            y_n = self.outpt_layer(h)
            p_n = un_halted_prob * lambda_n

            # track unhalted probability and flip coin to halt
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            halt = torch.bernoulli(lambda_n) * (1 - halted)

            # append p_n, y_n
            p.append(p_n)
            y.append(y_n)

            # update y_m
            halt_tiled = halt.view(-1, 1).repeat((1, self.n_classes))
            y_m = y_m * (1 - halt_tiled) + y_n * halt_tiled

            # update halted
            halted = halted + halt
            step_halted = step_halted + n * halt

            # propagate to obtain h_n
            embedding = self.cnn(x)
            concat = torch.cat([embedding, h], 1)
            h = self.mlp(concat)

            # break if we are in inference and all elements have halted
            if not self.training and halted.sum() == batch_size:
                break

        return torch.stack(p), torch.stack(y), step_halted, y_m

    def training_step(self, batch, batch_idx):
        _, steps, loss, acc = self._get_preds_steps_loss_acc(batch)

        # logging
        self.log('train/steps', steps)
        self.log('train/accuracy', acc)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, steps, loss, acc = self._get_preds_steps_loss_acc(batch)

        # logging
        self.log('val/steps', steps)
        self.log('val/accuracy', acc)
        self.log('val/loss', loss)

        # for custom callback
        return preds

    def test_step(self, batch, batch_idx):
        _, steps, loss, acc = self._get_preds_steps_loss_acc(batch)

        # logging
        self.log('test/steps', steps)
        self.log('test/accuracy', acc)
        self.log('test/loss', loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode='max', verbose=True),
                "monitor": 'val/accuracy',
                "interval": 'epoch',
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        early_stopping = EarlyStopping(monitor='val/accuracy', mode='max', patience=3)
        model_checkpoint = ModelCheckpoint(monitor="val/accuracy", mode='max')
        log_predictions = LogPredictionsCallback()
        return [early_stopping, model_checkpoint]

    def _get_preds_steps_loss_acc(self, batch):
        # extract the batch
        data, target = batch

        # forward pass
        p, y_hat, halted, y_hat_sampled = self(data)

        # calculate the loss
        loss_rec_ = self.loss_rec(p, y_hat, target)
        loss_reg_ = self.loss_reg(p)
        loss = loss_rec_ + self.beta * loss_reg_

        # calculate the accuracy
        preds = torch.argmax(y_hat_sampled, dim=1)
        acc = self.accuracy(preds, target)

        # calculate the average number of steps
        steps = halted.mean()

        return preds, steps, loss, acc


class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 8
            x, y = batch
            # we can directly use `wandb` for logging custom objects (image, video, audio, modecules and any other custom plot)
            self.log({'examples': [wandb.Image(x_i, caption=f'Ground Truth: {y_i}\nPrediction: {y_pred}')
                                   for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]})


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x


class CNN(nn.Module):
    def __init__(self, n_input=28, n_hidden=50, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # calculate size of convolution output
        self.lin_size = floor((floor((n_input - (kernel_size - 1)) / 2) - (kernel_size - 1)) / 2)
        self.fc1 = nn.Linear(self.lin_size ** 2 * 20, n_hidden)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
