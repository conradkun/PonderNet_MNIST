from torch._C import dtype
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

from losses import ReconstructionLoss, RegularizationLoss, Loss


class PonderMNIST(pl.LightningModule):
    '''
        PonderNet variant to perform image classification on MNIST. It is capable of
        adaptively choosing the number of steps for which to process an input.

        Parameters
        ----------
        n_hidden : int
            Hidden layer size of the propagated hidden state.

        n_hidden_lin :
            Hidden layer size of the underlying MLP.

        n_hidden_cnn : int
            Hidden layer size of the output of the underlying CNN.

        kernel_size : int
            Size of the kernel of the underlying CNN.

        max_steps : int
            Maximum number of steps the network is allowed to "ponder" for.

        lambda_p : float 
            Parameter of the geometric prior. Must be between 0 and 1.

        beta : float
            Hyperparameter to calculate the total loss.

        lr : float
            Learning rate.

        Modules
        -------
        cnn : CNN
            Learnable convolutional neural network to emgbed the image into a vector.

        mlp : MLP
            Learnable 3-layer machine learning perceptron to combine the hidden state with
            the image embedding.

        ouptut_layer : nn.Linear
            Linear module that serves as a multi-class classifier.

        lambda_layer : nn.Linear
            Linear module that generates the halting probability at each step.
    '''

    def __init__(self, n_hidden, n_hidden_lin, n_hidden_cnn, kernel_size, max_steps, lambda_p, beta, lr):
        super().__init__()

        # attributes
        self.n_classes = 10
        self.max_steps = max_steps
        self.lambda_p = lambda_p
        self.beta = beta
        self.n_hidden = n_hidden
        self.lr = lr

        # modules
        self.cnn = CNN(n_input=28, kernel_size=kernel_size, n_output=n_hidden_cnn)
        self.mlp = MLP(n_input=n_hidden_cnn + n_hidden, n_hidden=n_hidden_lin, n_output=n_hidden)
        self.outpt_layer = nn.Linear(n_hidden, self.n_classes)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        # losses
        self.loss_rec = ReconstructionLoss(nn.CrossEntropyLoss())
        self.loss_reg = RegularizationLoss(self.lambda_p, max_steps=self.max_steps, device=self.device)

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # save hparams on W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''
            Run the forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Batch of input features of shape `(batch_size, n_elems)`.

            Returns
            -------
            y : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the predictions for each step and each sample. In case
                `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.

            p : torch.Tensor
                Tensor of shape `(max_steps, batch_size)` representing
                the halting probabilities. Sums over rows (fixing a sample)
                are 1. In case `allow_halting=True` then the shape is
                `(steps, batch_size)` where `1 <= steps <= max_steps`.

            halting_step : torch.Tensor
                An integer for each sample in the batch that corresponds to
                the step when it was halted. The shape is `(batch_size,)`. The
                minimal value is 1 because we always run at least one step.
        '''
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
        halting_step = h.new_zeros((batch_size,), dtype=torch.long)  # stopping step

        # main loop
        for n in range(1, self.max_steps + 1):
            # obtain lambda_n
            if n == self.max_steps:
                lambda_n = h.new_ones(batch_size)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(h)).squeeze()

            if torch.any(lambda_n == 1) and n == 1:
                print('caught a 1')
                threshold = 0.01
                almost_ones = h.new_ones((1)) - 0.01
                lambda_n = torch.where(torch.abs(lambda_n - 1) > threshold, lambda_n, almost_ones)
                # raise ValueError

            # obtain output and p_n
            y_n = self.outpt_layer(h)
            p_n = un_halted_prob * lambda_n

            # append p_n, y_n
            p.append(p_n)
            y.append(y_n)

            # calculate halting step
            halting_step = torch.maximum(
                n
                * (halting_step == 0)
                * torch.bernoulli(lambda_n).to(torch.long),
                halting_step)

            # track unhalted probability and flip coin to halt
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # propagate to obtain h_n
            embedding = self.cnn(x)
            concat = torch.cat([embedding, h], 1)
            h = self.mlp(concat)

            # break if we are in inference and all elements have halting_step
            if not self.training and (halting_step > 0).sum() == batch_size:
                break

        return torch.stack(y), torch.stack(p), halting_step

    def training_step(self, batch, batch_idx):
        '''
            Perform the training step.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current training batch to train on.

            Returns
            -------
            loss : torch.Tensor
                Loss value of the current batch.
        '''
        loss, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('train/steps', steps)
        self.log('train/accuracy', acc)
        self.log('train/total_loss', loss.get_total_loss())
        self.log('train/reconstruction_loss', loss.get_rec_loss())
        self.log('train/regularization_loss', loss.get_reg_loss())

        return loss.get_total_loss()

    def validation_step(self, batch, batch_idx):
        '''
            Perform the validation step. Logs relevant metrics and returns
            the predictions to be used in a custom callback.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current validation batch to evaluate.

            Returns
            -------
            preds : torch.Tensor
                Predictions for the current batch.
        '''
        loss, preds, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('val/steps', steps)
        self.log('val/accuracy', acc)
        self.log('val/total_loss', loss.get_total_loss())
        self.log('val/reconstruction_loss', loss.get_rec_loss())
        self.log('val/regularization_loss', loss.get_reg_loss())

        # for custom callback
        return preds

    def test_step(self, batch, batch_idx):
        '''
            Perform the test step. Logs relevant metrics.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Current teest batch to evaluate.
        '''
        loss, _, acc, steps = self._get_loss_and_metrics(batch)

        # logging
        self.log('test/steps', steps)
        self.log('test/accuracy', acc)
        self.log('test/total_loss', loss.get_total_loss())
        self.log('test/reconstruction_loss', loss.get_rec_loss())
        self.log('test/regularization_loss', loss.get_reg_loss())

    def configure_optimizers(self):
        '''
            Configure the optimizers and learning rate schedulers.

            Returns
            -------
            config : dict
                Dictionary with `optimizer` and `lr_scheduler` keys, with an
                optimizer and a learning scheduler respectively.
        '''
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
        '''returns a list of callbacks'''
        early_stopping = EarlyStopping(monitor='val/accuracy', mode='max', patience=4)
        model_checkpoint = ModelCheckpoint(monitor="val/accuracy", mode='max')
        log_predictions = LogPredictionsCallback()
        return [early_stopping, model_checkpoint, log_predictions]

    def _get_loss_and_metrics(self, batch):
        '''
            Returns the losses, the predictions, the accuracy and the number of steps.

            Parameters
            ----------
            batch : (torch.Tensor, torch.Tensor)
                Batch to process.

            Returns
            -------
            loss : Loss
                Loss object from which all three losses can be retrieved.

            preds : torch.Tensor
                Predictions for the current batch.

            acc : torch.Tensor
                Accuracy obtained with the current batch.

            steps : torch.Tensor
                Average number of steps in the current batch.
        '''
        # extract the batch
        data, target = batch

        # forward pass
        y, p, halted_step = self(data)
        if torch.any(p == 0):
            raise ValueError('caught an inf!')

        # calculate the loss
        loss_rec_ = self.loss_rec(p, y, target)
        loss_reg_ = self.loss_reg(p)
        loss = Loss(loss_rec_, loss_reg_, self.beta)

        halted_index = (halted_step - 1).unsqueeze(0).unsqueeze(2).repeat(1, 1, self.n_classes)

        # calculate the accuracy
        logits = y.gather(dim=0, index=halted_index).squeeze()
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, target)

        # calculate the average number of steps
        steps = (halted_step * 1.0).mean()

        return loss, preds, acc, steps


class LogPredictionsCallback(Callback):
    '''
        Callback to log predictions from the validation set as the model improves.
    '''

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        '''Called when the validation batch ends.'''

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 8
            x, y = batch
            # we can directly use `wandb` for logging custom objects (image, video, audio, modecules and any other custom plot)
            wandb.log({'examples': [wandb.Image(x_i, caption=f'Ground Truth: {y_i}\nPrediction: {y_pred}')
                                    for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]})


class MLP(nn.Module):
    '''
        Simple 3-layer multi layer perceptron.

        Parameters
        ----------
        n_input : int
            Size of the input.

        n_hidden : int
            Number of units of the hidden layer.

        n_ouptut : int
            Size of the output.
    '''

    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__()
        self.i2h = nn.Linear(n_input, n_hidden)
        self.h2o = nn.Linear(n_hidden, n_output)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        '''forward pass'''
        x = F.relu(self.i2h(x))
        x = self.droput(x)
        x = F.relu(self.h2o(x))
        return x


class CNN(nn.Module):
    '''
        Simple convolutional neural network.

        Parameters
        ----------
        n_input : int
            Size of the input image. We assume the image is a square,
            and `n_input` is the size of one side.

        n_ouptut : int
            Size of the output.

        kernel_size : int
            Size of the kernel.
    '''

    def __init__(self, n_input=28, n_output=50, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.conv2_drop = nn.Dropout2d()

        # calculate size of convolution output
        self.lin_size = floor((floor((n_input - (kernel_size - 1)) / 2) - (kernel_size - 1)) / 2)
        self.fc1 = nn.Linear(self.lin_size ** 2 * 20, n_output)

    def forward(self, x):
        '''forward pass'''
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x
