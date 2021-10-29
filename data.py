
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl


class MNIST_DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', batch_size=256, num_workers=8, train_transform=None, test_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if train_transform is None:
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.test_transform = test_transform

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit, validate, test or predict step'''
        # we set up only relevant datasets when stage is specified (automatically set by Lightning)
        if stage in [None, 'fit', 'validate']:
            mnist_train = MNIST(self.data_dir, train=True, transform=self.train_transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.test_transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return mnist_test
