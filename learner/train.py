from lightning.pytorch.core.module import LightningModule 
import torch

import learner.data as data

import yaml

with open('learner/config.yaml') as file:
    config = yaml.safe_load(file.read())

# 学習データに対する処理

class TrainNet(LightningModule):

    def train_dataloader(self):

        train = data.train
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True, num_workers = config['train']['num_workers'])
    
    def training_step(self, batch, batch_nb):

        x,t = batch
        y = self.forward(x)
        loss = self.lossfun(y,t)
        results = {'loss' : loss}
        return results