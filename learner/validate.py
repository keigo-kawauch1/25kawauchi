from lightning.pytorch.core.module import LightningModule 
import torch

import learner.data as data

import yaml

with open('learner/config.yaml') as file:
    config = yaml.safe_load(file.read())

# 検証データに対する処理

class ValidationNet(LightningModule):

    def val_dataloader(self):

        val = data.val
        return torch.utils.data.DataLoader(val, self.batch_size, num_workers = config['validate']['num_workers'])
    
    def validation_step(self, batch, batch_nb):

        x,t = batch
        y = self.forward(x)
        loss = self.lossfun(y,t)
        results = {'val_loss' : loss}
        self.log("val_loss", loss)
        return results
    
    def validation_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        results = {'val_loss' : avg_loss}
        self.log('avg_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        return results