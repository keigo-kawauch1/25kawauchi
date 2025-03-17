from learner.train import TrainNet
from learner.validate import ValidationNet
from learner.utils import FuelCellConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import argparse

parser = argparse.ArgumentParser(description='Change yaml to define network')
parser.add_argument('--test', type=bool, default=False, help='change loading yaml when testing')
parser.add_argument('--time', type=str, default='', help='directry name of config.yaml')
args = parser.parse_args([])


if args.test:
    with open('weight/{}/config.yaml'.format(args.time)) as file:
        config = yaml.safe_load(file.read())
else:
    with open('learner/config.yaml') as file:
        config = yaml.safe_load(file.read())

eq_config = FuelCellConfig(*config['data']['eq_config'])
A = torch.tensor(eq_config.calculate_matrix(), dtype = torch.float32).to('cuda')

class InverseNet(TrainNet, ValidationNet):

    def __init__(self, input_size = eq_config.nb_sensor_measurement, hidden_layer = config['net']['hidden_layer'], output_size = eq_config.nb_mesh, batch_size = 40):

        super(InverseNet, self).__init__()
        for i in range(config['net']['nb_layer']):
            if i == 0:
                self.fc1 = nn.Linear(input_size, hidden_layer[0])
                #self.dropout1 = nn.Dropout(config['net']['dropout_rate'])
                if config['net']['batch']:
                    self.batchnorm1 = nn.BatchNorm1d(hidden_layer[0])
            elif i == config['net']['nb_layer'] - 1:
                exec('self.fc{} = nn.Linear(hidden_layer[-1], output_size)'.format(config['net']['nb_layer']))
            else:
                exec('self.fc{} = nn.Linear(hidden_layer[{}], hidden_layer[{}])'.format(i+1, i-1, i))
                if config['net']['dropout']:
                    exec('self.dropout{} = nn.Dropout({})'.format(i+1, config['net']['dropout_rate']))
                if config['net']['batch']:
                    exec('self.batchnorm{} = nn.BatchNorm1d(hidden_layer[{}])'.format(i+1, i))

        self.batch_size = batch_size

    # Forwardステップ
    def forward(self, x):

        for i in range(1, config['net']['nb_layer']):
            x = eval('self.fc{}(x)'.format(i))
            if config['net']['batch']:
                x = eval('self.batchnorm{}(x)'.format(i))
            x = eval('F.relu(x)')
            if config['net']['dropout']:  
                x = eval('self.dropout{}(x)'.format(i))



        x = eval('self.fc{}(x)'.format(config['net']['nb_layer']))

        return x

    # 損失関数
    def lossfun(self, y, t):
        if config['net']['loss'] == 'causal':
            loss = F.mse_loss(y, t)
        else:
            loss = F.mse_loss(A@(y.T), A@(t.T))
    
        return loss
    
    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr = 0.001)