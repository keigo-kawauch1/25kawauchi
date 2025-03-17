import torch.utils.data
import numpy as np
import random

from learner.utils import FuelCellConfig

import yaml

with open('learner/config.yaml') as file:
    config = yaml.safe_load(file.read())

# データセット作成
class CreateData:
    def __init__(self, eq_config):

        self.matrix = eq_config.calculate_matrix()
        self.nb_mesh = eq_config.nb_mesh
        self.nb_sensor_measurement = eq_config.nb_sensor_measurement

    # $ (Ax, t) \times nb_data $ のデータセット作成
    def create(self, nb_data):

        x = np.random.rand(self.nb_mesh, nb_data)
        y = self.matrix @ x
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_low_freqency(self, nb_data):
        
        _, _, V = np.linalg.svd(self.matrix)
        coefficient = np.random.rand(16, nb_data)
        x = (V[0:16]).T @ coefficient
        y = self.matrix @ x
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_nonbasis(self, nb_data):
        
        U, _, V = np.linalg.svd(self.matrix)
        x = np.random.rand(self.nb_mesh, nb_data)
        y = self.matrix @ x
        x = V @ x
        y = U.T @ y 
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_pert(self, nb_data):
        x = np.ones((self.nb_mesh, nb_data))
        for i in range(nb_data):
            point = random.randint(0,self.nb_mesh)
            if point !=self.nb_mesh:
                x[point][i] = 0
        pert = np.random.randn(self.nb_mesh, nb_data) * config['data']['noise']
        y = self.matrix @ (x + pert)
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_pert_nocenter(self, nb_data):
        x = np.ones((self.nb_mesh, nb_data))
        for i in range(nb_data):
            point = random.randint(0,self.nb_mesh)
            if point !=self.nb_mesh and point != (self.nb_mesh-1)//2:
                x[point][i] = 0
        pert = np.random.randn(self.nb_mesh, nb_data) * config['data']['noise']
        y = self.matrix @ (x + pert)
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_pert_mag(self, nb_data):
        x = np.ones((self.nb_mesh, nb_data))
        for i in range(nb_data):
            point = random.randint(0,self.nb_mesh)
            if point !=self.nb_mesh:
                x[point][i] = 0
        pert = np.random.randn(self.nb_sensor_measurement, nb_data) * config['data']['noise']
        y = self.matrix @ x + pert
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)
    
    def create_mag(self, nb_data):
        x = np.random.rand(self.nb_mesh, nb_data)
        y = self.matrix @ x + np.random.randn(self.nb_sensor_measurement, nb_data) * config['data']['noise']
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32)   
    
    def create_two_layer_defects(self, nb_data):
        random_integer = np.random.randint(0,1+3*self.nb_mesh, nb_data)
        x = np.array([])
        for j in range(len(random_integer)):
            i = random_integer[j]
            current = np.ones(self.nb_mesh)
            if 0 < i <= self.nb_mesh:
                current[i%self.nb_mesh] = 0
            elif i <= 2*self.nb_mesh:
                current[i%self.nb_mesh] = 0.2 if ((i%int(self.nb_mesh**0.5) in [7,8,9]) and ((i%self.nb_mesh)//int(self.nb_mesh**0.5) in [7,8,9])) else 0.4
            elif i <= 3*self.nb_mesh:
                current[i%self.nb_mesh] = 0.8 if ((i%int(self.nb_mesh**0.5) in [7,8,9]) and ((i%self.nb_mesh)//int(self.nb_mesh**0.5) in [7,8,9])) else 0.6
            x = np.append(x, current)
        x = x.reshape(-1,self.nb_mesh).T
        y = self.matrix @ (x + np.random.randn(self.nb_mesh, nb_data) * config['data']['noise'])
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32) 

    def create_two_defects(self, nb_data):
        x = np.array([])
        for _ in range(nb_data):
            random_int = np.random.rand(1)
            current = np.ones(self.nb_mesh)
            i = np.random.randint(0,100)
            current[i] = 0
            if random_int > -1:
                j = np.random.randint(0,100)
                while j == i:
                    j = np.random.randint(0,100)
                current[j] = 0
            x = np.append(x, current)
        x = x.reshape(-1,self.nb_mesh).T
        noise = np.random.normal(loc=0, scale=1, size=(nb_data, self.nb_mesh))
        y = self.matrix @ (x +  noise.T * config['data']['noise'])
        return torch.tensor(y.T, dtype = torch.float32), torch.tensor(x.T, dtype = torch.float32) 
    
nb_data = config['data']['nb_data']

eq_config = FuelCellConfig(*config['data']['eq_config'])
data = CreateData(eq_config)

if config['data']['train_data'] == 'all':
    x, t = data.create(nb_data)
elif config['data']['train_data'] == 'low':
    x, t = data.create_low_freqency(nb_data)
elif config['data']['train_data'] == 'nonbasis':
    x, t = data.create_nonbasis(nb_data)
elif config['data']['train_data'] == 'pert':
    x, t = data.create_pert(nb_data)
elif config['data']['train_data'] == 'nocenter':
    x, t = data.create_pert_nocenter(nb_data)
elif config['data']['train_data'] == 'magpert':
    x, t = data.create_pert_mag(nb_data)
elif config['data']['train_data'] == 'mag':
    x, t = data.create_mag(nb_data)
elif config['data']['train_data'] == 'two_layer':
    x, t = data.create_two_layer_defects(nb_data)
elif config['data']['train_data'] == 'two_defects':
    x, t = data.create_two_defects(nb_data)
dataset = torch.utils.data.TensorDataset(x, t)

n_train = int(len(dataset) * config['data']['split_ratio'][0])
n_val = int(len(dataset) *config['data']['split_ratio'][1])
n_test = len(dataset) - n_train - n_val

train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])