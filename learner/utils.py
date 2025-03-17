import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F

from typing import Any

def calculate_square_points(r, n):
    step = r / (n + 1)  # 点間の間隔
    half_r = r / 2

    # 各辺の座標を計算
    top = [(-half_r + step * i, half_r) for i in range(1, n + 1)]  # 上辺
    right = [(half_r, half_r - step * i) for i in range(1, n + 1)]  # 右辺
    bottom = [(half_r - step * i, -half_r) for i in range(1, n + 1)]  # 下辺
    left = [(-half_r, -half_r + step * i) for i in range(1, n + 1)]  # 左辺

    # 全ての点を結合
    points = top + right + bottom + left
    return np.array(points)

# 燃料電池逆問題の設定に使う関数・クラス

class FuelCellConfig:
    """
    センサとメッシュ関連の設定を行う.    
    """
    def __init__(
                self,
                n : int, # メッシュの分割数
                m : int, # センサの配置数
                r : float, # センサの中心からの距離
                s : float # メッシュのサイズ
            ) -> None :
        
        # 各種設定用パラメータ
        self.nb_mesh_split = n
        self.nb_mesh = n**2
        self.nb_sensor = m
        self.nb_sensor_measurement = 2*m
        self.radius = r
        self.size = s

        # NOTE : センサの位置は、原点中心, 半径 self.raidius の円形で, 時計回りに12時から12時までの順番になっている. 
        self.sensor_x_position = self.radius * np.sin((2*np.pi) * np.arange(self.nb_sensor) / self.nb_sensor)
        self.sensor_y_position = self.radius * np.cos((2*np.pi) * np.arange(self.nb_sensor) / self.nb_sensor)
        #self.sensor_x_position = [x[0] for x in calculate_square_points(60,5)]
        #self.sensor_y_position = [x[1] for x in calculate_square_points(60,5)]

        # NOTE : メッシュは左上から右に向かう順番に番号が定まっている.
        self.mesh_x_position = np.array([self.size * (i%self.nb_mesh_split)- (self.nb_mesh_split/2 - 1/2) * self.size for i in range(self.nb_mesh)])
        self.mesh_y_position = np.array([(self.nb_mesh_split/2 - 1/2) * self.size - self.size * (i//self.nb_mesh_split) for i in range(self.nb_mesh)])

    # NOTE : センサ番号 i からメッシュ番号 j までの距離の2条を計算する.
    def source_sensor_distance(
                self,
                i : int, # センサ番号
                j : int # メッシュ番号
            ) -> float:

        return ((self.mesh_y_position[j]-self.sensor_y_position[i])**2 + (self.mesh_x_position[j]-self.sensor_x_position[i])**2)

    # 支配方程式中の行列 A を求める.
    def calculate_matrix(self):
        
        A = np.zeros((self.nb_sensor_measurement, self.nb_mesh))

        for i in range(self.nb_sensor_measurement):
            for j in range(self.nb_mesh):
                A[i][j] = -(self.sensor_y_position[i//2]-self.mesh_y_position[j]) / self.source_sensor_distance(i//2,j) if i%2 == 0 else (self.sensor_x_position[i//2]-self.mesh_x_position[j]) / self.source_sensor_distance(i//2,j)
        return A

# reconstruct current distributiob
def reconstruct(model, observe):
    return model(torch.from_numpy(observe.astype(np.float32)).unsqueeze(0)).to('cpu').detach().numpy()[0]

# calculate net jacobian
def Calculate_jacobian(model, y):
    y = torch.tensor(y, dtype = torch.float32)
    J = model.state_dict()['fc1.weight']
    weight = model.state_dict()['fc1.weight']
    output = weight @ y

    bias = model.state_dict()['fc1.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J #relu型

    weight = model.state_dict()['fc2.weight']
    output = weight @ output
    J=weight @ J

    bias = model.state_dict()['fc2.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J

    weight = model.state_dict()['fc3.weight']
    output = weight @ output
    J=weight @ J

    bias = model.state_dict()['fc3.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J

    weight = model.state_dict()['fc4.weight']
    output = weight @ output
    J=weight @ J

    return J.numpy()

def create_false_positive(model, x, A, max_epoch = 10000, lamda = 200, gamma = 0.00001, eta = 0.0001):
    
    for param in model.parameters():
        param.requires_grad = False

    x = torch.tensor(x, dtype = torch.float32).to('cuda')
    A = torch.tensor(A, dtype = torch.float32).to('cuda')
    tmp_x = model(A @ x)
    nb_mesh = len(A[0])
    epoch = 0
    
    r = torch.random.rand(nb_mesh).to(torch.float32).to('cuda')
    v = torch.zeros(nb_mesh).to(torch.float32).to('cuda')


    while epoch < max_epoch:
        r.requires_grad = True

        pert_y = (A @ (x + r)).T
        pert_x = model(pert_y)

        loss = F.mse_loss(pert_x,tmp_x) - lamda * torch.linalg.norm(A @ r)**2 / 2
        loss.backward()
        r_grad = r.grad

        v = gamma * v + eta * r_grad
        i+=1

        r = r.detach()

    return r

# 特異ベクトルを描画
def show_singular_vector(u, vmin = -0.5, vmax = 0.5, title = 'singular vector'):

    fig = plt.figure(figsize = (5,10))
    fig.subplots_adjust(left=0.075,right=0.95,bottom=0.05,top=0.50,wspace=0.15,hspace=0.10)
    for i in range(25):
        ax = fig.add_subplot(5,5,i+1)
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        ax.imshow(u.T[i].reshape(5,5), vmin = vmin, vmax = vmax)

    fig.suptitle(title, x = 0.55, y = 0.52)
    norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)
    cmap = plt.get_cmap("viridis")
    cax = fig.add_axes([1, 0.025, 0.05, 0.5])
    plt.colorbar(mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax)
    plt.show()


def Calculate_jacobian(model, y):
    y = torch.tensor(y, dtype = torch.float32)
    J = model.state_dict()['fc1.weight']
    weight = model.state_dict()['fc1.weight']
    output = weight @ y

    bias = model.state_dict()['fc1.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J #relu型

    if config['net']['batch']:
        weight = model.state_dict()['batchnorm1.weight']
        bias = model.state_dict()['batchnorm1.bias']
        mean = model.state_dict()['batchnorm1.running_mean']
        var = model.state_dict()['batchnorm1.running_var']
        output = (output - mean) * weight / (var + torch.ones(1000) * 1e-5)**0.5 + bias
        J = torch.diag(weight / (var + torch.ones(1000) * 1e-5)**0.5) @ J

    weight = model.state_dict()['fc2.weight']
    output = weight @ output
    J=weight @ J

    bias = model.state_dict()['fc2.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J

    if config['net']['batch']:
        weight = model.state_dict()['batchnorm2.weight']
        bias = model.state_dict()['batchnorm2.bias']
        mean = model.state_dict()['batchnorm2.running_mean']
        var = model.state_dict()['batchnorm2.running_var']
        output = (output - mean) * weight / (var + torch.ones(1000) * 1e-5)**0.5 + bias
        J = torch.diag(weight / (var + torch.ones(1000) * 1e-5)**0.5) @ J

    weight = model.state_dict()['fc3.weight']
    output = weight @ output
    J=weight @ J

    bias = model.state_dict()['fc3.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J

    if config['net']['batch']:
        weight = model.state_dict()['batchnorm3.weight']
        bias = model.state_dict()['batchnorm3.bias']
        mean = model.state_dict()['batchnorm3.running_mean']
        var = model.state_dict()['batchnorm3.running_var']
        output = (output - mean) * weight / (var + torch.ones(1000) * 1e-5)**0.5 + bias
        J = torch.diag(weight / (var + torch.ones(1000) * 1e-5)**0.5) @ J

    weight = model.state_dict()['fc4.weight']
    output = weight @ output
    J=weight @ J

    bias = model.state_dict()['fc4.bias']
    output=output + bias
    J=torch.diag(torch.heaviside(output,torch.zeros(len(output))))@J

    return J.numpy()
