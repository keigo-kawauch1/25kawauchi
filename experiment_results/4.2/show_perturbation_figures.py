import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.cm as cm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from learner.net import InverseNet
from learner import utils
import matplotlib as mpl
import matplotlib.cm as cm
import yaml

model = InverseNet().to("cpu")
time = 'weight/2025-01-04:15:56/'
checkpoint = torch.load(time + 'weight.ckpt')
model.load_state_dict(checkpoint['state_dict'])

with open(time + 'config.yaml') as file:
    config = yaml.safe_load(file.read())

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


A = FuelCellConfig(*config['data']['eq_config']).calculate_matrix()

device = "cuda"

def calculate_gradient(f, p, y, A, r, lamda):
    u = y + torch.matmul(A, r)
    u.requires_grad = True
    g_u = torch.norm(f(u) - p, p=2)**2 / 2
    nabla_u_g = torch.autograd.grad(g_u, u, retain_graph=True)[0]
    nabla_r_Q = torch.matmul(A.T, nabla_u_g) - lamda * r
    return nabla_r_Q

def calculate_jacobian(f,y):
    torch_y = torch.tensor(y, device=device, dtype=torch.float32)
    torch_y.requires_grad = True
    f = f.to(device)
    torch_x = f(torch_y)
    J = torch.stack([torch.autograd.grad(x,torch_y, retain_graph=True)[0] for x in torch_x])
    f.to("cpu")
    return J.detach().to("cpu").numpy()

def find_unstable_perturbation(y, x, f, A, M, lamda, gamma, eta, tau, p, show_plot = False, stop_norm = 1e+2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f = f.to(device)
    
    x = torch.tensor(x, device=device, dtype=torch.float32)
    A = torch.tensor(A, device=device, dtype=torch.float32)
    p = torch.tensor(p, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)

    v = torch.zeros_like(x, device=device)
    r = tau * torch.rand_like(x, device=device)

    list_norm = []
    list_r = []
    norm = 0
    for i in range(M):
        nabla_r_Q = calculate_gradient(f, p, y, A, r, lamda)
        v = gamma * v + eta * nabla_r_Q
        r = r + v
        list_norm.append(torch.linalg.norm(r).to("cpu").numpy())
        if torch.linalg.norm(r) > norm:
            norm = 1e-3 * (torch.ceil(torch.linalg.norm(r) * 1e+3))
            list_r.append(r.cpu().detach().numpy())

        if norm >= stop_norm:
            M = i + 1
            break
    f.to("cpu")

    return list_r 


def calculate_defect_score(c):
    current = c.reshape(10,10)
    score = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if i == 0:
                if j == 0:
                    score[i][j] = (current[i+1][j] - current[i][j]) + (current[i][j+1] - current[i][j]) + 2 * (1 - current[i][j])
                elif j == 9:
                    score[i][j] = (current[i+1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + 2 * (1 - current[i][j])
                else:
                    score[i][j] = (current[i+1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + (current[i][j+1] - current[i][j]) + (1 - current[i][j])
            elif i == 9:
                if j == 0:
                    score[i][j] = (current[i-1][j] - current[i][j]) + (current[i][j+1] - current[i][j]) + 2 * (1 - current[i][j])
                elif j == 9:
                    score[i][j] = (current[i-1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + 2 * (1 - current[i][j])
                else:
                    score[i][j] = (current[i-1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + (current[i][j+1] - current[i][j]) + (1 - current[i][j])
            else:
                if j == 0:
                    score[i][j] =  (current[i-1][j] - current[i][j]) + (current[i+1][j] - current[i][j]) + (current[i][j+1] - current[i][j]) + (1 - current[i][j])
                elif j == 9:
                    score[i][j] =  (current[i-1][j] - current[i][j]) + (current[i+1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + (1 - current[i][j])
                else:
                    score[i][j] = (current[i-1][j] - current[i][j]) + (current[i][j-1] - current[i][j]) + (current[i][j+1] - current[i][j]) + (current[i][j-1] - current[i][j])
    
    return score

def count_defect_number(score, alpha) -> int:
    count = 0
    for i in range(10):
        for j in range(10):
            if score[i][j] > alpha:
                if i != 0 and j != 0:
                    if score[i-1][j] <= alpha and score[i][j-1] <= alpha:
                        count += 1
                elif i != 0:
                    if score[i-1][j] <= alpha:
                        count += 1
                else:
                    if score[i][j-1] <= alpha:
                        count += 1
    return count

def find_defect_position(c, alpha = 1.5):
    current = c.reshape(10,10)
    score = calculate_defect_score(c)
    
    defect_position = np.ones((10,10))

    for i in range(10):
        for j in range(10):
            if score[i][j] >= alpha:
                defect_position[i][j] = 0

    return defect_position

def relu(y):
    return np.array([max(0,z) for z in y])

def H(y):
    return np.diag([1 if z>=0 else 0 for z in y])

def calculate_approximation_error(model,y,pert):
    nb_layer = config['net']['nb_layer']
    W1 = model.state_dict()["fc1.weight"].numpy()
    b1 = model.state_dict()["fc1.bias"].numpy()
    if nb_layer >= 2:
        W2 = model.state_dict()["fc2.weight"].numpy()
        b2 = model.state_dict()["fc2.bias"].numpy()
        P1 = W1@y+W1@A@pert+b1
        N1 = W1@y+b1

        if nb_layer >= 3:
            W3 = model.state_dict()["fc3.weight"].numpy()
            b3 = model.state_dict()["fc3.bias"].numpy()
            P2 = W2@relu(P1)+b2
            N2 = W2@relu(N1)+b2

            if nb_layer >= 4:
                W4 = model.state_dict()["fc4.weight"].numpy()
                b4 = model.state_dict()["fc4.bias"].numpy()
                P3 = W3@relu(P2)+b3
                N3 = W3@relu(N2)+b3

                if nb_layer >= 5:
                    print("UnExpected model parameter")

                else:
                    Q = W4 @ (H(P3) - H(N3)) @ P3 + W4 @ H(N3) @ W3 @ (H(P2) - H(N2)) @ P2 + W4 @ H(N3) @ W3 @ H(N2) @ W2 @ (H(P1) - H(N1)) @ P1
                    return Q

            else:
                Q = W3 @ (H(P2) - H(N2)) @ P2 + W3 @ H(N2) @ W2 @ (H(P1) - H(N1)) @ P1
                return Q
        else:
            Q = W2 @ (H(W1@y+W1@A@pert+b1) - H(W1@y+b1)) @ (W1@y+W1@A@pert+b1)
            return Q
    else:
        print("UnExpected model parameter")


def find_approximation_range(model,y,pert):
    nb_layer = config['net']['nb_layer']
    W1 = model.state_dict()["fc1.weight"].numpy()
    b1 = model.state_dict()["fc1.bias"].numpy()

    list_range = []

    if nb_layer >= 2:
        W2 = model.state_dict()["fc2.weight"].numpy()
        b2 = model.state_dict()["fc2.bias"].numpy()
        N1 = W1@y+b1

        numerator = W2@H(N1)@W1@y
        numerator += b2
        numerator += W2@H(N1)@b1

        denominator = W2@H(N1)@W1@A@pert

        list_range.append(np.min(relu([n/d for n,d in zip(numerator,denominator)])))

        if nb_layer >= 3:
            W3 = model.state_dict()["fc3.weight"].numpy()
            b3 = model.state_dict()["fc3.bias"].numpy()
            N2 = W2@relu(N1)+b2

            numerator = W3@H(N2)@W2@H(N1)@W1@y
            numerator += b3
            numerator += W3@H(N2)@b2
            numerator += W3@H(N2)@W2@H(N1)@b1

            denominator = W3@H(N2)@W2@H(N1)@W1@A@pert

            list_range.append(np.min(relu([n/d for n,d in zip(numerator,denominator)])))

            if nb_layer >= 4:
                W4 = model.state_dict()["fc4.weight"].numpy()
                b4 = model.state_dict()["fc4.bias"].numpy()
                N3 = W3@relu(N2)+b3

                numerator = W4@H(N3)@W3@H(N2)@W2@H(N1)@W1@y
                numerator += b4
                numerator += W4@H(N3)@b3
                numerator += W4@H(N3)@W3@H(N2)@b2
                numerator += W4@H(N3)@W3@H(N2)@W2@H(N1)@b1

                denominator = W4@H(N3)@W3@H(N2)@W2@H(N1)@W1@A@pert

                list_range.append(np.min(relu([n/d for n,d in zip(numerator,denominator)])))

                if nb_layer >= 5:
                    print("UnExpected model parameter")
                    return 0

    else:
        print("UnExpected model parameter")
        return 0

    return list_range

def create_multistep_perturbation(model,y,size):
    list_pert = [np.zeros(100)]
    list_singular_value = []
    list_step_size = [0]
    list_reconstruction_approx = [np.zeros(100)]
    list_approx_error = [0]

    p = utils.reconstruct(model,y)

    while np.linalg.norm(list_pert[-1]) < size:
        print(f"size {np.linalg.norm(list_pert[-1])} / {size}")
        J = calculate_jacobian(model,y+A@list_pert[-1])
        U,S,V = np.linalg.svd(J@A)
        list_singular_value.append(S[0])
        pert = V[0]
        plus_step_size = min(find_approximation_range(model,y+A@list_pert[-1],pert)) + 0.001
        minus_step_size = min(find_approximation_range(model,y+A@list_pert[-1],-pert)) + 0.001

        plus_rec_error = np.linalg.norm(utils.reconstruct(model,y+A@list_pert[-1]+plus_step_size*A@pert)-p)
        minus_rec_error = np.linalg.norm(utils.reconstruct(model,y+A@list_pert[-1]-minus_step_size*A@pert)-p)

        if plus_rec_error > minus_rec_error:
            if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]+plus_step_size*pert):
                if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]-minus_step_size*pert):
                    break
                else:
                    list_pert.append(list_pert[-1]-minus_step_size*pert)
                    list_step_size.append(minus_step_size)
                    list_reconstruction_approx.append(list_reconstruction_approx[-1] - U.T[0]*S[0]*minus_step_size)
                    list_approx_error.append(list_approx_error[-1]+np.linalg.norm(calculate_approximation_error(model,y+A@list_pert[-2],-minus_step_size*pert)))
            else:
                list_pert.append(list_pert[-1]+plus_step_size*pert)
                list_step_size.append(plus_step_size)
                list_reconstruction_approx.append(list_reconstruction_approx[-1] + U.T[0]*S[0]*plus_step_size)
                list_approx_error.append(list_approx_error[-1]+np.linalg.norm(calculate_approximation_error(model,y+A@list_pert[-2],plus_step_size*pert)))

        else:
            if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]-minus_step_size*pert):
                if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]+plus_step_size*pert):
                    break
                else:
                    list_pert.append(list_pert[-1]+plus_step_size*pert)
                    list_step_size.append(plus_step_size)
                    list_reconstruction_approx.append(list_reconstruction_approx[-1] + U.T[0]*S[0]*plus_step_size)
                    list_approx_error.append(list_approx_error[-1]+np.linalg.norm(calculate_approximation_error(model,y+A@list_pert[-2],plus_step_size*pert)))

            else:
                list_pert.append(list_pert[-1]-minus_step_size*pert)
                list_step_size.append(minus_step_size)
                list_reconstruction_approx.append(list_reconstruction_approx[-1] - U.T[0]*S[0]*minus_step_size)
                list_approx_error.append(list_approx_error[-1]+np.linalg.norm(calculate_approximation_error(model,y+A@list_pert[-2],-minus_step_size*pert)))

    return list_pert,list_singular_value,list_step_size,list_reconstruction_approx,list_approx_error



defect_1 = 21
defect_2 = 78
ADD_NOISE = False
noise_level = 0.02


x = np.ones(100)
x[defect_1] = 0
x[defect_2] = 0


if ADD_NOISE:
    y = A @ (x + noise_level*np.random.randn(100))
else:
    y = A @ x

p = utils.reconstruct(model, y)
J = calculate_jacobian(model, y)

U,S,V = np.linalg.svd(J@A)


print(f"defect position is {defect_1}, {defect_2}")
print(f"maximum singular value is {S[0]}")

stop_norm = 0.071

plot_norms = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07]


### proposed perturbation
jacobian_pert = V[0]
list_jacobian_pert_norm = [0.001* i for i in range(int(stop_norm/0.001))]
list_jacobian_error = [np.linalg.norm(utils.reconstruct(model,y+A@jacobian_pert*i) - p) for i in list_jacobian_pert_norm]

list_singular_value_approximation = [S[0]*i for i in list_jacobian_pert_norm]

list_approximation_error = [np.linalg.norm(calculate_approximation_error(model,y,jacobian_pert*i)) for i in list_jacobian_pert_norm]
upper_bounds = [jac + app for jac,app in zip(list_singular_value_approximation,list_approximation_error)]
lower_bounds = [jac - app for jac,app in zip(list_singular_value_approximation,list_approximation_error)]


### multi step perturbation

list_multistep_pert ,list_multistep_svalue, list_mustistep_stepsize, list_multistep_reconstruction_approx, list_multistep_residual_error = create_multistep_perturbation(model,y,size = stop_norm)
list_multistep_reconstruction_error = [np.linalg.norm(utils.reconstruct(model,y+A@mpert) - p) for mpert in list_multistep_pert]
list_multistep_pert_norm = [np.linalg.norm(mpert) for mpert in list_multistep_pert]
multistep_upper_bounds = [np.linalg.norm(jac) + app for jac,app in zip(list_multistep_reconstruction_approx,list_multistep_residual_error)]
multistep_lower_bounds = [np.linalg.norm(jac) - app for jac,app in zip(list_multistep_reconstruction_approx,list_multistep_residual_error)]

plot_norms = [pnorm for pnorm in plot_norms if pnorm < list_multistep_pert_norm[-1]]

### ノルムに応じた擾乱の形成

multi_indices = np.searchsorted(list_multistep_pert_norm, plot_norms, side='right')
list_plot_multi_pert =  [list_multistep_pert[ind] for ind in multi_indices]
list_plot_multi_reconstruction_approx = [list_multistep_reconstruction_approx[ind] for ind in multi_indices]

### 
def calculate_magnitude_from_perturbation(pert):
    magnetic_flux = A@pert

    flux_u = np.array([magnetic_flux[2*i] for i in range(20)])
    flux_v = np.array([magnetic_flux[2*i+1] for i in range(20)])

    magnitude = np.sqrt(flux_u**2 + flux_v**2)

    flux_u /= np.max(magnitude)
    flux_v /= np.max(magnitude)

    return flux_u,flux_v,magnitude

sensor_x = FuelCellConfig(*config['data']['eq_config']).sensor_x_position
sensor_y = FuelCellConfig(*config['data']['eq_config']).sensor_y_position





fig1 = plt.figure()
for pert,norm in zip(list_plot_multi_pert,plot_norms):
    count = 2
    plt.figure(fig1.number)
    plt.plot([i for i in range(1,41)], np.abs(V@(pert/np.linalg.norm(pert)))[:40], label=f"perturbation norm {norm}")
    plt.xlabel("number of singular vectors")
    plt.ylabel("abs. inner product of singular vectors")
    exec(f"fig{count} = plt.figure(figsize=(12,5))")
    exec(f"plt.figure(fig{count}.number)")
    J_temp = calculate_jacobian(model,y+A@pert)
    U_temp,S_temp,V_temp =  np.linalg.svd(J_temp@A)

    if norm == 0:
        for j in range(1,41):
            exec(f"ax{j} = fig{count}.add_subplot(4,10,{j})")
            right_singular_vector = V_temp[j-1].reshape(10,10)
            exec(f"ax{j}.imshow(right_singular_vector,vmin=-0.3,vmax=0.3)")
            exec(f"ax{j}.xaxis.set_visible(False)")
            exec(f"ax{j}.yaxis.set_visible(False)")
        plt.suptitle("right singular vectors")

        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/right_singular_vectors_defect_{defect_1}+{defect_2}_norm_{norm}_noise_level_{noise_level}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/right_singular_vectors_defect_{defect_1}+{defect_2}_norm_{norm}.png")

        plt.figure()
        left_singular_vector = U_temp.T[0].reshape(10,10)
        plt.imshow(left_singular_vector,vmin=-0.3,vmax=0.3)
        plt.xticks([])
        plt.yticks([])
        plt.title(f"left singular vectors")

        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/left_singular_vector_defect_{defect_1}+{defect_2}_norm_{norm}_noise_level_{noise_level}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/left_singular_vector_defect_{defect_1}+{defect_2}_norm_{norm}.png")

    else:
        figure = plt.figure(figsize=(10,8))
        right_singular_vector = V_temp[0]
        left_singular_vector = U_temp.T[0]

        ax_01 = figure.add_subplot(2,2,1)
        im_01 = ax_01.imshow(right_singular_vector.reshape(10,10),vmin=-0.3,vmax=0.3)
        ax_01.xaxis.set_visible(False)
        ax_01.yaxis.set_visible(False)
        ax_01.set_title("right singular vector")
        figure.colorbar(im_01)

        ax_02 = figure.add_subplot(2,2,2)
        im_02 = ax_02.imshow(left_singular_vector.reshape(10,10),vmin=-0.4,vmax=0.4)
        ax_02.xaxis.set_visible(False)
        ax_02.yaxis.set_visible(False)
        ax_02.set_title("left singular vector")
        figure.colorbar(im_02)

        r_flux_u,r_flux_v,r_flux_magnitude = calculate_magnitude_from_perturbation(right_singular_vector)
        ax_03 = figure.add_subplot(2,2,3)
        cbar_03 = ax_03.quiver(sensor_x,sensor_y,r_flux_u,r_flux_v,r_flux_magnitude,cmap="Blues")
        ax_03.xaxis.set_visible(False)
        ax_03.yaxis.set_visible(False)
        ax_03.axis("equal")
        ax_03.set_title("magnetic flux (right singular vector)")
        figure.colorbar(cbar_03)

        l_flux_u,l_flux_v,l_flux_magnitude = calculate_magnitude_from_perturbation(left_singular_vector)
        ax_04 = figure.add_subplot(2,2,4)
        cbar_04 = ax_04.quiver(sensor_x,sensor_y,l_flux_u,l_flux_v,l_flux_magnitude,cmap="Blues")
        ax_04.xaxis.set_visible(False)
        ax_04.yaxis.set_visible(False)
        ax_04.axis("equal")
        ax_04.set_title("magnetic flux (left singular vector)")
        figure.colorbar(cbar_04)


        plt.suptitle(f"shape of right and left singular vector\n and corresponding magnetic flux (norm {norm})")

        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/singular_vector_defect_{defect_1}+{defect_2}_norm_{norm}_noise_level_{noise_level}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/singular_vector_defect_{defect_1}+{defect_2}_norm_{norm}.png")

    count += 1

plt.figure(fig1.number)
plt.title("inner product of perturbation direction (origin and perturbated)")
plt.legend()

if ADD_NOISE:
    plt.savefig(f"experiment_results/4.2/fig/inner_product_defect_{defect_1}+{defect_2}_norm_{norm}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/4.2/fig/inner_product_defect_{defect_1}+{defect_2}_norm_{norm}.png")

plt.close()



### 再構成誤差の比較

for i in range(len(plot_norms)):
    temp_norm = plot_norms[i]
    fig = plt.figure(figsize=(22,5))

    ax1 = fig.add_subplot(1,4,1)
    im1 = ax1.imshow(utils.reconstruct(model,y).reshape(10,10), vmin=0, vmax=1.2)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title("reconstruction of $x$")

    ax2 = fig.add_subplot(1,4,2)
    im2 = ax2.imshow(utils.reconstruct(model,y+A@(list_plot_multi_pert[i])).reshape(10,10), vmin=0, vmax=1.2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title("reconstruction of $x+r$")

    ax3 = fig.add_subplot(1,4,3)
    im3 = ax3.imshow(find_defect_position(utils.reconstruct(model,y+A@(list_plot_multi_pert[i]))).reshape(10,10), vmin=0, vmax=1.2, cmap=cm.plasma)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_title("reconstruction of $x+r$")

    ax4 = fig.add_subplot(1,4,4)
    im4 = ax4.imshow(list_plot_multi_reconstruction_approx[i].reshape(10,10), vmin=-1, vmax=1)
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.set_title("approx. of reconstruct error")

    plt.colorbar(im1)
    plt.colorbar(im2)
    plt.colorbar(im3)
    plt.colorbar(im4)

    plt.suptitle("reconstructions and reconstruction error")

    plt.savefig(f"experiment_results/4.2/fig/approximation_reconstruction_error_defect_{defect_1}+{defect_2}_norm_{temp_norm}.png")

    plt.close()

plt.figure()
final_pert = list_plot_multi_pert[-1] / np.linalg.norm(list_plot_multi_pert[-1])
plt.imshow(final_pert.reshape(10,10), vmin=-0.3, vmax=0.3)
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.title("perturbation shape")
plt.savefig(f"experiment_results/4.2/fig/final_pert_shape_defect_{defect_1}+{defect_2}.png")