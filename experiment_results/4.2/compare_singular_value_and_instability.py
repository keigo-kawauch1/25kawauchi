import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from learner.net import InverseNet
from learner import utils
from learner.utils import FuelCellConfig
import matplotlib as mpl
import matplotlib.cm as cm
import yaml

model = InverseNet().to("cpu")
time = 'weight/2025-01-04:15:56/'
checkpoint = torch.load(time + 'weight.ckpt')
model.load_state_dict(checkpoint['state_dict'])

with open(time + 'config.yaml') as file:
    config = yaml.safe_load(file.read())

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



defect_1 = 32
defect_2 = 79
ADD_NOISE = False
MULTISTEP = True
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

### adversarial perturbation
M = 2000
lamda = 1e+1
gamma = 0.8
eta = 1e-5
tau = 1e-3
stop_norm = 0.071

plot_norms = [0.03,0.05,0.06,0.07]



list_adversarial_pert = find_unstable_perturbation(y,x,model,A,M,lamda,gamma,eta,tau,p,stop_norm=stop_norm)
list_adversarial_error = [np.linalg.norm(utils.reconstruct(model,y+A@pert) - p) for pert in list_adversarial_pert]
list_adversarial_norm = [np.linalg.norm(pert) for pert in list_adversarial_pert]


### proposed perturbation
jacobian_pert = V[0]
list_jacobian_pert_norm = [0.001* i for i in range(int(list_adversarial_norm[-1]/0.001))]
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

### normal perturbation
normal_pert = np.random.randn(100)
normal_pert *= 1 / np.linalg.norm(normal_pert)
list_normal_error = [np.linalg.norm(utils.reconstruct(model,y+A@normal_pert*i) - p) for i in list_jacobian_pert_norm]

### ノルムに応じた擾乱の形成
adv_indices = np.searchsorted(list_adversarial_norm, plot_norms, side='right')
list_plot_adv_pert = [list_adversarial_pert[ind] for ind in adv_indices]

list_plot_jac_pert = [jacobian_pert*pnorm for pnorm in plot_norms]
list_plot_norm_pert = [normal_pert*pnorm for pnorm in plot_norms]

multi_indices = np.searchsorted(list_multistep_pert_norm, plot_norms, side='right')
list_plot_multi_pert =  [list_multistep_pert[ind] for ind in multi_indices]


### 擾乱同士の形状の比較
def plot_perturbation_with_norm(adv_pert,jac_pert,norm_pert,norm):
    fig = plt.figure(figsize=(9,4))
    gs = gridspec.GridSpec(1,4,width_ratios=[1,1,1,0.05])

    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow((adv_pert/np.linalg.norm(adv_pert)).reshape(10,10),vmin=-0.3,vmax=0.3)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title("adversarial perturbation\n (Antun et al.)")

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow((jac_pert/np.linalg.norm(jac_pert)).reshape(10,10),vmin=-0.3,vmax=0.3)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title("proposed perturbation")

    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow((norm_pert/np.linalg.norm(norm_pert)).reshape(10,10),vmin=-0.3,vmax=0.3)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_title("random perturbation")
    
    cbar_ax = fig.add_subplot(gs[3])
    fig.colorbar(im3,cax=cbar_ax)

    fig.suptitle("shape of perturbations")

    if MULTISTEP:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/multistep_perturbation_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_norm_{norm}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/multistep_perturbation_defect_{defect_1}+{defect_2}_norm_{norm}.png")
    else:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/perturbation_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_norm_{norm}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/perturbation_defect_{defect_1}+{defect_2}_norm_{norm}.png")
    plt.close()

if MULTISTEP:
    for i in range(len(plot_norms)):
        plot_perturbation_with_norm(list_plot_adv_pert[i],list_plot_multi_pert[i],list_plot_norm_pert[i],plot_norms[i])
else:
    for i in range(len(plot_norms)):
        plot_perturbation_with_norm(list_plot_adv_pert[i],list_plot_jac_pert[i],list_plot_norm_pert[i],plot_norms[i])


### 再構成結果

def plot_reconstruction_results_with_pert(adv_pert,jac_pert,norm_pert,norm):
    fig = plt.figure(figsize=(12,7))
    gs = gridspec.GridSpec(2,4,width_ratios=[1,1,1,0.05])

    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(utils.reconstruct(model, y+A@adv_pert).reshape(10,10),vmin=0,vmax=1.2)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title("reconstruction\n (adversarial perturbation (Antun et al.))")

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(utils.reconstruct(model, y+A@jac_pert).reshape(10,10),vmin=0,vmax=1.2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title("reconstruction\n (proposed perturbation)")

    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(utils.reconstruct(model, y+A@norm_pert).reshape(10,10),vmin=0,vmax=1.2)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.set_title("reconstruction\n (random perturbation)")

    ax4 = fig.add_subplot(gs[4])
    im4 = ax4.imshow(find_defect_position(utils.reconstruct(model,y+A@adv_pert)),vmin=0,vmax=1.2)
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.set_title("defect position\n (adversarial perturbation (Antun et al.))")

    ax5 = fig.add_subplot(gs[5])
    im5 = ax5.imshow(find_defect_position(utils.reconstruct(model, y+A@jac_pert)),vmin=0,vmax=1.2)
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.set_title("defect position\n (proposed perturbation)")

    ax6 = fig.add_subplot(gs[6])
    im6 = ax6.imshow(find_defect_position(utils.reconstruct(model, y+A@norm_pert)),vmin=0,vmax=1.2)
    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_visible(False)
    ax6.set_title("defect position\n (random perturbation)")

    cbar_ax = fig.add_subplot(gs[3])
    fig.colorbar(im3,cax=cbar_ax)

    fig.suptitle("shape of perturbated reconstruction images")

    if MULTISTEP:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/multistep_perturbated_reconstruction_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_norm_{norm}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/multistep_perturbated_reconstruction_defect_{defect_1}+{defect_2}_norm_{norm}.png")
    else:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/perturbated_reconstruction_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_norm_{norm}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/perturbated_reconstruction_defect_{defect_1}+{defect_2}_norm_{norm}.png")
    plt.close()



if MULTISTEP:
    for i in range(len(plot_norms)):
        plot_reconstruction_results_with_pert(list_plot_adv_pert[i],list_plot_multi_pert[i],list_plot_norm_pert[i],plot_norms[i])
else:
    for i in range(len(plot_norms)):
        plot_reconstruction_results_with_pert(list_plot_adv_pert[i],list_plot_jac_pert[i],list_plot_norm_pert[i],plot_norms[i])

### 擾乱なしの再構成結果

fig = plt.figure(figsize=(7,4))

gs=gridspec.GridSpec(1,3,width_ratios=[1,1,0.05])

ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(utils.reconstruct(model, y).reshape(10,10),vmin=0,vmax=1.2)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_title("reconstruction")

ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(find_defect_position(utils.reconstruct(model, y)),vmin=0,vmax=1.2)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_title("defect position")

cbar_ax = fig.add_subplot(gs[2])
fig.colorbar(im1,cax=cbar_ax)

fig.suptitle("shape of reconstruction image")

if ADD_NOISE:
    plt.savefig(f"experiment_results/4.2/fig/reconstruction_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/4.2/fig/reconstruction_defect_{defect_1}+{defect_2}.png")
plt.close()


### 再構成結果の再構成誤差
def plot_reconstruction_error(adv_list,adv_norm_list,jac_list,jac_norm_list,norm_list,norm_norm_list):
    label = "proposed (musti step)" if MULTISTEP else "proposed (single step)"
    plt.figure()
    plt.plot(adv_norm_list, adv_list, label="adversarial (Antun et al.)")
    plt.plot(jac_norm_list, jac_list, label=label)
    plt.plot(norm_norm_list, norm_list, label="gaussian perturbation")
    plt.legend()
    plt.xlabel("norm of perturbation")
    plt.ylabel("norm of reconstruction error")
    plt.title("reconstruction error")
    if MULTISTEP:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/multistep_reconstruction_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/multistep_reconstruction_error_defect_{defect_1}+{defect_2}.png")
    else:
        if ADD_NOISE:
            plt.savefig(f"experiment_results/4.2/fig/reconstruction_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
        else:
            plt.savefig(f"experiment_results/4.2/fig/reconstruction_error_defect_{defect_1}+{defect_2}.png")      
    plt.close()

if MULTISTEP:
    plot_reconstruction_error(list_adversarial_error,list_adversarial_norm,list_multistep_reconstruction_error,list_multistep_pert_norm,list_normal_error,list_jacobian_pert_norm)
else:
    plot_reconstruction_error(list_adversarial_error,list_adversarial_norm,list_jacobian_error,list_jacobian_pert_norm,list_normal_error,list_jacobian_pert_norm)

### 最大特異値のヒストグラム
plt.figure()
plt.hist(list_multistep_svalue)
plt.title("maximum singular values (each step)")
plt.xlabel("maximum singular value")
plt.ylabel("frequency")
if ADD_NOISE:
    plt.savefig(f"experiment_results/4.2/fig/svalue_multistep_hist_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/4.2/fig/svalue_multistep_hist_defect_{defect_1}+{defect_2}.png")
plt.close()

print(pd.Series(list_multistep_svalue).describe())


list_surrounding_svalue = []
for i in range(100):
    J = calculate_jacobian(model, y+0.01*A@(-np.ones(100)+2*np.random.rand(100)))
    _,S,_ = np.linalg.svd(J@A)
    list_surrounding_svalue.append(S[0])

plt.figure()
plt.hist(list_surrounding_svalue)
plt.title("maximum singular values (around source)")
plt.xlabel("maximum singular value")
plt.ylabel("frequency")
if ADD_NOISE:
    plt.savefig(f"experiment_results/4.2/fig/svalue_around_source_hist_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/4.2/fig/svalue_around_source_hist_defect_{defect_1}+{defect_2}.png")
plt.close()

print(pd.Series(list_surrounding_svalue).describe())

### 磁場の形状

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


plt.figure()
flux_u,flux_v,flux_magnitude = calculate_magnitude_from_perturbation(x)
quiver = plt.quiver(sensor_x,sensor_y,flux_u,flux_v,flux_magnitude,cmap="Blues")
cbar = plt.colorbar(quiver)
plt.title("true magnetic flux")
if ADD_NOISE:
    plt.savefig(f"experiment_results/4.2/fig/magnetic_flux_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/4.2/fig/magnetic_flux_defect_{defect_1}+{defect_2}.png")

plt.close()

def plot_magnetic_flux_caused_by_perturbation(adv_pert,jac_pert,norm_pert,norm):
    fig = plt.figure(figsize=(22,6))

    ax1 = fig.add_subplot(1,3,1)
    flux_u,flux_v,flux_magnitude = calculate_magnitude_from_perturbation(adv_pert)
    quiver1 = ax1.quiver(sensor_x,sensor_y,flux_u,flux_v,flux_magnitude,cmap="Blues")
    cbar1 = fig.colorbar(quiver1)
    ax1.set_title("magnetic flux\n (adversarial perturbation (Antun et al.))")
    ax2 = fig.add_subplot(1,3,2)
    flux_u,flux_v,flux_magnitude = calculate_magnitude_from_perturbation(jac_pert)
    quiver2 = ax2.quiver(sensor_x,sensor_y,flux_u,flux_v,flux_magnitude,cmap="Blues")
    cbar2 = fig.colorbar(quiver2)
    ax2.set_title("magnetic flux\n (proposed perturbation)")
    ax3 = fig.add_subplot(1,3,3)
    flux_u,flux_v,flux_magnitude = calculate_magnitude_from_perturbation(norm_pert)
    quiver3 = ax3.quiver(sensor_x,sensor_y,flux_u,flux_v,flux_magnitude,cmap="Blues")
    cbar3 = fig.colorbar(quiver3)
    ax3.set_title("magnetic flux\n (random perturbation)")

    fig.suptitle("magnetic flux created by perturbation")

    if ADD_NOISE:
        plt.savefig(f"experiment_results/4.2/fig/perturbation_magnetic_flux_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_norm_{norm}.png")
    else:
        plt.savefig(f"experiment_results/4.2/fig/perturbation_magnetic_flux_defect_{defect_1}+{defect_2}_norm_{norm}.png")

    plt.close()

if MULTISTEP:
    for i in range(len(plot_norms)):
        plot_magnetic_flux_caused_by_perturbation(list_plot_adv_pert[i],list_plot_multi_pert[i],list_plot_norm_pert[i],plot_norms[i])
else:
    for i in range(len(plot_norms)):
        plot_magnetic_flux_caused_by_perturbation(list_plot_adv_pert[i],list_plot_jac_pert[i],list_plot_norm_pert[i],plot_norms[i])
