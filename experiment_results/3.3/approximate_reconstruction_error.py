import torch
import matplotlib.pyplot as plt
import numpy as np

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

def find_defect_position(c, alpha = 2.0):
    current = c.reshape(10,10)
    score = calculate_defect_score(c)
    
    defect_position = np.zeros((10,10))

    for i in range(10):
        for j in range(10):
            if score[i][j] >= alpha:
                defect_position[i][j] = 1

    figure = plt.figure(figsize=(10,5))
    ax1 = figure.add_subplot(1,2,1)
    ax1.imshow(current,vmin=0,vmax=1.2)
    ax2 = figure.add_subplot(1,2,2)
    ax2.imshow(defect_position)
    plt.suptitle("defect_num:{}".format(count_defect_number(score,alpha)), y=0.9)
    plt.show()

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
    list_approx_error = [np.zeros(100)]

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
                    list_approx_error.append(list_approx_error[-1]+calculate_approximation_error(model,y+A@list_pert[-2],-minus_step_size*pert))
            else:
                list_pert.append(list_pert[-1]+plus_step_size*pert)
                list_step_size.append(plus_step_size)
                list_reconstruction_approx.append(list_reconstruction_approx[-1] + U.T[0]*S[0]*plus_step_size)
                list_approx_error.append(list_approx_error[-1]+calculate_approximation_error(model,y+A@list_pert[-2],plus_step_size*pert))

        else:
            if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]-minus_step_size*pert):
                if np.linalg.norm(list_pert[-1]) > np.linalg.norm(list_pert[-1]+plus_step_size*pert):
                    break
                else:
                    list_pert.append(list_pert[-1]+plus_step_size*pert)
                    list_step_size.append(plus_step_size)
                    list_reconstruction_approx.append(list_reconstruction_approx[-1] + U.T[0]*S[0]*plus_step_size)
                    list_approx_error.append(list_approx_error[-1]+calculate_approximation_error(model,y+A@list_pert[-2],plus_step_size*pert))

            else:
                list_pert.append(list_pert[-1]-minus_step_size*pert)
                list_step_size.append(minus_step_size)
                list_reconstruction_approx.append(list_reconstruction_approx[-1] - U.T[0]*S[0]*minus_step_size)
                list_approx_error.append(list_approx_error[-1]+calculate_approximation_error(model,y+A@list_pert[-2],-minus_step_size*pert))

    list_approx_error = [np.linalg.norm(p) for p in list_approx_error]

    return list_pert,list_singular_value,list_step_size,list_reconstruction_approx,list_approx_error



defect_1 = 33
defect_2 = 65
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

print(f"maximum singular value is {S[0]}")


### adversarial perturbation
M = 2000
lamda = 1e+1
gamma = 0.8
eta = 1e-5
tau = 1e-3

list_adversarial_pert = find_unstable_perturbation(y,x,model,A,M,lamda,gamma,eta,tau,p,stop_norm=0.09)
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
list_multistep_pert ,list_multistep_svalue, list_mustistep_stepsize, list_multistep_reconstruction_approx, list_multistep_residual_error = create_multistep_perturbation(model,y,size = 0.09)
list_multistep_reconstruction = [np.linalg.norm(utils.reconstruct(model,y+A@mpert) - p) for mpert in list_multistep_pert]
list_multistep_pert_norm = [np.linalg.norm(mpert) for mpert in list_multistep_pert]
multistep_upper_bounds = [np.linalg.norm(jac) + app for jac,app in zip(list_multistep_reconstruction_approx,list_multistep_residual_error)]
multistep_lower_bounds = [np.linalg.norm(jac) - app for jac,app in zip(list_multistep_reconstruction_approx,list_multistep_residual_error)]


### show results
if not os.path.exists("expriment_results/3.3/fig"):
    os.makedirs("expriment_results/3.3/fig")

plt.figure()
plt.plot(list_adversarial_norm, list_adversarial_error, label="adversarial (Antun et al.)")
plt.plot(list_jacobian_pert_norm, list_jacobian_error, label="proposed")
plt.xlabel("norm of perturbation")
plt.ylabel("norm of reconstruction error")
plt.legend()
plt.title("reconstruction error")
if ADD_NOISE:
    plt.savefig(f"experiment_results/3.3/fig/reconstruction_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/3.3/fig/reconstruction_error_defect_{defect_1}+{defect_2}.png")
plt.close()


plt.figure()
plt.plot(list_adversarial_norm, list_adversarial_error, label="adversarial (Antun et al.)")
plt.plot(list_jacobian_pert_norm, list_jacobian_error, label="proposed (single step)")
plt.plot(list_multistep_pert_norm, list_multistep_reconstruction, label="proposed (musti step)")
plt.xlabel("norm of perturbation")
plt.ylabel("norm of reconstruction error")
plt.legend()
plt.title("reconstruction error")
if ADD_NOISE:
    plt.savefig(f"experiment_results/3.3/fig/reconstruction_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_multistep.png")
else:
    plt.savefig(f"experiment_results/3.3/fig/reconstruction_error_defect_{defect_1}+{defect_2}_multistep.png")
plt.close()


plt.figure()
plt.plot(list_jacobian_pert_norm, list_singular_value_approximation, label="proposed approximation")
plt.plot(list_jacobian_pert_norm, list_jacobian_error, label="reconstruct error")
plt.plot(list_jacobian_pert_norm, upper_bounds, label="upper_bound")
plt.plot(list_jacobian_pert_norm, lower_bounds, label="lower_bound")
plt.xlabel("norm of perturbation")
plt.ylabel("norm of reconstruction error")
plt.legend()
plt.title("approximation error")
if ADD_NOISE:
    plt.savefig(f"experiment_results/3.3/fig/approximation_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}.png")
else:
    plt.savefig(f"experiment_results/3.3/fig/approximation_error_defect_{defect_1}+{defect_2}.png")
plt.close()

plt.figure()
plt.plot(list_multistep_pert_norm, [np.linalg.norm(pert) for pert in list_multistep_reconstruction_approx], label="proposed approximation (multi step)")
plt.plot(list_multistep_pert_norm, list_multistep_reconstruction, label="reconstruct error")
plt.plot(list_multistep_pert_norm, multistep_upper_bounds, label="upper_bound")
plt.plot(list_multistep_pert_norm, multistep_lower_bounds, label="lower_bound")
plt.xlabel("norm of perturbation")
plt.ylabel("norm of reconstruction error")
plt.legend()
plt.title("approximation error")
if ADD_NOISE:
    plt.savefig(f"experiment_results/3.3/fig/approximation_error_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_multistep.png")
else:
    plt.savefig(f"experiment_results/3.3/fig/approximation_error_defect_{defect_1}+{defect_2}_multistep.png")
plt.close()

plt.figure()
plt.hist(list_multistep_svalue)
plt.title("maximum singular values (each step)")
plt.xlabel("maximum singular value")
plt.ylabel("frequency")
if ADD_NOISE:
    plt.savefig(f"experiment_results/3.3/fig/svalue_hist_defect_{defect_1}+{defect_2}_noise_level_{noise_level}_multistep.png")
else:
    plt.savefig(f"experiment_results/3.3/fig/svalue_hist_defect_{defect_1}+{defect_2}_multistep.png")
plt.close()