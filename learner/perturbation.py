import torch
import numpy as np

def calculate_gradient(f, p, y, A, r, lamda):
    u = y + torch.matmul(A, r)
    u.requires_grad = True
    g_u = torch.norm(f(u) - p, p=2)**2 / 2
    nabla_u_g = torch.autograd.grad(g_u, u, retain_graph=True)[0]
    nabla_r_Q = torch.matmul(A.T, nabla_u_g) - lamda * r
    return nabla_r_Q

def find_unstable_perturbation(x, f, A, M, lamda, gamma, eta, tau, p):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    f = f.to(device)
    
    x = torch.tensor(x, device=device, dtype=torch.float32)
    A = torch.tensor(A, device=device, dtype=torch.float32)
    p = torch.tensor(p, device=device, dtype=torch.float32)

    y = torch.matmul(A, x)
    v = torch.zeros_like(x, device=device)
    r = tau * torch.rand_like(x, device=device)

    for _ in range(M):
        nabla_r_Q = calculate_gradient(f, p, y, A, r, lamda)
        v = gamma * v + eta * nabla_r_Q
        r = r + v

    return r.cpu().detach().numpy()

# サンプルの使用例
if __name__ == "__main__":
    # ニューラルネットワークの定義（例）
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.layer1 = torch.nn.Linear(10, 10)
            self.layer2 = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.layer2(x)
            return x

    # モデルのインスタンス化とGPUへの移動
    model = SimpleNN().to('cuda')
    
    # 入力データとパラメータの定義
    x = np.random.rand(10)  # 例としてランダムな入力画像
    A = np.random.rand(10, 10)  # サンプリング行列
    M = 100  # 最大反復回数
    lamda = 0.1
    gamma = 0.9
    eta = 0.01
    tau = 0.1
    p = np.random.rand(10)  # 目標ベクトル（例としてランダム）

    # 不安定な摂動の計算
    r = find_unstable_perturbation(x, model, A, M, lamda, gamma, eta, tau, p)
    print("Calculated Perturbation:", r)
