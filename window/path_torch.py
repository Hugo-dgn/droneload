import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 

def find_dvice_cuda():
    return device

t = None
M = None
def define_t(n_point):
    global t
    t = torch.linspace(0, 1, n_point).float().to(device)

def define_M(t):
    global M
    M = torch.stack([t**(6-n) for n in range(1, 7)]).to(device) 

A = torch.tensor([[0, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 0],
                  [5, 4, 3, 2, 1, 0],
                  [0, 0, 0, 2, 0, 0],
                  [20, 12, 6, 2, 0, 0]], dtype=torch.float).to(device) 

A_inv = torch.linalg.inv(A)

def get_path_torch(x0, x1, v0, v1, a0, a1, L, T, n_point):

    x0_ad = x0/L
    x1_ad = x1/L

    v0_ad = T*v0/L
    v1_ad = T*v1/L

    a0_ad = T**2*a0/L
    a1_ad = T**2*a1/L

    X = torch.stack([x0_ad, x1_ad, v0_ad, v1_ad, a0_ad, a1_ad])

    Y = torch.matmul(A_inv, X)

    if t is None or t.shape[0] != n_point:
        define_t(n_point)
        define_M(t)

    U = L*torch.matmul(torch.transpose(Y, 0, 1), M)

    return T*t, U
