import torch
import numpy as np
import droneload.window.loss as loss
from droneload.window.path_torch import get_path_torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 

def get_L_T_grid(X, Y):
    M = np.stack((X, Y), axis=2)
    return M

def analyse_direction_constante_torch(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s, n_point):
    x0 = torch.tensor(x0).to(device) 
    direction_win = torch.tensor(direction_win).to(device) 
    v_win = torch.tensor(v_win).to(device) 
    v0 = torch.tensor(v0).to(device) 
    a0 = torch.tensor(a0).to(device) 
    a1 = torch.tensor(a1).to(device)
    #########
    # calcul la trajectoire
    L = np.linspace(L_min, L_max, n_L)
    T = np.linspace(T_min, T_max, n_T)

    X, Y = np.meshgrid(L, T)
    L_T_grid = get_L_T_grid(X, Y)

    Loss = torch.zeros((n_T, n_L))
    v1 = v_win.to(device)/torch.linalg.norm(v_win.float())*norme_v1

    norme_direction_win = torch.linalg.norm(direction_win.float())

    def calcul_loss(val_L_T):
        val_L, val_T = val_L_T
        x1 = x0 + val_L*direction_win/norme_direction_win
        t, val_u = get_path_torch(x0, x1, v0, v1, a0, a1, val_L, val_T, n_point)
        return loss.loss(t, val_u, val_T, conv_L_m, conv_T_s)

    for i in range(n_T):
        for j in range(n_L):
            Loss[i, j] = calcul_loss(L_T_grid[i, j])

    ##########
    # transforme les matrices pour matcher le format d'Imshow

    Loss = Loss.flip(dims=(0,))

    return Loss