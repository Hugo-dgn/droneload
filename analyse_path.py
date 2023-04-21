import window
import matplotlib.pyplot as plt
import numpy as np
import loss
from matplotlib.colors import LogNorm
from locals import *

#########
#Paramètres de dimensionement

L_min = 1
L_max = 10
unite_L = "m"

T_min = 1
T_max = 10
unite_T = "s"

n_T = 100
n_L = 100

L =  np.linspace(L_min, L_max, n_L)
T = np.linspace(T_min, T_max, n_T)

#########
#calucule la trajectoire

Loss = np.zeros((n_T, n_L))
L_loss_max_v = np.zeros((n_T, n_L))
L_max_a = np.zeros((n_T, n_L))
L_id_v_dist = np.zeros((n_T, n_L))
L_loss_lenght = np.zeros((n_T, n_L))

v1 = v_win/np.linalg.norm(v_win)*norme_v1


for n_t, val_T in enumerate(T):
    for n_l, val_L in enumerate(L):
        x1 = x0+val_L*direction_win/np.linalg.norm(direction_win)
        t, val_u = window.get_path(x0, x1, v0, v1, a0, a1, val_L, val_T, n_point)

        Loss[n_t, n_l] = loss.loss(t, val_u, val_L, val_T)
        L_loss_max_v[n_t, n_l] = loss.loss_max_v(t, val_u, val_L, val_T)
        L_id_v_dist[n_t, n_l] = loss.loss_id_v_dist(t, val_u, val_L, val_T)
        L_max_a[n_t, n_l] = loss.loss_max_a(t, val_u, val_L, val_T)
        L_loss_lenght[n_t, n_l] = loss.loss_lenght(t, val_u, val_L, val_T)


##########
#transforme les matrice pour match le format d'Imshow

Loss = Loss[::-1]
L_loss_max_v = L_loss_max_v[::-1]
L_max_a = L_max_a[::-1]
L_id_v_dist = L_id_v_dist[::-1]
L_loss_lenght = L_loss_lenght[::-1]

##########
#plot des loss

plt.figure()

plt.imshow(Loss, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

plt.colorbar()
plt.xlabel(f"L ({unite_L})")
plt.ylabel(f"T ({unite_T})")
plt.title('loss', fontsize=8)

plt.figure()

plt.subplot(2, 2, 1)

plt.imshow(L_loss_lenght, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

plt.colorbar()
plt.xlabel(f"L ({unite_L})")
plt.ylabel(f"T ({unite_T})")
plt.title('loss_lenght', fontsize=8)

plt.subplot(2, 2, 2)

plt.imshow(L_loss_max_v, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

plt.colorbar()
plt.xlabel(f"L ({unite_L})")
plt.ylabel(f"T ({unite_T})")
plt.title('max_v', fontsize=8)

plt.subplot(2, 2, 3)

plt.imshow(L_id_v_dist, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

plt.colorbar()
plt.xlabel(f"L ({unite_L})")
plt.ylabel(f"T ({unite_T})")
plt.title('id_dist', fontsize=8)

plt.subplot(2, 2, 4)

plt.imshow(L_max_a, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

plt.colorbar()
plt.xlabel(f"L ({unite_L})")
plt.ylabel(f"T ({unite_T})")
plt.title('max_a', fontsize=8)

plt.show()