import numpy as np
import window.loss as loss
from window.path import get_path

def analyse_direction_constante(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s,  n_point):

    #########
    #calucule la trajectoire

    L =  np.linspace(L_min, L_max, n_L)
    T = np.linspace(T_min, T_max, n_T)

    Loss = np.zeros((n_T, n_L))
    L_loss_max_v = np.zeros((n_T, n_L))
    L_max_a = np.zeros((n_T, n_L))
    L_id_v_dist = np.zeros((n_T, n_L))
    L_loss_lenght = np.zeros((n_T, n_L))

    v1 = v_win/np.linalg.norm(v_win)*norme_v1


    for n_t, val_T in enumerate(T):
        for n_l, val_L in enumerate(L):
            x1 = x0+val_L*direction_win/np.linalg.norm(direction_win)
            t, val_u = get_path(x0, x1, v0, v1, a0, a1, val_L, val_T, n_point)

            Loss[n_t, n_l] = loss.loss(t, val_u, val_T, conv_L_m, conv_T_s)
            L_loss_max_v[n_t, n_l] = loss.loss_max_v(t, val_u, val_T, conv_L_m, conv_T_s)
            L_id_v_dist[n_t, n_l] = loss.loss_id_v_dist(t, val_u, val_T, conv_L_m, conv_T_s)
            L_max_a[n_t, n_l] = loss.loss_max_a(t, val_u, val_T, conv_L_m, conv_T_s)
            L_loss_lenght[n_t, n_l] = loss.loss_lenght(t, val_u, val_T, conv_L_m, conv_T_s)


    ##########
    #transforme les matrice pour match le format d'Imshow

    Loss = Loss[::-1]
    L_loss_max_v = L_loss_max_v[::-1]
    L_max_a = L_max_a[::-1]
    L_id_v_dist = L_id_v_dist[::-1]
    L_loss_lenght = L_loss_lenght[::-1]

    return Loss, L_loss_max_v, L_max_a, L_id_v_dist, L_loss_lenght