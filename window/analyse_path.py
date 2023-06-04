import numpy as np
import window.loss as loss
from window.path import get_path

def get_L_T_grid(X, Y):
    M = np.stack((X, Y), axis=2)
    return M

def analyse_direction_constante(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s,  n_point, all_loss=True):

    #########
    #calucule la trajectoire
    L =  np.linspace(L_min, L_max, n_L)
    T = np.linspace(T_min, T_max, n_T)

    X, Y = np.meshgrid(L, T)
    L_T_grid = get_L_T_grid(X, Y)

    Loss = np.zeros((n_T, n_L))
    if all_loss:
        L_loss_max_v = np.zeros((n_T, n_L))
        L_max_a = np.zeros((n_T, n_L))
        L_id_v_dist = np.zeros((n_T, n_L))
        L_loss_lenght = np.zeros((n_T, n_L))
    else:
        L_loss_max_v = None
        L_max_a = None
        L_id_v_dist = None
        L_loss_lenght = None

    v1 = v_win/np.linalg.norm(v_win)*norme_v1
    if not all_loss:
        def calcul_loss(val_L_T):
            val_L, val_T = val_L_T
            x1 = x0+val_L*direction_win/np.linalg.norm(direction_win)
            t, val_u = get_path(x0, x1, v0, v1, a0, a1, val_L, val_T, n_point)
            return loss.loss(t, val_u, val_T, conv_L_m, conv_T_s)
        Loss = np.apply_along_axis(calcul_loss, 2, L_T_grid)
    else:
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
    if all_loss:
        L_loss_max_v = L_loss_max_v[::-1]
        L_max_a = L_max_a[::-1]
        L_id_v_dist = L_id_v_dist[::-1]
        L_loss_lenght = L_loss_lenght[::-1]

    return Loss, L_loss_max_v, L_max_a, L_id_v_dist, L_loss_lenght

def get_unite_sphere_vector(phi, theta):
    A = np.cos(phi)*np.cos(theta)
    B = np.sin(phi)*np.cos(theta)
    C = np.sin(theta)
    M = np.stack((A, B, C), axis=2)
    return M

def analyse(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, v_win, norme_v1, conv_L_m, conv_T_s,  n_point, n_angle_phi, n_angle_theta):
    list_phi = np.linspace(-np.pi, np.pi, n_angle_phi)
    list_theta = np.linspace(-np.pi/2, np.pi/2, n_angle_theta)

    X, Y = np.meshgrid(list_phi, list_theta)

    A = get_unite_sphere_vector(X, Y)

    def calcul_loss(direction_win):
        return np.amin(analyse_direction_constante(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s,  n_point, all_loss=False)[0])

    A = np.apply_along_axis(calcul_loss, 2, A)[::-1]
    return A

def analyse_all_cube(lenght_side, T, v0, a0, v1, a1, n_point, n_point_path, conv_L_m, conv_T_s):
    X = np.linspace(0, 2*lenght_side, n_point)
    Y = np.linspace(0, 2*lenght_side, n_point)
    Z = np.linspace(0, 2*lenght_side, n_point)
    M_loss = np.zeros((n_point, n_point, n_point))

    x1 = np.array([lenght_side, lenght_side, lenght_side])

    for i_x, x in enumerate(X):
        for i_y, y in enumerate(Y):
            for i_z, z in enumerate(Z):
                x0 = np.array([x, y, z])
                L = np.linalg.norm(x1-x0)
                if L > 0:
                    t, val_u = get_path(x0, x1, v0, v1, a0, a1, L, T, n_point_path)
                    M_loss[i_x, i_y, i_z] = loss.loss(t, val_u, T, conv_L_m, conv_T_s)
    return M_loss