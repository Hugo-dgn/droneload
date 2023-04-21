import numpy as np
import matplotlib.pyplot as plt

def loss(t, val_u, L, T):
    max_va = loss_max_v_a(t, val_u, L, T)
    l_dist_v_id = loss_id_v_dist(t, val_u, L, T)
    l_lenght = loss_lenght(t, val_u, L, T)

    return max_va + l_dist_v_id + l_lenght

def loss_max_v(t, val_u, L, T):
    val_v = np.gradient(np.transpose(val_u), T/len(t))[0]
    max_norme_val_v = np.max(np.linalg.norm(val_v, axis=1))
    return (T/L*max_norme_val_v)**2

def loss_max_a(t, val_u, L, T):
    val_v = np.gradient(np.transpose(val_u), T/len(t))[0]
    val_a = np.gradient(val_v, T/len(t))[0]
    max_norme_val_a = np.max(np.linalg.norm(val_a, axis=1))
    return (T**2/L*max_norme_val_a)**2

def loss_max_v_a(t, val_u, L, T):
    val_v = np.gradient(np.transpose(val_u), T/len(t))[0]
    val_a = np.gradient(val_v, T/len(t))[0]
    max_norme_val_v = np.max(np.linalg.norm(val_v, axis=1))
    max_norme_val_a = np.max(np.linalg.norm(val_a, axis=1))

    return (T/L*max_norme_val_v)**2 + (T**2/L*max_norme_val_a)**2

def loss_id_v_dist(t, val_u, L, T):
    val_v = np.transpose(np.gradient(np.transpose(val_u), T/len(t))[0])
    identity = np.linspace(np.linalg.norm(val_v[:,0]), np.linalg.norm(val_v[:,-1]), len(t))

    dist_id = np.linalg.norm(T*(np.linalg.norm(val_v, axis=0) - identity)/L)

    return dist_id**2

def loss_lenght(t, val_u, L, T):
    val_v = np.linalg.norm(np.gradient(np.transpose(val_u), T/len(t))[0], axis=1)
    lenght = np.trapz(val_v, t)
    return (lenght/L)**2