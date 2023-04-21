import numpy as np

def loss(t, val_u, val_T, conv_L_m, conv_T_s):
    max_va = loss_max_v_a(t, val_u, val_T, conv_L_m, conv_T_s)
    l_dist_v_id = loss_id_v_dist(t, val_u, val_T, conv_L_m, conv_T_s)
    l_lenght = loss_lenght(t, val_u, val_T, conv_L_m, conv_T_s)

    return 1*max_va + 1*l_dist_v_id + 1*l_lenght

def loss_max_v(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = np.gradient(np.transpose(val_u*conv_L_m), val_T*conv_T_s/len(t))[0]
    max_norme_val_v = np.max(np.linalg.norm(val_v, axis=1))
    return (conv_T_s/conv_L_m*max_norme_val_v)**2

def loss_max_a(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = np.gradient(np.transpose(val_u*conv_L_m), val_T*conv_T_s/len(t))[0]
    val_a = np.gradient(val_v, val_T*conv_T_s/len(t))[0]
    max_norme_val_a = np.max(np.linalg.norm(val_a, axis=1))
    return (conv_T_s**2/conv_L_m*max_norme_val_a)**2

def loss_max_v_a(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = np.gradient(np.transpose(val_u*conv_L_m), val_T*conv_T_s/len(t))[0]
    val_a = np.gradient(val_v, val_T*conv_T_s/len(t))[0]
    max_norme_val_v = np.max(np.linalg.norm(val_v, axis=1))
    max_norme_val_a = np.max(np.linalg.norm(val_a, axis=1))

    return (conv_T_s/conv_L_m*max_norme_val_v)**2 + (conv_T_s**2/conv_L_m*max_norme_val_a)**2

def loss_id_v_dist(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = np.transpose(np.gradient(np.transpose(val_u*conv_L_m), val_T*conv_T_s/len(t))[0])
    identity = np.linspace(np.linalg.norm(val_v[:,0]), np.linalg.norm(val_v[:,-1]), len(t))

    dist_id = np.linalg.norm(conv_T_s*(np.linalg.norm(val_v, axis=0) - identity)/conv_L_m)

    return dist_id**2

def loss_lenght(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = np.gradient(np.transpose(val_u*conv_L_m), val_T*conv_T_s/len(t))[0]
    norm_val_v = np.linalg.norm(val_v, axis=1)
    lenght = np.trapz(norm_val_v, t)
    return (lenght/conv_L_m)**2