import torch

def loss(t, val_u, val_T, conv_L_m, conv_T_s):
    max_va = loss_max_v_a(t, val_u, val_T, conv_L_m, conv_T_s)
    l_dist_v_id = loss_id_v_dist(t, val_u, val_T, conv_L_m, conv_T_s)
    l_length = loss_length(t, val_u, val_T, conv_L_m, conv_T_s)

    return 1*max_va + 1*l_dist_v_id + 1*l_length

def loss_max_v(t, val_u, val_T, conv_L_m, conv_T_s):
    val_u_t = torch.tensor(val_u*conv_L_m).transpose(0, 1).unsqueeze(-1)
    val_T_t = torch.tensor(val_T*conv_T_s/len(t), dtype=torch.float32)
    val_v = torch.autograd.grad(val_u_t, t, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True)[0]
    val_v_t = val_v.transpose(0, 1).unsqueeze(-1)
    max_norme_val_v = torch.norm(val_v_t, dim=1, p=2).max()

    return (conv_T_s/conv_L_m*max_norme_val_v)**2

def loss_max_a(t, val_u, val_T, conv_L_m, conv_T_s):
    val_u_t = torch.tensor(val_u*conv_L_m).transpose(0, 1).unsqueeze(-1)
    val_T_t = torch.tensor(val_T*conv_T_s/len(t), dtype=torch.float32)
    val_v = torch.autograd.grad(val_u_t, t, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True)[0]
    val_v_t = val_v.transpose(0, 1).unsqueeze(-1)
    val_a = torch.autograd.grad(val_v_t, t, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True)[0]
    val_a_t = val_a.transpose(0, 1).unsqueeze(-1)
    max_norme_val_a = torch.norm(val_a_t, dim=1, p=2).max()

    return (conv_T_s**2/conv_L_m*max_norme_val_a)**2

def loss_max_v_a(t, val_u, val_T, conv_L_m, conv_T_s):
    val_u_t = torch.tensor(val_u*conv_L_m).transpose(0, 1).unsqueeze(-1)
    val_T_t = torch.tensor(val_T*conv_T_s/len(t), dtype=torch.float32)
    val_v = torch.autograd.grad(val_u_t, t, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True)[0]
    val_v_t = val_v.transpose(0, 1).unsqueeze(-1)
    val_a = torch.autograd.grad(val_v_t, t, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True)[0]
    val_a_t = val_a.transpose(0, 1).unsqueeze(-1)
    max_norme_val_v = torch.norm(val_v_t, dim=1, p=2).max()
    max_norme_val_a = torch.norm(val_a_t, dim=1, p=2).max()

    return (conv_T_s/conv_L_m*max_norme_val_v)**2 + (conv_T_s**2/conv_L_m*max_norme_val_a)**2

def loss_id_v_dist(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = torch.transpose(torch.gradient(torch.transpose(val_u*conv_L_m,0,1), val_T*conv_T_s/len(t))[0],0,1)
    identity = torch.linspace(torch.norm(val_v[:,0]), torch.norm(val_v[:,-1]), len(t))

    dist_id = torch.norm(conv_T_s*(torch.norm(val_v, dim=0) - identity)/conv_L_m)

    return dist_id**2

def loss_lenght(t, val_u, val_T, conv_L_m, conv_T_s):
    val_v = torch.gradient(torch.transpose(val_u*conv_L_m,0,1), val_T*conv_T_s/len(t))[0]
    norm_val_v = torch.norm(val_v, dim=1)
    lenght = torch.trapz(norm_val_v, t)
    return (lenght/conv_L_m)**2
