import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from matplotlib.animation import FuncAnimation
from window.path import get_path
from window.path_torch import get_path_torch
from window.window import Window



def plot_analyse_direction_constante(Loss, L_loss_lenght, L_loss_max_v, L_id_v_dist, L_max_a, L_min, L_max, T_min, T_max, n_L, n_T):
    plt.figure()

    plt.imshow(Loss, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

    i = np.unravel_index(np.argmin(Loss), Loss.shape)

    print(f"minimal loss = {Loss[i]}")
    print(f"T = {T_min + (n_T-i[0]-1)*(T_max-T_min)/n_T}")
    print(f"L = {L_min + i[1]*(L_max-L_min)/n_L}")

    plt.colorbar()
    plt.xlabel(f"L (m)")
    plt.ylabel(f"T (s)")
    plt.title('loss', fontsize=8)

    if L_loss_lenght is not None:
        plt.figure()

        plt.subplot(2, 2, 1)

        plt.imshow(L_loss_lenght, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

        plt.colorbar()
        plt.xlabel(f"L (m)")
        plt.ylabel(f"T (s)")
        plt.title('loss_lenght', fontsize=8)

        plt.subplot(2, 2, 2)

        plt.imshow(L_loss_max_v, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

        plt.colorbar()
        plt.xlabel(f"L (m)")
        plt.ylabel(f"T (s)")
        plt.title('max_v', fontsize=8)

        plt.subplot(2, 2, 3)

        plt.imshow(L_id_v_dist, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

        plt.colorbar()
        plt.xlabel(f"L (m)")
        plt.ylabel(f"T (s)")
        plt.title('id_dist', fontsize=8)

        plt.subplot(2, 2, 4)

        plt.imshow(L_max_a, norm=LogNorm(), extent=[L_min, L_max, T_min, T_max])

        plt.colorbar()
        plt.xlabel(f"L (m)")
        plt.ylabel(f"T (t)")
        plt.title('max_a', fontsize=8)

    plt.show()

def plot_analyse_total(Loss_angle, n_angle_phi, n_angle_theta):
    plt.figure()

    plt.imshow(Loss_angle, norm=LogNorm(), extent=[-np.pi, np.pi, -np.pi/2, np.pi/2])

    i = np.unravel_index(np.argmin(Loss_angle), Loss_angle.shape)

    print(f"minimal loss = {Loss_angle[i]}")
    phi_min = -np.pi + (n_angle_phi-i[0]-1)*(2*np.pi)/n_angle_phi
    theta_min = -np.pi/2 + i[1]*(np.pi)/n_angle_theta
    direction_win = np.array([np.cos(phi_min)*np.cos(theta_min), np.sin(phi_min)*np.cos(theta_min), np.sin(theta_min)])
    print(f"phi = {phi_min}")
    print(f"theta = {theta_min}")
    print(f"direction = {direction_win}")

    plt.colorbar()
    plt.xlabel(f"phi (rad)")
    plt.ylabel(f"theta (rad)")
    plt.title('loss', fontsize=8)

    plt.show()

def plot_path(L, T, x0, v0, a0, a1, norme_v1, direction_win, v_win,scale, theta, n_point, use_torch=False):
    #calcule des paramètre imposés

    x1 = x0+L*direction_win/np.linalg.norm(direction_win)
    v1 = v_win/np.linalg.norm(v_win)*norme_v1
    win = Window(x1, v_win, scale, theta)
    #########

    #########
    #calucule la trajectoire
    if use_torch:
        x0 = torch.tensor(x0, dtype=torch.float)
        x1 = torch.tensor(x1, dtype=torch.float)
        v0 = torch.tensor(v0, dtype=torch.float)
        v1 = torch.tensor(v1, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        t, val_u = get_path_torch(x0, x1, v0, v1, a0, a1, L, T, n_point)
        t = t.numpy()
        val_u = val_u.numpy()
    else:
        t, val_u = get_path(x0, x1, v0, v1, a0, a1, L, T, n_point)

    val_v = np.gradient(np.transpose(val_u), T/n_point)[0]
    norme_val_v = np.linalg.norm(val_v, axis=1)

    #########

    #########
    #plot u

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    def define_animation(x, y, z):

        # Création de la figure et du plot 3D
        line, = ax.plot(x, y, z)
        point, = ax.plot([x[0]], [y[0]], [z[0]], 'ro')

        # Fonction d'animation
        def animate(i):
            point.set_data([x[i]], [y[i]])
            point.set_3d_properties(z[i])
            return line, point,

        return animate

    animate = define_animation(val_u[0,:], val_u[1,:], val_u[2,:])
    ani = FuncAnimation(fig, animate, frames=n_point, interval=T/n_point*1000)

    #ax.plot3D(val_u[:,0], val_u[:,1], val_u[:,2], 'red')

    #########

    #########
    #plot la fenètre

    corner = win.get_corner()
    corner = np.column_stack([corner[:,0], corner[:,1], corner[:,2], corner[:,3], corner[:,0]])


    ax.plot3D(corner[0,:], corner[1,:], corner[2,:], 'green')

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    delta = 0.5*max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_c = (x_max + x_min)/2
    y_c = (y_max + y_min)/2
    z_c = (z_max + z_min)/2

    ax.set_xlim([x_c - delta, x_c + delta])
    ax.set_ylim([y_c - delta, y_c + delta])
    ax.set_zlim([z_c - delta, z_c + delta])

    #########

    #########
    #enregistre l'animation

    ani.save("data/animation.gif", writer="pillow")

    #########

    #########
    #plot v

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, norme_val_v)

    plt.subplot(2, 2, 2)
    plt.plot(t, val_v[:,0])

    plt.subplot(2, 2, 3)
    plt.plot(t, val_v[:,1])

    plt.subplot(2, 2, 4)
    plt.plot(t, val_v[:,2])

    #########

    plt.show()