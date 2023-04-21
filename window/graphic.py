import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.animation import FuncAnimation
from window.path import get_path
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

def plot_path(L, T, x0, v0, a0, a1, norme_v1, direction_win, v_win,scale, theta, n_point):
    #calcule des paramètre imposés

    x1 = x0+L*direction_win/np.linalg.norm(direction_win)
    v1 = v_win/np.linalg.norm(v_win)*norme_v1
    win = Window(x1, v_win, scale, theta)
    #########

    #########
    #calucule la trajectoire

    t, val_u = get_path(x0, x1, v0, v1, a0, a1, L, T, n_point)

    val_v = np.gradient(np.transpose(val_u), T/n_point)[0]
    norme_val_v = np.linalg.norm(val_v, axis=1)

    #########

    #########
    #plot u

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
    corner = corner.transpose()


    ax.plot3D(corner[:,0], corner[:,1], corner[:,2], 'green')

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