import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.animation import FuncAnimation
from window.path import get_path
from window.window import Window
from skimage.measure import marching_cubes


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
    ani = FuncAnimation(fig, animate, frames=n_point, interval=T/n_point*1000, blit=True)

    #########

    #########
    #plot la fenètre

    corner = win.get_corner()
    corner = np.column_stack([corner[:,0], corner[:,1], corner[:,2], corner[:,3], corner[:,0]])
    corner = corner.transpose()


    ax.plot3D(corner[:,0], corner[:,1], corner[:,2], 'green')

    #équilibre les axes

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    m_x = sum(xlim)/2
    m_y = sum(ylim)/2
    m_z = sum(ylim)/2

    d_x = xlim[1] - xlim[0]
    d_y = ylim[1] - ylim[0]
    d_z = zlim[1] - zlim[0]

    d = max([d_x, d_y, d_z])

    ax.set_xlim((m_x - d/1.9, m_x + d/1.9))
    ax.set_ylim((m_y - d/1.9, m_y + d/1.9))
    ax.set_zlim((m_z - d/1.9, m_z + d/1.9))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    #########

    #########
    #enregistre l'animation

    ani.save("data/animation.gif", writer="pillow")

    #########

    #########
    #plot v

    plt.figure()

    plt.plot(t, norme_val_v)

    plt.xlabel("t (s)")
    plt.ylabel("v (m/s)")

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

def plot_iso_s(M, threshold, lenght_side, v_win, scale, theta):
    x_win = np.array([len(M)//2, len(M)//2, len(M)//2])
    win = Window(x_win, v_win*len(M)/lenght_side, scale, theta)

    verts, faces, _, _ = marching_cubes(M, threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap='viridis', alpha=0.5)

    n_tiks = len(ax.get_xticks())
    new_ticklabels = np.linspace(0, 2*lenght_side, n_tiks)

    ax.set_xticklabels(np.round(new_ticklabels))
    ax.set_yticklabels(np.round(new_ticklabels))
    ax.set_zticklabels(np.round(new_ticklabels))

    ax.set_xlim(0, len(M))
    ax.set_ylim(0, len(M))
    ax.set_zlim(0, len(M))

    u, v, w = v_win/np.linalg.norm(v_win)*len(M)/3
    ax.quiver(x_win[0], x_win[1], x_win[2], u, v, w, arrow_length_ratio=0.1)

    corner = win.get_corner()
    corner = np.column_stack([corner[:,0], corner[:,1], corner[:,2], corner[:,3], corner[:,0]])
    corner = corner.transpose()

    ax.plot3D(corner[:,0], corner[:,1], corner[:,2], 'green')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    # Affichage du tracé
    plt.show()