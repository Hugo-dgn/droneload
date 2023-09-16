import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


import droneload

######################################## rectFinder ########################################


target_rect_corners = np.array(
        [[-7.5, 0, -5],
         [7.5, 0, -5],
         [7.5, 0, 5],
         [-7.5, 0, 5]]
    )


def video_rectangle(args):
    
    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)
    
    while True:
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_sobel(image)
        rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
        
        droneload.rectFinder.remove_old_rects(10)

        for rect in rects:
            rect.define_3D(target_rect_corners)
            center = rect.center2D()
            rect.fit(args.fit)
            pos, retval, rvec, tvec = rect.compute()
            droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)
        droneload.rectFinder.draw_rectangles(frame, rects)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()

def image_rectangle(args):
    frame  = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_sobel(image)
    rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
    
    droneload.rectFinder.remove_old_rects(10)

    for rect in rects:
        center = rect.center()
        rect.compute(target_rect)

        droneload.rectFinder.draw_coordinate(frame, center, rect.rvecs, rect.tvecs)

    droneload.rectFinder.draw_rectangles(frame, rects)
        
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
######################################## pathFinder ######################################## 

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

def init_scene():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])
    
    return ax

def animate_scene(args):
    
    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)
    
    ax = init_scene()
    
    while True:
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_sobel(image)
        rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
        
        droneload.rectFinder.remove_old_rects(10)
        
        ax.clear()
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])

        for rect in rects:
            rect.define_3D(target_rect_corners)
            center = rect.center2D()
            rect.fit(args.fit)
            pos, retval, rvec, tvec = rect.compute()
            
            if retval:
                droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)
                
                window = droneload.pathFinder.Window(target_rect_corners)
                
                x0 = pos
                n = window.n * np.sign(np.dot(window.n, window.p-x0))
                v1 = n
                L = np.linalg.norm(window.p-x0)
                T = 10
                n_point = 1000
                
                t, path = droneload.pathFinder.get_path(x0=x0, x1=window.p, v0=[0, 0, 0], v1=v1, a0=[0, 0, 0], a1=[0, 0, 0], L=L, T=T, n_point=n_point)
                
                droneload.pathFinder.draw_path_plt(ax, path)
                droneload.pathFinder.draw_path_cv(frame, path, rvec, tvec)
                
            
        droneload.rectFinder.draw_scene(ax)
        droneload.rectFinder.draw_rectangles(frame, rects)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()
    
    
def plot_scene(window, ax):
    corners = window.corners.copy().T
    corners = np.column_stack([corners[:,0], corners[:,1], corners[:,2], corners[:,3], corners[:,0]])


    ax.plot3D(corners[0,:], corners[1,:], corners[2,:], 'green')

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

def plot_analyse_total(Loss_angle, n_angle_phi, n_angle_theta):
    plt.figure()

    plt.imshow(Loss_angle, norm=LogNorm(), extent=[-np.pi, np.pi, -np.pi/2, np.pi/2])

    i = np.unravel_index(np.argmin(Loss_angle), Loss_angle.shape)

    print(f"minimal loss = {Loss_angle[i]}")
    phi_min = -np.pi + (n_angle_phi-i[0]-1)*(2*np.pi)/n_angle_phi
    theta_min = -np.pi/2 + i[1]*(np.pi)/n_angle_theta
    window.p = np.array([np.cos(phi_min)*np.cos(theta_min), np.sin(phi_min)*np.cos(theta_min), np.sin(theta_min)])
    print(f"phi = {phi_min}")
    print(f"theta = {theta_min}")
    print(f"direction = {window.p}")

    plt.colorbar()
    plt.xlabel(f"phi (rad)")
    plt.ylabel(f"theta (rad)")
    plt.title('loss', fontsize=8)

    plt.show()

def _plot_path(T, x0, v0, a0, a1, norme_v1, window, n_point):
    #calcule des paramètre imposés
    
    L = np.linalg.norm(window.p - x0)
    
    a0 = np.array(a0)
    a1 = np.array(a1)
    v0 = np.array(v0)
    x0 = np.array(x0)
    
    window.n = np.sign(window.n @ (window.p - x0)) * window.n

    x1 = window.p
    v1 = window.n/np.linalg.norm(window.n)*norme_v1
    #########

    #########
    #calucule la trajectoire
    
    t, val_u = droneload.pathFinder.get_path(x0, x1, v0, v1, a0, a1, L, T, n_point)

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

    plot_scene(window, ax)

    #########

    #########
    #enregistre l'animation

    #ani.save("data/animation.gif", writer="pillow")

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
    
def plot_path(args):
    target_rect.corners += np.array([10, 10, 10])
    corners = target_rect.corners
    window = droneload.pathFinder.Window(corners)
    
    _plot_path(args.T, args.x0, args.v0, args.a0, args.a1, args.norme_v1, window, args.n_point)

def image_path(args):
    frame  = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_sobel(image)
    rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
    
    droneload.rectFinder.draw_rectangles(frame, rects)

    if len(rects) == 1:
        rect = rects[0]
        center = rect.center()
        retval, rvecs, tvecs, inliers = droneload.rectFinder.get_3D_vecs(target_rect, rect)
        droneload.rectFinder.draw_coordinate(frame, center, rvecs, tvecs)
    else:
        message = "More than 1 rectangle found"
        raise ValueError(message)
    
    cv2.imshow("frame", frame)
    cv2.waitKey(100)
    
    corners = target_rect.corners
    
    x0 = np.array([0, 0, 0])
    
    corners = droneload.rectFinder.find_pos_3D(corners, tvecs, rvecs)
        
    window = droneload.pathFinder.Window(corners)
    
    _plot_path(args.T, x0, args.v0, args.a0, args.a1, args.norme_v1, window, args.n_point)