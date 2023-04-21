import window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from locals import *
import loss
#calcule des paramètre imposés

x1 = x0+L*direction_win/np.linalg.norm(direction_win)
v1 = v_win/np.linalg.norm(v_win)*norme_v1
win = window.Window(x1, v_win, scale, theta)
#########

#########
#calucule la trajectoire

t, val_u = window.get_path(x0, x1, v0, v1, a0, a1, L, T, n_point)

val_v = np.gradient(np.transpose(val_u), T/n_point)[0]
norme_val_v = np.linalg.norm(val_v, axis=1)

#########

#########
#print les loss

print(f"loss={loss.loss(t, val_u, L, T)}")
print("")
print(f"loss_max_v={loss.loss_max_v(t, val_u, L, T)}")
print(f"loss_max_a={loss.loss_max_a(t, val_u, L, T)}")
print(f"loss_max_v_a{loss.loss_max_v_a(t, val_u, L, T)}")
print(f"loss_id_v_dist={loss.loss_id_v_dist(t, val_u, L, T)}")

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

