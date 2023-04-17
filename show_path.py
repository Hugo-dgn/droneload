import window
import matplotlib.pyplot as plt
import numpy as np

###############
#paramètre de la fenetre

x = [4, 4, 4]
v = np.array([0.1, 0, 0])
scale = 1
theta = 0

###############

###############
#position et direction initiales

x1 = [0, 0, 0]

v2_norme = 1.5

###############

x2 = x

v1 = np.array(x2) - np.array(x1)
v2 = v/np.linalg.norm(v)*v2_norme

win = window.Window(x, v, scale, theta)

corner = win.get_corner()
corner = np.column_stack([corner[:,0], corner[:,1], corner[:,2], corner[:,3], corner[:,0]])
corner = corner.transpose()

u = window.get_path(x1, x2, v1, v2)

T = np.linspace(0, 1, 1000)
Y = np.array([u(t) for t in T])

fig = plt.figure()

ax = plt.axes(projection="3d")

x=Y[:,0]
y=Y[:,1]
z=Y[:,2]

ax.plot3D(x, y, z, 'red')

ax.plot3D(corner[:,0], corner[:,1], corner[:,2], 'blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()