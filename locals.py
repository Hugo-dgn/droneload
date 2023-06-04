import numpy as np

#########
#Paramètre de la fenètre

direction_win = np.array([ 1,  1,  1])
v_win = np.array([1, 0, 0])
scale = 1
theta = 0

#########
#Paramètres du mouvement

x0 = np.array([0, 0, 0])
v0 = np.array([0, 0, 0])
norme_v1 = 1
a0 = np.array([0, 0, 0])
a1 = np.array([0, 0, 0])

#########

#########
#Paramètres de dimensionement pour calculer un path

L = 5
T = 10
n_point = 100

conv_L_m = 1
conv_T_s = 1

#########

#########
#Paramètres de dimensionement pour calculer la matrice de loss

L_min = 1
L_max = 15

T_min = 1
T_max = 15

n_T = 1000
n_L = 1000

n_angle_phi = 100
n_angle_theta = 100

##########
#Paramètres de dimensionement pour calculer les isosurface

lenght_side = 6
n_point_path = 1000
n_point_world_matrix = 30
threshold = 50

##########
#paramètres video

pool_size = 1 #réduction de la résolution

#détection de rectangles
alpha = 2 #rapport en petit et grand coté du rectangle (alpha < 1)

alpha_1 = 1000 #facteur de loss devant delta_1
alpha_2 = 1000 #facteur de loss devant delta_2
tol = 10