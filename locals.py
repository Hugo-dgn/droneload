import numpy as np

#########
#Paramètre de la fenètre

direction_win = np.array([ 1,  0,  0])
v_win = np.array([1, 1, 0])
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

L = 1.8
T = 2.4
n_point = 1000

conv_L_m = 1
conv_T_s = 1

#########

#########
#Paramètres de dimensionement pour calculer la matrice de loss

L_min = 1
L_max = 15

T_min = 1
T_max = 15

n_T = 100
n_L = 100

n_angle_phi = 100
n_angle_theta = 100

##########