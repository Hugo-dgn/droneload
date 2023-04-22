import numpy as np

# Définir la fonction f
def f(x, y):
    A = x*y
    B = x+y
    C = x-y
    M = np.stack((A, B, C), axis=2)
    return M

# Définir les listes numpy x et y
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Créer une grille 2D de toutes les combinaisons possibles entre les éléments de x et y
X, Y = np.meshgrid(x, y)

# Appliquer la fonction f à chaque couple de coordonnées
M = f(X, Y)

# Afficher la matrice M
print(M)

def my_func(v):
    return v[0] + v[1] + v[2]

print(M.shape)
M = np.apply_along_axis(my_func, 2, M)
print(M)

