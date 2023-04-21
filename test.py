import numpy as np

# Définition des données de la fonction
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)**2

# Calcul de l'intégrale
integral = np.trapz(y, x)

print("L'intégrale de la fonction est :", integral)