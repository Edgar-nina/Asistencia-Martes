import numpy as np

def gauss_seidel(A, b, x0, tol=1e-6, max_iterations=100):
    """
    Resuelve el sistema de ecuaciones lineales Ax = b usando el método de Gauss-Seidel.
    
    :param A: Matriz de coeficientes (numpy array)
    :param b: Vector del lado derecho (numpy array)
    :param x0: Adivinanza inicial para la solución (numpy array)
    :param tol: Tolerancia para el criterio de parada
    :param max_iterations: Máximo número de iteraciones
    :return: Aproximación de la solución (numpy array)
    """
    n = len(b)
    x = x0.copy()
    
    
    errors = []
    
    for iteration in range(max_iterations):
        x_new = x.copy()
        
        for i in range(n):
            
            s1 = sum(A[i][j] * x_new[j] for j in range(i))  # Términos actualizados
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))  # Términos no actualizados
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        
        
        print(f"Iteración {iteration + 1}: {x_new}, Error: {error}")
        
        
        if error < tol:
            print(f"Convergió en {iteration + 1} iteraciones.")
            return x_new, errors
        
        x = x_new
    
    print("No convergió.")
    return x, errors


A = np.array([[3, -0.1, -0.2],
              [0.1, 7, -0.3],
              [0.3, -0.2, 10]], dtype=float)
b = np.array([7.85, -19.3, 71.4], dtype=float)


x0 = np.zeros(len(b))


solution, errors = gauss_seidel(A, b, x0)


print("Solución:", solution)


import matplotlib.pyplot as plt

plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.xlabel('Iteración')
plt.ylabel('Error')
plt.title('Convergencia del Método de Gauss-Seidel')
plt.yscale('log')  # Escala logarítmica para visualizar la convergencia
plt.grid(True)
plt.show()