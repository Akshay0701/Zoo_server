import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Parameters
Du = 0.16  # Diffusion rate of U
Dv = 0.08  # Diffusion rate of V
F = 0.035  # Feed rate
k = 0.065  # Kill rate

# Grid size
N = 256

# Time step
dt = 1.0

# Number of iterations
num_iterations = 10000

# Initialize U and V
U = np.ones((N, N))
V = np.zeros((N, N))

# Initial condition: add some noise
r = 20
U[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50
V[N//2-r:N//2+r, N//2-r:N//2+r] = 0.25
U += 0.05 * np.random.random((N, N))
V += 0.05 * np.random.random((N, N))

# Laplacian kernel
laplacian_kernel = np.array([[0.05, 0.2 , 0.05],
                             [0.2 , -1.0, 0.2 ],
                             [0.05, 0.2 , 0.05]])

# Reaction-diffusion function
def update(U, V, Du, Dv, F, k):
    laplacian_U = convolve(U, laplacian_kernel, mode='constant')
    laplacian_V = convolve(V, laplacian_kernel, mode='constant')
    
    reaction = U * V**2
    dU = Du * laplacian_U - reaction + F * (1 - U)
    dV = Dv * laplacian_V + reaction - (F + k) * V
    
    U += dU * dt
    V += dV * dt
    
    return U, V

# Run the simulation
for i in range(num_iterations):
    U, V = update(U, V, Du, Dv, F, k)
    if i % 1000 == 0:  # Plot every 1000 iterations
        plt.imshow(U, cmap='inferno')
        plt.title(f"Iteration {i}")
        plt.axis('off')
        plt.show()

# Final plot
plt.imshow(U, cmap='inferno')
plt.title("Final pattern")
plt.axis('off')
plt.show()
