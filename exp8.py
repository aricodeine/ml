import numpy as np

# Step 1: Define the input data
X = np.array([[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1], [2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9]])

# Step 2: Subtract the mean from the data
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# Step 4: Calculate the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort the eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Choose the number of components and project the data onto the new subspace
n_components = 2
components = eigenvectors[:, :n_components]
X_pca = np.dot(X_centered, components)
print(cov_matrix)
# Print the original data and the transformed data
print("Original Data:")
print(X)
print("Transformed Data:")
print(X_pca)