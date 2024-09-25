import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# Load the dataset
ds = pd.read_csv("Factors.csv")
ds = ds.select_dtypes(exclude=['object', 'category'])
# Define features and observation matrix
X = np.array(ds.iloc[:, :-1].values)
y = np.array(ds.iloc[:, -1].values) 

# Create bias term
ones = np.ones((X.shape[0], 1))
X = np.c_[ones, X]

# Calculate weights/coefficients
weights = np.linalg.inv(X.T @ X) @ X.T @ y 
print("Weights:", weights)

# Prediction matrix 
predictions = np.array(X @ weights)
print(predictions)

# mse/error
mse = (np.linalg.norm(y-predictions)**2)/len(y)


print(f"MSE is: {mse}")


plt.figure(figsize=(10, 6))
plt.scatter(y, predictions, color='black', label='Predicted Values', alpha=0.6)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values with Line of Best Fit')
plt.legend()
plt.grid()
plt.show()