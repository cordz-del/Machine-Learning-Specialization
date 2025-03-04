import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Print model parameters
print("Intercept:", lin_reg.intercept_)
print("Coefficient:", lin_reg.coef_)

# Predict on new data
X_new = np.array([[0], [2]])
y_predict = lin_reg.predict(X_new)

# Visualize the results
plt.scatter(X, y, color='blue', label="Data")
plt.plot(X_new, y_predict, color='red', linewidth=2, label="Prediction")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()
