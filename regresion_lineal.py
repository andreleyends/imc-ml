import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([-5, 0, 10, 20, 30]).reshape(-1, 1)
y = np.array([23, 32, 50, 68, 86])

modelo = LinearRegression()
modelo.fit(X, y)

print(f"Pendiente (m): {modelo.coef_[0]:.4f}")
print(f"Intercepto (b): {modelo.intercept_:.4f}")

nuevos = np.array([[100], [38], [-40]])
predicciones = modelo.predict(nuevos)
for valor, pred in zip(nuevos, predicciones):
    print(f"Celsius {valor[0]} -> Fahrenheit {pred:.2f}")

pred_train = modelo.predict(X)
print(f"MSE: {mean_squared_error(y, pred_train):.4f}")
print(f"R2: {r2_score(y, pred_train):.4f}")

formula = f"F = {modelo.coef_[0]:.4f} * C + {modelo.intercept_:.4f}"
print(f"Formula encontrada: {formula}")