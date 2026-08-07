import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generar_datos(n=20000, seed=42):
    """Genera un dataset sintético de datos 'reales' de personas."""
    rng = np.random.default_rng(seed)
    # Alturas realistas entre 1.20 y 2.20 m
    altura = rng.uniform(1.20, 2.20, n)
    # Pesos entre 35 y 160 kg
    peso = rng.uniform(35, 160, n)
    # El IMC depende del peso y la altura (solo para ETIQUETAR los datos)
    imc = peso / (altura ** 2)
    # Pequeño ruido para simular datos reales
    imc = imc + rng.normal(0, 0.3, n)
    X = np.column_stack([peso, altura])
    return X, imc