import os
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor

DATA_FILE = os.environ.get(
    "IMC_DATA_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_imc.csv"),
)

modelo = None
X = None
y = None


def imc_formula(peso, altura):
    """IMC real calculado con la fórmula estándar: peso / altura²."""
    return peso / (altura ** 2)


def cargar_ejemplos():
    """Carga los ejemplos aprendidos de usuarios. Devuelve listas de peso, altura e imc."""
    pesos, alturas, imcs = [], [], []
    if not os.path.exists(DATA_FILE):
        return pesos, alturas, imcs
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for fila in reader:
            if len(fila) != 3:
                continue
            try:
                pesos.append(float(fila[0]))
                alturas.append(float(fila[1]))
                imcs.append(float(fila[2]))
            except ValueError:
                continue
    return pesos, alturas, imcs


def guardar_ejemplo(peso, altura, imc):
    """Guarda un ejemplo real proporcionado por un usuario."""
    escribir_header = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if escribir_header:
            writer.writerow(["peso", "altura", "imc"])
        writer.writerow([peso, altura, imc])


def aprender_de_uso(peso, altura):
    """Guarda un ejemplo (el IMC real sale de la fórmula) y reentrena el modelo."""
    guardar_ejemplo(peso, altura, imc_formula(peso, altura))
    return entrenar()


def listar_ejemplos():
    """Devuelve los ejemplos guardados como lista de dicts para mostrarlos."""
    pesos, alturas, imcs = cargar_ejemplos()
    return [
        {"peso": round(p, 2), "altura": round(a, 3), "imc": round(i, 2)}
        for p, a, i in zip(pesos, alturas, imcs)
    ]


def entrenar():
    """Entrena el modelo con los ejemplos guardados. Si no hay datos, el modelo queda vacío."""
    global modelo, X, y
    pesos, alturas, imcs = cargar_ejemplos()
    if not pesos:
        modelo = None
        X = None
        y = None
        return False
    X = np.column_stack([pesos, alturas])
    y = np.array(imcs)
    modelo = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
    modelo.fit(X, y)
    return True


def predecir(peso, altura):
    """Predice el IMC con el modelo. Retorna None si el modelo aún no ha aprendido."""
    if modelo is None:
        return None
    return float(modelo.predict(np.array([[peso, altura]]))[0])


def predecir_con_confianza(peso, altura):
    """Predice el IMC y calcula una confianza (0-100) a partir de la dispersión
    entre las predicciones de cada árbol del bosque y la cantidad de datos."""
    if modelo is None:
        return None, None
    x = np.array([[peso, altura]])
    imcs = np.array([tree.predict(x)[0] for tree in modelo.estimators_])
    imc = float(np.mean(imcs))
    std = float(np.std(imcs))
    n = numero_ejemplos()

    confianza_datos = np.clip((n / 20.0) ** 0.7, 0.0, 1.0)
    confianza_arboles = np.clip(1.0 - (std / max(abs(imc), 1e-6)) * 4.0, 0.0, 1.0)
    confianza = round(100 * confianza_datos * confianza_arboles)
    return imc, min(confianza, 99)


def metricas():
    """R² y RMSE del modelo sobre los datos aprendidos. None si no hay datos suficientes."""
    if modelo is None or X is None or y is None or len(X) < 2:
        return None, None
    pred = modelo.predict(X)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - pred) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return r2, rmse


def numero_ejemplos():
    return len(X) if X is not None else 0


def confianza_global():
    """Confianza global del modelo (0-100): combina R² y volumen de datos."""
    r2, _ = metricas()
    if modelo is None or r2 is None:
        return None
    n = numero_ejemplos()
    confianza_r2 = np.clip(max(r2, 0.0) * 100, 0, 95)
    confianza_datos = np.clip((n / 20.0) ** 0.7, 0.0, 1.0) * 100
    return round(max(confianza_r2, 5) * 0.6 + confianza_datos * 0.4)


RECOMENDACIONES = {
    "Bajo peso": {
        "titulo": "Recomendaciones para bajo peso",
        "items": [
            "Aumenta tu ingesta calórica con alimentos nutritivos y densos en energía (frutos secos, palta, aceites saludables).",
            "Incluye proteínas de calidad en cada comida: huevos, pollo, pescado, legumbres y lácteos.",
            "Realiza entrenamiento de fuerza para ganar masa muscular, no solo grasa.",
            "Consulta con un nutricionista para descartar causas médicas del bajo peso.",
        ],
    },
    "Peso normal": {
        "titulo": "Mantén tu peso saludable",
        "items": [
            "Sigue una dieta equilibrada: mitad de tu plato de vegetales, un cuarto de proteínas y un cuarto de carbohidratos.",
            "Realiza al menos 150 minutos de actividad física moderada a la semana.",
            "Duerme entre 7 y 9 horas: afecta directamente el apetito y el metabolismo.",
            "Controla tu IMC periódicamente para detectar cambios a tiempo.",
        ],
    },
    "Sobrepeso": {
        "titulo": "Recomendaciones para sobrepeso",
        "items": [
            "Reduce alimentos ultraprocesados, azúcares añadidos y bebidas azucaradas.",
            "Incorpora caminatas diarias: empieza con 30 minutos y aumenta gradualmente.",
            "Sirve porciones más pequeñas y come despacio para sentirte satisfecho antes.",
            "Busca apoyo profesional para un plan de pérdida de peso sostenible, no dietas extremas.",
        ],
    },
    "Obesidad": {
        "titulo": "Recomendaciones para obesidad",
        "items": [
            "Consulta con un médico o nutricionista para un plan personalizado y seguimiento.",
            "Comienza con actividad física de bajo impacto (natación, bicicleta, caminata).",
            "Cambia bebidas por agua e incluye fibra para mejorar la saciedad.",
            "Establece metas pequeñas y medibles de peso; evita perder más de 0.5-1 kg por semana.",
        ],
    },
}


def recomendaciones(clas):
    """Devuelve recomendaciones según la clasificación del IMC."""
    return RECOMENDACIONES.get(clas, RECOMENDACIONES["Peso normal"])


def clasificar(imc):
    if imc is None:
        return None, None
    if imc < 18.5:
        return "Bajo peso", "#1e40af"
    if imc < 25:
        return "Peso normal", "#065f46"
    if imc < 30:
        return "Sobrepeso", "#92400e"
    return "Obesidad", "#991b1b"