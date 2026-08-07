from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from imc_modelo import generar_datos

app = Flask(__name__)

# ---------- ENTRENAMIENTO DEL MODELO (ML) ----------
X, y = generar_datos(n=20000)
modelo = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
modelo.fit(X, y)

# Muestras de evaluación (fuera, no usadas tras entrenar)
pred = modelo.predict(X)
rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
print(f"[ML] Modelo entrenado. RMSE={rmse:.3f}, R2={r2:.4f}")


def clasificar(imc):
    if imc < 18.5:
        return "Bajo peso", "#1e40af"
    if imc < 25:
        return "Peso normal", "#065f46"
    if imc < 30:
        return "Sobrepeso", "#92400e"
    return "Obesidad", "#991b1b"


@app.route("/")
def home():
    return render_template("index_imc.html", rmse=round(rmse, 3), r2=round(r2, 4))


@app.route("/api/calcular", methods=["POST"])
def calcular():
    data = request.get_json()
    peso = float(data.get("peso"))
    altura = float(data.get("altura"))  # en metros

    if not (peso > 0 and altura > 0):
        return jsonify({"error": "Valores inválidos"}), 400

    entrada = np.array([[peso, altura]])
    imc = float(modelo.predict(entrada)[0])
    clas, color = clasificar(imc)
    return jsonify({"imc": round(imc, 2), "clasificacion": clas, "color": color})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"[Flask] Servidor en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)