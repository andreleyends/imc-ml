from flask import Flask, request, jsonify, render_template
import os

import imc_modelo

app = Flask(__name__)

# ---------- CARGA INICIAL DEL MODELO (aprende de usuarios reales) ----------
entrenado_inicial = imc_modelo.entrenar()
if entrenado_inicial:
    r2_i, rmse_i = imc_modelo.metricas()
    print(f"[ML] Modelo cargado con {imc_modelo.numero_ejemplos()} ejemplos reales. "
          f"R2={r2_i if r2_i is None else round(r2_i, 4)}, RMSE={rmse_i if rmse_i is None else round(rmse_i, 3)}")
else:
    print("[ML] El modelo aún no ha aprendido: esperando el primer uso de la app.")


def cargar_float(datos, clave):
    val = datos.get(clave)
    if val in (None, ""):
        return None
    try:
        return float(str(val).replace(",", "."))
    except (TypeError, ValueError):
        return None


def estado_modelo():
    r2, rmse = imc_modelo.metricas()
    return {
        "entrenado": imc_modelo.modelo is not None,
        "ejemplos": imc_modelo.numero_ejemplos(),
        "r2": round(r2, 4) if r2 is not None else None,
        "rmse": round(rmse, 3) if rmse is not None else None,
        "confianza": imc_modelo.confianza_global(),
    }


@app.route("/")
def home():
    return render_template("index_imc.html", estado=estado_modelo())


@app.route("/api/estado")
def api_estado():
    return jsonify(estado_modelo())


@app.route("/api/datos")
def api_datos():
    return jsonify({"ejemplos": imc_modelo.listar_ejemplos()})


@app.route("/api/calcular", methods=["POST"])
def calcular():
    data = request.get_json(silent=True) or {}
    peso = cargar_float(data, "peso")
    altura = cargar_float(data, "altura")

    if not (peso and altura and peso > 0 and altura > 0):
        return jsonify({"error": "Ingresa peso y altura válidos y mayores que 0."}), 400

    # Aprende de este uso: el IMC real se obtiene con la fórmula clásica.
    imc_modelo.aprender_de_uso(peso, altura)

    imc, confianza = imc_modelo.predecir_con_confianza(peso, altura)
    imc_real = imc_modelo.imc_formula(peso, altura)
    clas, color = imc_modelo.clasificar(imc)
    confianza = min(confianza, 99) if confianza is not None else None
    return jsonify({
        "imc": round(imc, 2),
        "imc_real": round(imc_real, 2),
        "clasificacion": clas,
        "color": color,
        "confianza": confianza,
        "ejemplos": imc_modelo.numero_ejemplos(),
        "recomendaciones": imc_modelo.recomendaciones(clas),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"[Flask] Servidor en http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)