import math

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
modelo = joblib.load("models/modelo.joblib")


@app.get("/health")
def health():
    return jsonify(status="ok", modelo_carregado=True)


@app.post("/predict")
def predict():
    dados = request.get_json(silent=True)
    precos = dados.get("precos") if isinstance(dados, dict) else None

    if not isinstance(precos, list) or len(precos) != 7:
        return jsonify(erro="Envie sete preços em 'precos'."), 400

    if any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or p <= 0
        for p in precos
    ):
        return jsonify(erro="Os preços devem ser números positivos e finitos."), 400

    previsao = float(modelo.predict([precos])[0])
    return jsonify(previsao_usd=previsao, horizonte="dia seguinte")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)