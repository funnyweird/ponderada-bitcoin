from pathlib import Path
import json

import joblib
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

csv = Path("data/bitcoin.csv")

if csv.exists():
    dados = pd.read_csv(csv, parse_dates=["Date"])
else:
    historico = yf.Ticker("BTC-USD").history(
        start="2023-01-01",
        end="2026-01-01",
        interval="1d",
        auto_adjust=False,
    )
    if historico.empty:
        raise RuntimeError("O Yahoo Finance não retornou dados.")

    dados = historico[["Close"]].reset_index()
    dados["Date"] = pd.to_datetime(dados["Date"]).dt.strftime("%Y-%m-%d")
    dados.to_csv(csv, index=False)

dados["Date"] = pd.to_datetime(dados["Date"])
dados = dados.sort_values("Date").reset_index(drop=True)

if dados["Date"].duplicated().any() or dados["Close"].isna().any():
    raise ValueError("Há datas duplicadas ou preços ausentes no CSV.")

if not dados["Date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
    raise ValueError("O histórico precisa ter dias consecutivos.")

# Entradas: sete dias, do mais antigo ao mais recente.
# Resposta: fechamento do dia seguinte.
precos = dados["Close"].to_numpy()
X = [precos[i - 7:i] for i in range(7, len(precos))]
y = precos[7:]

corte = int(len(X) * 0.8)
if corte < 1 or corte >= len(X):
    raise ValueError("Dados insuficientes para treino e teste.")

modelo = LinearRegression()
corte_validacao = int(corte * 0.8)

X_ajuste = X[:corte_validacao]
y_ajuste = y[:corte_validacao]

X_validacao = X[corte_validacao:corte]
y_validacao = y[corte_validacao:corte]

linear = LinearRegression()
linear.fit(X_ajuste, y_ajuste)

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge.fit(X_ajuste, y_ajuste)

mae_linear_val = mean_absolute_error(
    y_validacao, linear.predict(X_validacao)
)
mae_ridge_val = mean_absolute_error(
    y_validacao, ridge.predict(X_validacao)
)
mae_ref_val = mean_absolute_error(
    y_validacao, [janela[-1] for janela in X_validacao]
)

usar_ridge = (
    mae_ridge_val < mae_linear_val
    and mae_ridge_val < mae_ref_val
)

print("Comparação na validação:")
print(json.dumps({
    "mae_linear_usd": float(mae_linear_val),
    "mae_ridge_usd": float(mae_ridge_val),
    "mae_referencia_usd": float(mae_ref_val),
    "ridge_aprovado": bool(usar_ridge),
}, indent=2))

# Reajusta o modelo escolhido usando todos os exemplos de treino.
if usar_ridge:
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    nome_modelo = "Ridge com StandardScaler"
else:
    modelo = LinearRegression()
    nome_modelo = "Regressão linear"

modelo.fit(X[:corte], y[:corte])

previsoes = modelo.predict(X[corte:])
mae = mean_absolute_error(y[corte:], previsoes)

referencia = [janela[-1] for janela in X[corte:]]
mae_referencia = mean_absolute_error(y[corte:], referencia)

joblib.dump(modelo, "models/modelo.joblib")

metricas = {
    "modelo": nome_modelo,
    "treino": corte,
    "teste": len(X) - corte,
    "ajuste_validacao": corte_validacao,
    "validacao": corte - corte_validacao,
    "mae_linear_validacao_usd": float(mae_linear_val),
    "mae_ridge_validacao_usd": float(mae_ridge_val),
    "mae_referencia_validacao_usd": float(mae_ref_val),
    "mae_usd": float(mae),
    "mae_referencia_usd": float(mae_referencia),
    "melhor_que_referencia": bool(mae < mae_referencia),
}

Path("models/metricas.json").write_text(
    json.dumps(metricas, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
Path("data/exemplo.json").write_text(
    json.dumps({"precos": precos[-7:].tolist()}, indent=2),
    encoding="utf-8",
)

print("Modelo salvo em models/modelo.joblib")
print(json.dumps(metricas, indent=2, ensure_ascii=False))