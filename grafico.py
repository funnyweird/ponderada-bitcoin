from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import pandas as pd

# Repete a organização dos dados usada no treinamento.
dados = pd.read_csv("data/bitcoin.csv", parse_dates=["Date"])
dados = dados.sort_values("Date").reset_index(drop=True)

precos = dados["Close"].to_numpy()
X = [precos[i - 7:i] for i in range(7, len(precos))]
y = precos[7:]
datas = dados["Date"].iloc[7:]

corte = int(len(X) * 0.8)

# Carrega o modelo existente, sem treinar novamente.
modelo = joblib.load("models/modelo.joblib")
previsoes = modelo.predict(X[corte:])
referencia = [janela[-1] for janela in X[corte:]]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(datas.iloc[corte:], y[corte:],
        label="Preço real", color="#222222", linewidth=1.8)
ax.plot(datas.iloc[corte:], previsoes,
        label="Regressão linear", color="#2563eb", alpha=0.85)
ax.plot(datas.iloc[corte:], referencia,
        label="Referência: último fechamento",
        color="#d97706", linestyle="--", alpha=0.75)

ax.set_title("Bitcoin: preço real e previsões no conjunto de teste")
ax.set_xlabel("Data do fechamento previsto")
ax.set_ylabel("Preço de fechamento (US$)")
ax.legend()
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()

Path("evidencias").mkdir(exist_ok=True)
fig.savefig("evidencias/grafico-teste.png", dpi=150)
plt.close(fig)

print("Gráfico salvo em evidencias/grafico-teste.png")