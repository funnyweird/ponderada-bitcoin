import json
from pathlib import Path
from urllib.request import Request, urlopen

with urlopen("http://localhost:8000/health", timeout=10) as resposta:
    print("Status:", resposta.read().decode())

dados = Path("data/exemplo.json").read_bytes()

pedido = Request(
    "http://localhost:8000/predict",
    data=dados,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(pedido, timeout=10) as resposta:
    resultado = json.load(resposta)
    print("Predição:", resultado)