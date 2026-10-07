import os
import joblib
import pandas as pd

def carregar_modelo(caminho_modelo):
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo nao encontrado em: {caminho_modelo}")
    return joblib.load(caminho_modelo)

def fazer_previsao(modelo, dados_entrada):
    if isinstance(dados_entrada, dict):
        dados_entrada = pd.DataFrame([dados_entrada])
    
    previsoes = modelo.predict(dados_entrada)
    
    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(dados_entrada)
        return previsoes, probabilidades
    
    return previsoes, None

if __name__ == "__main__":
    caminho = os.path.join("modelos", "modelo_final.pkl")
    
    try:
        modelo = carregar_modelo(caminho)
        
        exemplo_dados = pd.DataFrame([{
            'feature1': 1.0,
            'feature2': 2.0
        }])
        
        preds, probs = fazer_previsao(modelo, exemplo_dados)
        print("Previsao:", preds)
        if probs is not None:
            print("Probabilidade:", probs)
            
    except Exception as e:
        print("Erro ao executar predicao:", e)
