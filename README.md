# Actividad 4 – ML2: Redes Neuronales para Predicción de Churn (Telco)

Repositorio correspondiente a la **Actividad 4 de Machine Learning II**, donde se implementan y comparan modelos de Deep Learning (MLP y CNN 1D) sobre el dataset **Telco Churn**, manteniendo el esquema base de preprocesamiento (imputación, one-hot encoding y escalamiento).

---

## 🎯 Objetivo

Predecir la probabilidad de que un cliente realice **Churn** (abandono del servicio), formulado como un problema de **clasificación binaria desbalanceada**.

---

## 📦 Dataset

- Fuente: `data/data-churn.csv`
- Observaciones: ~7.043 clientes
- Target: `Churn` (Yes/No → 1/0)
- Proporción clase positiva (churn): ~26.5%

---

## ⚙️ Preprocesamiento (Base solicitada)

Se mantuvo el esquema estándar solicitado:

- **Numéricas**: imputación mediana + `StandardScaler`
- **Categóricas**: imputación por moda + `OneHotEncoder(handle_unknown="ignore")`
- Split train/test estratificado (80/20)
- Métrica principal: **F1 sobre clase positiva (churn=1)**

---

# ✅ Paso 1 — MLP (Perceptrón Multi-Capa)

## Arquitectura

- Dense(64, ReLU) + Dropout
- Dense(32, ReLU) + Dropout
- Dense(1, Sigmoid)

**Loss:** Binary Crossentropy  
**Optimizador:** Adam

## Curvas de entrenamiento
![MLP Loss](figures/mlp_loss.png)
![MLP AUC](figures/mlp_auc.png)

## Resultados en test (MLP)

- Accuracy: 0.7807  
- Precision: 0.5910  
- Recall: 0.5642  
- F1: 0.5773  
- ROC-AUC: 0.8366  
- PR-AUC: 0.6345  

**Gráficos:**
![MLP CM](figures/mlp_cm.png)
![MLP ROC](figures/mlp_roc.png)
![MLP PR](figures/mlp_pr.png)

---

# ✅ Paso 2 — Experimentos (Learning Rate y Batch Size)

Se evalúa el impacto en convergencia, estabilidad y tiempo de entrenamiento.

### Learning Rate (comparación)
(Insertar tabla/resultado del notebook)

### Batch Size (16/32/64)
(Insertar tabla/resultado del notebook)

---

# ✅ Paso 3 — CNN 1D

## Justificación

Aunque churn es tabular, se reinterpretan features como señal 1D para explorar detección de patrones locales (kernels + pooling).

## Arquitectura (CNN)

- Conv1D + MaxPooling
- Conv1D + GlobalMaxPooling
- Dense final + Sigmoid

## Curvas y resultados (CNN)

![CNN Loss](figures/cnn_loss.png)
![CNN AUC](figures/cnn_auc.png)

**Resultados en test (CNN 1D):**
- Accuracy: 0.7956  
- Precision: 0.6453  
- Recall: 0.5107  
- F1: 0.5701  
- ROC-AUC: 0.8387  
- PR-AUC: 0.6323  

**Gráficos:**
![CNN CM](figures/cnn_cm.png)
![CNN ROC](figures/cnn_roc.png)
![CNN PR](figures/cnn_pr.png)

---

# ✅ Paso 4 — Comparación final y análisis crítico

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.7807 | 0.5910 | 0.5642 | 0.5773 | 0.8366 | 0.6345 |
| CNN 1D | 0.7956 | 0.6453 | 0.5107 | 0.5701 | 0.8387 | 0.6323 |

**Lectura crítica:**
- MLP ofrece mejor equilibrio (F1/Recall) para un problema de retención.
- CNN aumenta precision pero reduce recall, lo que podría dejar escapar churn reales.
- En dataset tabular moderado, modelos clásicos pueden competir fuertemente con redes.

---

## 🧾 Reproducibilidad

Instalación:

```bash
pip install -r requirements.txt
