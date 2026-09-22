# Predicción de Suscripción a Depósito Bancario (Bank Marketing)

Proyecto grupal (3 integrantes) — Licenciatura en Ciencia de Datos (UNGB)

## Objetivo

Predecir si un cliente contratará un depósito a plazo (`y`) a partir de
datos de campañas de marketing telefónico de un banco, con foco en detectar
la mayor cantidad posible de clientes reales interesados (recall de la
clase minoritaria).

## Dataset

[Bank Marketing (UCI Machine Learning Repository, id=222)](https://archive.ics.uci.edu/dataset/222/bank+marketing),
con más de 45.000 registros y variables demográficas, económicas y de
contacto de campaña.

## Flujo de trabajo

1. **Preparación de datos:**
   - Binarización de la variable objetivo (`yes`/`no` → 1/0).
   - Tratamiento de valores nulos (categoría "Unknown" en variables de
     texto, mediana en numéricas).
   - Partición en Train (80%) / Dev (10%) / Test (10%), estratificada.
   - One-Hot Encoding de variables categóricas y escalado con
     `StandardScaler`.

2. **Modelo baseline — Regresión Logística:**
   - Entrenamiento y evaluación en Dev y Test (Accuracy, Precisión, Recall,
     F1, matriz de confusión, curva ROC, curva de aprendizaje).
   - Análisis de los coeficientes para identificar las variables más
     influyentes.
   - **Hallazgo clave:** el dataset está fuertemente desbalanceado y el
     baseline lograba solo ~34% de recall sobre la clase minoritaria
     (clientes que sí contratan), lo que en un contexto de negocio real
     implica perder a la mayoría de los clientes potenciales.

3. **Red Neuronal (TensorFlow / Keras):**
   - Arquitectura densa (128 → 64 → 32 → 1) con activación ReLU, Dropout
     (0.3) y regularización L2 en cada capa oculta.
   - `EarlyStopping` y `ReduceLROnPlateau` para evitar sobreajuste y
     mejorar la convergencia.
   - **Ponderación de clases (`class_weight`)** para compensar el
     desbalance, calculada en función de la proporción real de cada clase.
   - Métrica de seguimiento: Recall (en lugar de Accuracy), por ser la más
     relevante para el objetivo de negocio.
   - Varias iteraciones de arquitectura documentadas en el notebook
     (número de capas, neuronas por capa, tasa de aprendizaje).

## Resultados

| Modelo                          | Recall (clase minoritaria) |
|----------------------------------|---------------------------:|
| Regresión Logística (baseline)   | 0.34                        |
| **Red Neuronal (class weighting)** | **0.89**                  |

La red neuronal con ponderación de clases mejoró drásticamente el recall
(34% → 89%), priorizando conscientemente la detección de clientes
potenciales por sobre la exactitud global (accuracy general del 84%,
con precisión de la clase positiva del 41%).

## Herramientas

Python · TensorFlow / Keras · Scikit-learn · Category Encoders · Pandas ·
Matplotlib · Seaborn

## Próximos pasos

- Explorar técnicas adicionales de balanceo (SMOTE) como alternativa/
  complemento al `class_weight`.
- Ajustar el umbral de decisión (threshold) según el costo de negocio de
  falsos positivos vs. falsos negativos.
