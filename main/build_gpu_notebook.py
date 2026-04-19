import json
from pathlib import Path


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


cells = []

cells.append(
    md(
        """
# Proyecto de analitica de GPUs con IA

**Curso:** Modelos de Computacion  
**Enfoque:** AED + Clasificacion + NLP + OpenAI  
**Dataset:** all-gpus.csv

---

Este cuaderno reemplaza el enfoque anterior y se centra en el dataset de GPUs. Incluye analisis exploratorio de datos (AED), modelos de clasificacion para predecir `vendor`, un modelo NLP sobre descripciones tecnicas y un bloque de IA con OpenAI para resumen y preguntas en lenguaje natural.
"""
    )
)

cells.append(
    md(
        """
## 1. Preparacion del entorno (Google Colab y local)

Esta celda instala las dependencias necesarias para ejecutar todo el flujo.
"""
    )
)

cells.append(
    code(
        """
!pip -q install pandas numpy matplotlib seaborn scikit-learn openai python-dotenv
"""
    )
)

cells.append(md("## 2. Importaciones y configuracion"))

cells.append(
    code(
        """
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
from openai import OpenAI
import openai

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)
pd.set_option("display.max_columns", 120)

print("Entorno listo")
"""
    )
)

cells.append(
    md(
        """
## 3. (Opcional) Montar Google Drive

Descomenta esta celda si quieres leer el CSV desde tu Drive en Colab.
"""
    )
)

cells.append(
    code(
        """
# from google.colab import drive
# drive.mount('/content/drive')
"""
    )
)

cells.append(
    md(
        """
## 4. Carga robusta del dataset

Se buscan varias rutas compatibles con Colab y entorno local.
"""
    )
)

cells.append(
    code(
        """
rutas_candidatas = [
    Path('/content/all-gpus.csv'),
    Path('/content/data/all-gpus.csv'),
    Path('/content/drive/MyDrive/all-gpus.csv'),
    Path('/content/drive/MyDrive/data/all-gpus.csv'),
    Path('all-gpus.csv'),
    Path('data/all-gpus.csv'),
    Path('../data/all-gpus.csv'),
]

ruta_csv = next((p for p in rutas_candidatas if p.exists()), None)
if ruta_csv is None:
    raise FileNotFoundError('No se encontro all-gpus.csv. Sube el archivo a Colab o ajusta rutas_candidatas.')

df = pd.read_csv(ruta_csv)
print(f'CSV cargado desde: {ruta_csv}')
print(f'Dimensiones iniciales: {df.shape}')
df.head()
"""
    )
)

cells.append(md("## 5. Exploracion inicial y calidad de datos"))

cells.append(
    code(
        """
print('Primeras columnas:')
print(df.columns.tolist())

print('\nTipos de datos:')
print(df.dtypes)

print('\nResumen de memoria (deep):')
df.info(memory_usage='deep')
"""
    )
)

cells.append(
    code(
        """
nulos = df.isna().sum().sort_values(ascending=False)
nulos_pct = (nulos / len(df) * 100).round(2)
perfil_nulos = pd.DataFrame({'nulos': nulos, 'porcentaje': nulos_pct})
perfil_nulos.head(20)
"""
    )
)

cells.append(
    md(
        """
## 6. Limpieza y transformacion

Objetivos:
- normalizar columnas clave,
- eliminar duplicados por `id`,
- convertir fechas y tipos,
- preparar un dataframe limpio para analisis y modelado.
"""
    )
)

cells.append(
    code(
        """
df_limpio = df.copy()

if 'vendor' in df_limpio.columns:
    df_limpio['vendor'] = df_limpio['vendor'].astype(str).str.strip().str.lower()

if 'releaseDate' in df_limpio.columns:
    df_limpio['releaseDate'] = pd.to_datetime(df_limpio['releaseDate'], errors='coerce')

if 'id' in df_limpio.columns:
    antes = len(df_limpio)
    df_limpio = df_limpio.drop_duplicates(subset=['id']).reset_index(drop=True)
    print(f'Duplicados eliminados por id: {antes - len(df_limpio)}')

columnas_categoria = ['vendor', 'manufacturer', 'architecture', 'generation', 'memoryType', 'busInterface', 'slot']
for col in columnas_categoria:
    if col in df_limpio.columns:
        df_limpio[col] = df_limpio[col].astype('category')

print('Dimensiones despues de limpieza:', df_limpio.shape)
df_limpio.head()
"""
    )
)

cells.append(md("## 7. AED (Analisis Exploratorio de Datos)"))

cells.append(
    code(
        """
print('Cantidad de vendors:', df_limpio['vendor'].nunique())
display(df_limpio['vendor'].value_counts().head(15))

if 'architecture' in df_limpio.columns:
    print('\nTop arquitecturas:')
    display(df_limpio['architecture'].value_counts().head(15))

if 'generation' in df_limpio.columns:
    print('\nTop generaciones:')
    display(df_limpio['generation'].value_counts().head(15))
"""
    )
)

cells.append(
    code(
        """
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

df_limpio['vendor'].value_counts().head(10).plot(kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Top vendors')
axes[0, 0].set_ylabel('Cantidad')

if 'tdp' in df_limpio.columns:
    sns.histplot(df_limpio['tdp'], bins=30, kde=True, ax=axes[0, 1], color='tomato')
    axes[0, 1].set_title('Distribucion de TDP')

if 'memorySize' in df_limpio.columns:
    sns.histplot(df_limpio['memorySize'], bins=30, kde=True, ax=axes[1, 0], color='seagreen')
    axes[1, 0].set_title('Distribucion de memorySize')

if 'boostClock' in df_limpio.columns:
    sns.histplot(df_limpio['boostClock'], bins=30, kde=True, ax=axes[1, 1], color='goldenrod')
    axes[1, 1].set_title('Distribucion de boostClock')

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """
num_df = df_limpio.select_dtypes(include=['number']).copy()
if not num_df.empty:
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Matriz de correlacion (variables numericas)')
    plt.show()
else:
    print('No hay columnas numericas suficientes para correlacion')
"""
    )
)

cells.append(
    md(
        """
## 8. Modelo de clasificacion tabular (objetivo: vendor)

Se comparan dos modelos con buenas practicas de preprocesamiento en pipeline:
- LogisticRegression (baseline),
- RandomForestClassifier.
"""
    )
)

cells.append(
    code(
        """
target = 'vendor'
columnas_excluir = {'id', 'name', 'url', target}

df_modelo = df_limpio.copy()
df_modelo = df_modelo[df_modelo[target].notna()].copy()

X = df_modelo[[c for c in df_modelo.columns if c not in columnas_excluir]].copy()
y = df_modelo[target].astype(str)

conteo_vendor = y.value_counts()
vendors_validos = conteo_vendor[conteo_vendor >= 5].index
mask = y.isin(vendors_validos)
X = X[mask].copy()
y = y[mask].copy()

if 'releaseDate' in X.columns:
    X['releaseDate'] = pd.to_datetime(X['releaseDate'], errors='coerce').map(lambda x: x.timestamp() if pd.notna(x) else np.nan)

num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pre_num = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

pre_cat = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', pre_num, num_cols),
        ('cat', pre_cat, cat_cols),
    ]
)

modelo_logreg = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', LogisticRegression(max_iter=1500)),
])

modelo_rf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])

print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
print('Clases incluidas:', sorted(y.unique().tolist()))
"""
    )
)

cells.append(
    code(
        """
resultados = []

for nombre, modelo in [('LogisticRegression', modelo_logreg), ('RandomForest', modelo_rf)]:
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average='macro')
    cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1).mean()
    resultados.append({'modelo': nombre, 'accuracy': acc, 'f1_macro': f1m, 'cv_f1_macro': cv})

resultados_df = pd.DataFrame(resultados).sort_values(by='f1_macro', ascending=False)
resultados_df
"""
    )
)

cells.append(
    code(
        """
mejor_nombre = resultados_df.iloc[0]['modelo']
mejor_modelo = modelo_logreg if mejor_nombre == 'LogisticRegression' else modelo_rf

pred_mejor = mejor_modelo.predict(X_test)
print(f'Mejor modelo: {mejor_nombre}')
print('\nClassification report:')
print(classification_report(y_test, pred_mejor))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, pred_mejor, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Matriz de confusion - {mejor_nombre}')
plt.xlabel('Prediccion')
plt.ylabel('Real')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    md(
        """
## 9. Modelo NLP clasico (TF-IDF + LogisticRegression)

Se construye una descripcion de texto por GPU y se entrena un clasificador de lenguaje para predecir `vendor`.
"""
    )
)

cells.append(
    code(
        """
columnas_texto = ['name', 'gpuName', 'architecture', 'generation', 'memoryType', 'busInterface', 'manufacturer']
columnas_texto = [c for c in columnas_texto if c in df_limpio.columns]

df_nlp = df_limpio[df_limpio['vendor'].notna()].copy()
if not columnas_texto:
    raise ValueError('No hay columnas de texto disponibles para el modelo NLP.')

df_nlp['texto_gpu'] = (
    df_nlp[columnas_texto]
    .astype('string')
    .fillna('')
    .agg(' '.join, axis=1)
    .str.replace(r'\\s+', ' ', regex=True)
    .str.strip()
)

conteo_nlp = df_nlp['vendor'].astype(str).value_counts()
vendors_nlp_validos = conteo_nlp[conteo_nlp >= 5].index
df_nlp = df_nlp[df_nlp['vendor'].astype(str).isin(vendors_nlp_validos)].copy()

X_text = df_nlp['texto_gpu']
y_text = df_nlp['vendor'].astype(str)

Xtr, Xte, ytr, yte = train_test_split(X_text, y_text, test_size=0.2, random_state=42, stratify=y_text)

modelo_nlp = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
    ('clf', LogisticRegression(max_iter=1500)),
])

modelo_nlp.fit(Xtr, ytr)
pred_nlp = modelo_nlp.predict(Xte)

print('Accuracy NLP:', round(accuracy_score(yte, pred_nlp), 4))
print('F1 macro NLP:', round(f1_score(yte, pred_nlp, average='macro'), 4))
print('\nReporte NLP:')
print(classification_report(yte, pred_nlp))
"""
    )
)

cells.append(
    md(
        """
## 10. OpenAI: resumen tecnico del AED y Q&A

Este bloque usa un modelo de OpenAI para:
- resumir hallazgos del AED,
- responder preguntas en lenguaje natural sobre el dataset y modelos.

Si no hay `OPENAI_API_KEY`, el bloque se omite sin romper el notebook.
"""
    )
)

cells.append(
    code(
        """
dotenv_paths = [
    Path('.env'),
    Path('../.env'),
    Path('/content/.env'),
    Path('/content/drive/MyDrive/.env'),
]
for env_path in dotenv_paths:
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break

api_key = os.getenv('OPENAI_API_KEY')

resumen_modelos = resultados_df.to_dict(orient='records') if 'resultados_df' in globals() else []
contexto_eda = {
    'filas': int(df_limpio.shape[0]),
    'columnas': int(df_limpio.shape[1]),
    'vendors_top': df_limpio['vendor'].astype(str).value_counts().head(10).to_dict(),
    'nulos_top': perfil_nulos.head(10).to_dict(orient='index'),
    'modelos_tabulares': resumen_modelos,
}

if not api_key:
    print('No se detecto OPENAI_API_KEY. Se omite el bloque OpenAI.')
else:
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=4)

    prompt_resumen = f'''
Eres un analista de datos senior.
Genera un resumen ejecutivo en espanol (maximo 12 lineas) a partir de este contexto:
{contexto_eda}
Incluye: calidad de datos, comportamiento por vendor, y conclusiones de modelado.
'''

    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'Responde en espanol tecnico y claro.'},
                {'role': 'user', 'content': prompt_resumen},
            ],
            temperature=0.2,
        )
        resumen_openai = resp.choices[0].message.content
        print('Resumen OpenAI:\n')
        print(resumen_openai)
    except openai.AuthenticationError as e:
        print(f'Error de autenticacion OpenAI: {e}')
    except openai.RateLimitError as e:
        print(f'Rate limit OpenAI: {e}')
    except openai.APIConnectionError as e:
        print(f'Error de conexion OpenAI: {e}')
    except openai.APIStatusError as e:
        print(f'Error de API OpenAI ({e.status_code}): {e}')

    pregunta = 'Que vendor parece mas diferenciable segun los resultados y por que?'
    try:
        resp_qa = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'Eres un asistente de analitica de GPUs.'},
                {'role': 'user', 'content': f'Contexto: {contexto_eda}\nPregunta: {pregunta}'},
            ],
            temperature=0.2,
        )
        print('\nRespuesta Q y A OpenAI:\n')
        print(resp_qa.choices[0].message.content)
    except Exception as e:
        print(f'No se pudo completar Q y A OpenAI: {e}')
"""
    )
)

cells.append(
    md(
        """
## 11. Conclusiones

- Se realizo un AED completo del dataset de GPUs con control de calidad de datos.
- Se construyeron modelos de clasificacion tabular para predecir `vendor` con pipeline robusto.
- Se implemento un modelo NLP clasico (TF-IDF + Regresion Logistica) sobre texto tecnico de GPUs.
- Se integro OpenAI para resumen ejecutivo y Q&A en lenguaje natural.
- El cuaderno queda listo para ejecucion en Google Colab y tambien en entorno local.
"""
    )
)

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent / "jarvis.ipynb"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generado: {out_path}")
