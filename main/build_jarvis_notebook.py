import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
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
# JARVIS - Analitica de Spotify con IA

**Curso:** Modelos de Computacion  
**Enfoque:** AED + Clasificacion + NLP + OpenAI  
**Dataset:** spotify_2015_2025_85k.csv

---

Este cuaderno integra los temas solicitados: analisis exploratorio de datos (AED), modelos de clasificacion, modelos de lenguaje de procesamiento natural (NLP) e IA generativa con OpenAI.
"""
    )
)

cells.append(
    md(
        """
## Mapa de aplicacion de temas

1. **AED**: secciones 5, 6, 7 y 8 (calidad de datos, limpieza, visualizacion e interpretacion).  
2. **Modelos de clasificacion**: seccion 9 (clasificacion tabular de genero musical).  
3. **Modelos de lenguaje NLP**: seccion 10 (TF-IDF + Regresion Logistica).  
4. **Modelos de IA OpenAI**: seccion 11 (resumen ejecutivo y Q&A sobre hallazgos).
"""
    )
)

cells.append(md("## 1. Preparacion del entorno (Google Colab y local)"))
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
from matplotlib.ticker import FuncFormatter

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

# Configuracion visual basada en buenas practicas de matplotlib
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

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

cells.append(md("## 4. Carga robusta del dataset"))
cells.append(
    code(
        """
rutas_candidatas = [
    Path('/content/spotify_2015_2025_85k.csv'),
    Path('/content/data/spotify_2015_2025_85k.csv'),
    Path('/content/drive/MyDrive/spotify_2015_2025_85k.csv'),
    Path('/content/drive/MyDrive/data/spotify_2015_2025_85k.csv'),
    Path('spotify_2015_2025_85k.csv'),
    Path('data/spotify_2015_2025_85k.csv'),
    Path('../data/spotify_2015_2025_85k.csv'),
]

ruta_csv = next((p for p in rutas_candidatas if p.exists()), None)
if ruta_csv is None:
    raise FileNotFoundError('No se encontro spotify_2015_2025_85k.csv. Sube el archivo a Colab o ajusta rutas_candidatas.')

df = pd.read_csv(ruta_csv)
print(f'CSV cargado desde: {ruta_csv}')
print(f'Dimensiones iniciales: {df.shape}')
df.head()
"""
    )
)

cells.append(md("## 5. AED - Exploracion inicial y calidad de datos"))
cells.append(
    md(
        """
**Concepto tecnico:** *AED (Analisis Exploratorio de Datos)* es el proceso de revisar estructura, tipos, nulos y patrones antes de modelar.  
**Objetivo:** entender la calidad del dataset para evitar errores y decisiones sesgadas.
"""
    )
)
cells.append(
    code(
        """
print('Columnas del dataset:')
print(df.columns.tolist())

print('\\nTipos de datos:')
print(df.dtypes)

print('\\nResumen de memoria (deep):')
df.info(memory_usage='deep')
"""
    )
)
cells.append(
    code(
        """
nulos = df.isna().sum().sort_values(ascending=False)
nulos_pct = (nulos / len(df) * 100).round(3)
perfil_nulos = pd.DataFrame({'nulos': nulos, 'porcentaje': nulos_pct})
perfil_nulos.head(20)
"""
    )
)

cells.append(md("## 6. Limpieza y transformacion"))
cells.append(
    md(
        """
**Concepto tecnico:** *preprocesamiento* significa preparar datos para analisis/modelado (tipos correctos, variables coherentes, duplicados).  
**Objetivo:** garantizar que cada variable tenga formato util para graficas y modelos.
"""
    )
)
cells.append(
    code(
        """
df_limpio = df.copy()

# Normalizacion de texto en columnas relevantes
for col in ['genre', 'country', 'label', 'artist_name', 'track_name', 'album_name']:
    if col in df_limpio.columns:
        df_limpio[col] = df_limpio[col].astype('string').str.strip()

# Traduccion de generos al espanol para facilitar exposicion
mapa_genero_es = {
    'Pop': 'Pop',
    'Rock': 'Rock',
    'Hip-Hop': 'Hip-Hop',
    'R&B': 'R y B',
    'EDM': 'Electronica (EDM)',
    'Jazz': 'Jazz',
    'Classical': 'Clasica',
    'Country': 'Country',
    'Indie': 'Indie',
    'Metal': 'Metal',
    'Folk': 'Folk',
    'Reggaeton': 'Regueton',
}
df_limpio['genero_es'] = df_limpio['genre'].map(mapa_genero_es).fillna(df_limpio['genre'])

# Variable explicita traducida para lectura verbal
df_limpio['explicita_es'] = df_limpio['explicit'].astype(int).map({0: 'No explicita', 1: 'Explicita'})

# Conversion de fecha a formato temporal para analisis de tendencia
df_limpio['release_date'] = pd.to_datetime(df_limpio['release_date'], errors='coerce')
df_limpio['release_year'] = df_limpio['release_date'].dt.year
df_limpio['release_month'] = df_limpio['release_date'].dt.month

# Eliminar duplicados por id de pista
antes = len(df_limpio)
df_limpio = df_limpio.drop_duplicates(subset=['track_id']).reset_index(drop=True)
print(f'Duplicados eliminados por track_id: {antes - len(df_limpio)}')

# Conversion a categoricas para eficiencia y coherencia semantica
for col in ['genre', 'genero_es', 'country', 'label', 'explicita_es', 'explicit', 'mode', 'key']:
    if col in df_limpio.columns:
        df_limpio[col] = df_limpio[col].astype('category')

print('Dimensiones despues de limpieza:', df_limpio.shape)
df_limpio.head()
"""
    )
)

cells.append(md("## 7. AED - Resumen general"))
cells.append(
    code(
        """
resumen = {
    'registros': int(df_limpio.shape[0]),
    'columnas': int(df_limpio.shape[1]),
    'generos_unicos': int(df_limpio['genero_es'].nunique()),
    'paises_unicos': int(df_limpio['country'].nunique()),
    'labels_unicos': int(df_limpio['label'].nunique()),
    'popularidad_promedio': float(df_limpio['popularity'].mean()),
    'streams_promedio': float(df_limpio['stream_count'].mean()),
}
resumen
"""
    )
)

cells.append(md("## 8. AED - Visualizaciones e interpretacion"))

cells.append(
    code(
        """
# 8.1 Participacion por genero (barras horizontales)
conteo_genero = df_limpio['genero_es'].value_counts().sort_values(ascending=True)
total_canciones = conteo_genero.sum()
porc_genero = (conteo_genero / total_canciones * 100).round(1)

fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
palette = sns.color_palette('Blues', n_colors=len(conteo_genero))
bars = ax.barh(conteo_genero.index, conteo_genero.values, color=palette)

ax.set_title('Participacion de canciones por genero')
ax.set_xlabel('Cantidad de canciones')
ax.set_ylabel('Genero musical')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(',', '.')))
ax.set_xlim(0, conteo_genero.max() * 1.18)
ax.grid(axis='x', alpha=0.25)

for bar, valor, porc in zip(bars, conteo_genero.values, porc_genero.values):
    etiqueta = f"{valor:,}".replace(',', '.') + f" ({porc}%)"
    ax.text(valor + total_canciones * 0.0018, bar.get_y() + bar.get_height() / 2, etiqueta, va='center', fontsize=8)

plt.show()
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Grafica 8.1**  
- **Que muestra:** porcentaje y volumen de canciones por genero.  
- **Como leerla:** una barra mas larga indica mayor presencia del genero en la muestra.  
- **Conclusion clave:** permite identificar si el dataset esta balanceado o no por genero.
"""
    )
)

cells.append(
    code(
        """
# 8.2 Evolucion anual de lanzamientos
serie_anual = df_limpio.groupby('release_year').size().sort_index()
variacion_anual = (serie_anual.pct_change() * 100).round(2)
promedio_anual = serie_anual.mean()
anio_max = int(serie_anual.idxmax())
anio_min = int(serie_anual.idxmin())

fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
ax.plot(serie_anual.index, serie_anual.values, marker='o', linewidth=2.2, color='tab:green', label='Canciones por anio')
ax.axhline(promedio_anual, color='tab:orange', linestyle='--', linewidth=1.5, label=f'Promedio anual: {promedio_anual:,.0f}'.replace(',', '.'))
ax.set_title('Evolucion anual de lanzamientos (2015-2025)')
ax.set_xlabel('Anio')
ax.set_ylabel('Cantidad de canciones')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(',', '.')))
ax.grid(True, alpha=0.3)
ax.legend(loc='best')

ax.annotate(
    f"Maximo: {serie_anual[anio_max]:,}".replace(',', '.') + f" ({anio_max})",
    xy=(anio_max, serie_anual[anio_max]),
    xytext=(anio_max - 1.2, serie_anual[anio_max] + 80),
    arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1},
    fontsize=9,
)
ax.annotate(
    f"Minimo: {serie_anual[anio_min]:,}".replace(',', '.') + f" ({anio_min})",
    xy=(anio_min, serie_anual[anio_min]),
    xytext=(anio_min + 0.3, serie_anual[anio_min] - 90),
    arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1},
    fontsize=9,
)

display(
    pd.DataFrame({
        'anio': serie_anual.index,
        'canciones': serie_anual.values,
        'variacion_%_vs_anio_anterior': variacion_anual.values,
    }).tail(6)
)

plt.show()
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Grafica 8.2**  
- **Que muestra:** tendencia temporal de volumen de lanzamientos.  
- **Como leerla:** picos y valles reflejan anos con mayor/menor produccion relativa.  
- **Conclusion clave:** se observan fluctuaciones anuales, no una tendencia lineal fuerte.
"""
    )
)

cells.append(
    code(
        """
# 8.3 Popularidad por genero (boxplot ordenado por mediana)
top10 = df_limpio['genero_es'].value_counts().head(10).index
df_top10 = df_limpio[df_limpio['genero_es'].isin(top10)].copy()
orden_mediana = (
    df_top10.groupby('genero_es')['popularity']
    .median()
    .sort_values(ascending=False)
    .index
)

fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
sns.boxplot(
    data=df_top10,
    x='genero_es',
    y='popularity',
    order=orden_mediana,
    ax=ax,
    palette='Set2',
    showfliers=False,
)
ax.set_title('Distribucion de popularidad por genero (top 10 por volumen)')
ax.set_xlabel('Genero')
ax.set_ylabel('Popularidad')
ax.tick_params(axis='x', rotation=30)
plt.show()

resumen_pop = (
    df_top10.groupby('genero_es')['popularity']
    .agg(['median', 'mean', 'std', 'min', 'max'])
    .rename(columns={
        'median': 'mediana',
        'mean': 'promedio',
        'std': 'desv_estandar',
        'min': 'minimo',
        'max': 'maximo',
    })
    .round(2)
    .sort_values(by='mediana', ascending=False)
)
display(resumen_pop)
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Grafica 8.3**  
- **Que muestra:** distribucion de popularidad por genero usando medianas y dispersion.  
- **Como leerla:** la linea central de cada caja es la mediana; cajas mas altas indican mas variabilidad.  
- **Conclusion clave:** algunos generos concentran mayor popularidad central y otros tienen resultados mas dispersos.
"""
    )
)

cells.append(
    code(
        """
# 8.4 Bailabilidad vs energia (densidad + tendencia)
sample_scatter = df_limpio.sample(n=min(12000, len(df_limpio)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
hb = ax.hexbin(
    sample_scatter['danceability'],
    sample_scatter['energy'],
    cmap='viridis',
    gridsize=35,
    mincnt=1,
)
ax.set_title('Bailabilidad vs energia (densidad de canciones)')
ax.set_xlabel('Bailabilidad')
ax.set_ylabel('Energia')
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label('Cantidad de canciones')

# Tendencia lineal para apoyar interpretacion verbal
x = sample_scatter['danceability'].to_numpy()
y = sample_scatter['energy'].to_numpy()
coef = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = coef[0] * x_line + coef[1]
ax.plot(x_line, y_line, color='white', linewidth=2.0, label='Tendencia lineal')

corr_xy = np.corrcoef(x, y)[0, 1]
ax.text(
    0.02,
    0.97,
    f"Correlacion: {corr_xy:.2f}",
    transform=ax.transAxes,
    va='top',
    fontsize=10,
    bbox={'facecolor': 'white', 'alpha': 0.75, 'edgecolor': 'none'},
)
ax.legend(loc='lower right')
plt.show()
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Grafica 8.4**  
- **Que muestra:** concentracion de canciones en combinaciones de bailabilidad y energia.  
- **Como leerla:** tonos mas intensos indican mayor densidad de canciones en esa zona.  
- **Conclusion clave:** la relacion global entre ambas variables es interpretable con la linea de tendencia y la correlacion.
"""
    )
)

cells.append(
    code(
        """
# 8.5 Matriz de correlacion (triangular, anotada y en espanol)
cols_num = [
    'duration_ms', 'popularity', 'danceability', 'energy', 'key', 'loudness',
    'mode', 'instrumentalness', 'tempo', 'stream_count', 'explicit'
]
num_df = df_limpio[cols_num].copy()
num_df['explicit'] = num_df['explicit'].astype(int)
num_df['mode'] = num_df['mode'].astype(int)

corr = num_df.corr(numeric_only=True)
traduccion_vars = {
    'duration_ms': 'duracion_ms',
    'popularity': 'popularidad',
    'danceability': 'bailabilidad',
    'energy': 'energia',
    'key': 'tonalidad',
    'loudness': 'sonoridad',
    'mode': 'modo',
    'instrumentalness': 'instrumentalidad',
    'tempo': 'tempo',
    'stream_count': 'reproducciones',
    'explicit': 'explicita',
}
corr_es = corr.rename(index=traduccion_vars, columns=traduccion_vars)
mask = np.triu(np.ones_like(corr_es, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
sns.heatmap(
    corr_es,
    mask=mask,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt='.2f',
    linewidths=0.4,
    cbar_kws={'label': 'Correlacion'},
    ax=ax,
)
ax.set_title('Matriz de correlacion - Spotify (triangular)')
plt.show()

pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(key=lambda s: s.abs(), ascending=False)
    .head(6)
)
tabla_corr = pairs.reset_index()
tabla_corr.columns = ['variable_1', 'variable_2', 'correlacion']
tabla_corr['variable_1'] = tabla_corr['variable_1'].map(traduccion_vars)
tabla_corr['variable_2'] = tabla_corr['variable_2'].map(traduccion_vars)
tabla_corr['correlacion'] = tabla_corr['correlacion'].round(3)

print('Top 6 correlaciones (valor absoluto):')
display(tabla_corr)
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Grafica 8.5**  
- **Que muestra:** intensidad y direccion de relacion entre variables numericas.  
- **Como leerla:** valores cercanos a 1 o -1 indican relacion fuerte; cercanos a 0, relacion debil.  
- **Conclusion clave:** la tabla inferior resume las relaciones mas relevantes para argumentar resultados.
"""
    )
)

cells.append(md("## 9. Modelos de clasificacion (tema requerido)"))
cells.append(
    md(
        """
**Concepto tecnico:** *clasificacion supervisada* es predecir una categoria objetivo a partir de variables de entrada.  
**Objetivo del modelo:** predecir el **genero musical** (`genero_es`).
"""
    )
)
cells.append(
    code(
        """
target = 'genero_es'

df_modelo = df_limpio.copy()
df_modelo = df_modelo[df_modelo[target].notna()].copy()

# Filtrado minimo de clases para estabilidad estadistica
conteo_gen = df_modelo[target].astype(str).value_counts()
generos_validos = conteo_gen[conteo_gen >= 30].index
df_modelo = df_modelo[df_modelo[target].astype(str).isin(generos_validos)].copy()

# Variables temporales auxiliares
df_modelo['release_year'] = pd.to_datetime(df_modelo['release_date'], errors='coerce').dt.year
df_modelo['release_month'] = pd.to_datetime(df_modelo['release_date'], errors='coerce').dt.month

cols_excluir = {'track_id', 'track_name', 'artist_name', 'album_name', 'release_date', 'genre', 'genero_es'}
X = df_modelo[[c for c in df_modelo.columns if c not in cols_excluir]].copy()
y = df_modelo[target].astype(str)

num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Concepto tecnico: train/test split separa entrenamiento y prueba para medir generalizacion real
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocesamiento por tipo de dato
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
    ('clf', LogisticRegression(max_iter=1200)),
])

modelo_rf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
])

print(f'Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}')
print('Generos incluidos:', sorted(y.unique().tolist()))
"""
    )
)

cells.append(
    code(
        """
# Entrenamiento y comparacion
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
print('\\nReporte de clasificacion (tabular):')
print(classification_report(y_test, pred_mejor))
"""
    )
)

cells.append(
    code(
        """
# Matriz de confusion normalizada (porcentaje)
labels = sorted(y.unique())
cm_norm = confusion_matrix(y_test, pred_mejor, labels=labels, normalize='true')

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
sns.heatmap(
    cm_norm,
    cmap='Blues',
    ax=ax,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt='.1%',
    cbar_kws={'label': 'Proporcion por genero real'},
)
ax.set_title(f'Matriz de confusion normalizada - {mejor_nombre}')
ax.set_xlabel('Prediccion')
ax.set_ylabel('Real')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
plt.show()
"""
    )
)

cells.append(
    code(
        """
# Matriz de confusion en valores absolutos (conteos)
cm_abs = confusion_matrix(y_test, pred_mejor, labels=labels)

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
sns.heatmap(
    cm_abs,
    cmap='YlGnBu',
    ax=ax,
    xticklabels=labels,
    yticklabels=labels,
    annot=True,
    fmt='d',
    cbar_kws={'label': 'Cantidad de observaciones'},
)
ax.set_title(f'Matriz de confusion absoluta - {mejor_nombre}')
ax.set_xlabel('Prediccion')
ax.set_ylabel('Real')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)
plt.show()
"""
    )
)

cells.append(
    md(
        """
**Lectura rapida - Matrices de confusion**  
- **Normalizada (%):** indica que proporcion de cada genero real fue clasificado correctamente o confundido.  
- **Absoluta (conteos):** muestra volumen real de errores/aciertos por clase.  
- **Uso en exposicion:** primero explica la normalizada (comparacion justa), luego respalda con conteos absolutos.
"""
    )
)

cells.append(md("## 10. Modelo de lenguaje NLP (tema requerido)"))
cells.append(
    md(
        """
**Concepto tecnico:** *NLP* (Natural Language Processing) permite modelar texto.  
**Enfoque aplicado:** convertir texto a vectores con **TF-IDF** y clasificar genero con Regresion Logistica.
"""
    )
)
cells.append(
    code(
        """
df_nlp = df_limpio.copy()
df_nlp = df_nlp[df_nlp['genero_es'].notna()].copy()

texto_cols = ['track_name', 'artist_name', 'album_name', 'label', 'country']
texto_cols = [c for c in texto_cols if c in df_nlp.columns]

# Construccion de texto consolidado por pista
df_nlp['texto'] = (
    df_nlp[texto_cols]
    .astype('string')
    .fillna('')
    .agg(' '.join, axis=1)
    .str.replace(r'\\s+', ' ', regex=True)
    .str.strip()
)

# Filtrado de clases con muy pocas observaciones
conteo_nlp = df_nlp['genero_es'].astype(str).value_counts()
generos_nlp_validos = conteo_nlp[conteo_nlp >= 30].index
df_nlp = df_nlp[df_nlp['genero_es'].astype(str).isin(generos_nlp_validos)].copy()

X_text = df_nlp['texto']
y_text = df_nlp['genero_es'].astype(str)

Xtr, Xte, ytr, yte = train_test_split(X_text, y_text, test_size=0.2, random_state=42, stratify=y_text)

modelo_nlp = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=70000)),
    ('clf', LogisticRegression(max_iter=1500)),
])

modelo_nlp.fit(Xtr, ytr)
pred_nlp = modelo_nlp.predict(Xte)

print('Accuracy NLP:', round(accuracy_score(yte, pred_nlp), 4))
print('F1 macro NLP:', round(f1_score(yte, pred_nlp, average='macro'), 4))
print('\\nReporte NLP:')
print(classification_report(yte, pred_nlp))
"""
    )
)

cells.append(md("## 11. Modelos de IA OpenAI (tema requerido)"))
cells.append(
    md(
        """
**Concepto tecnico:** un modelo generativo puede sintetizar hallazgos y responder preguntas sobre resultados analiticos.  
**Enfoque aplicado:** resumen ejecutivo del AED/modelos + pregunta puntual de interpretacion.
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
contexto = {
    'registros': int(df_limpio.shape[0]),
    'generos_top': df_limpio['genero_es'].astype(str).value_counts().head(8).to_dict(),
    'paises_top': df_limpio['country'].astype(str).value_counts().head(8).to_dict(),
    'labels_top': df_limpio['label'].astype(str).value_counts().head(8).to_dict(),
    'modelos_tabulares': resumen_modelos,
}

if not api_key:
    print('No se detecto OPENAI_API_KEY. Se omite bloque OpenAI.')
else:
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=4)

    prompt_resumen = f'''
Eres un analista senior de musica y datos.
Genera un resumen ejecutivo en espanol (maximo 12 lineas) a partir de este contexto:
{contexto}
Incluye: calidad de datos, tendencias por genero y conclusiones de modelado.
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
        print('Resumen OpenAI:\\n')
        print(resp.choices[0].message.content)
    except openai.AuthenticationError as e:
        print(f'Error de autenticacion OpenAI: {e}')
    except openai.RateLimitError as e:
        print(f'Rate limit OpenAI: {e}')
    except openai.APIConnectionError as e:
        print(f'Error de conexion OpenAI: {e}')
    except openai.APIStatusError as e:
        print(f'Error de API OpenAI ({e.status_code}): {e}')

    pregunta = 'Que genero parece mas diferenciable y por que?' 
    try:
        resp_qa = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'Eres un asistente de analitica musical.'},
                {'role': 'user', 'content': f'Contexto: {contexto}\\nPregunta: {pregunta}'},
            ],
            temperature=0.2,
        )
        print('\\nRespuesta Q y A OpenAI:\\n')
        print(resp_qa.choices[0].message.content)
    except Exception as e:
        print(f'No se pudo completar Q y A OpenAI: {e}')
"""
    )
)

cells.append(md("## 12. Glosario tecnico (nivel intermedio)"))
cells.append(
    md(
        """
- **AED:** analisis exploratorio para entender forma y calidad de los datos.  
- **Variable objetivo (target):** columna que el modelo intenta predecir.  
- **Train/Test Split:** separacion entre datos para entrenar y datos para evaluar.  
- **Pipeline:** cadena de pasos automatizados (preprocesamiento + modelo).  
- **One-Hot Encoding:** transforma categorias en columnas numericas binarias.  
- **F1 Macro:** promedio del F1 por clase, util cuando hay varias clases.  
- **Matriz de confusion:** tabla de aciertos y errores por clase real/predicha.  
- **TF-IDF:** pondera palabras por importancia relativa dentro del corpus.  
- **N-grama:** secuencia de n palabras contiguas (ej. bigrama = 2 palabras).  
- **Temperatura (LLM):** controla aleatoriedad de la respuesta generativa.
"""
    )
)

cells.append(md("## 13. Guion breve para exposicion"))
cells.append(
    md(
        """
1. Primero explico calidad del dataset (nulos, tipos y volumen).  
2. Luego muestro distribucion por genero y evolucion temporal para contexto.  
3. Presento relaciones entre variables (bailabilidad, energia, popularidad).  
4. Paso a clasificacion tabular y comparo modelos con F1 macro.  
5. Interpreto matrices de confusion (porcentaje y conteo).  
6. Finalmente muestro NLP y apoyo con resumen/Q&A de OpenAI.
"""
    )
)

cells.append(
    md(
        """
## 14. Conclusiones

- Se aplicaron los cuatro temas solicitados de forma trazable y explicable.
- Las graficas fueron redisenadas para lectura verbal clara y argumentacion academica.
- La clasificacion tabular y NLP permiten comparar enfoques sobre datos estructurados y texto.
- OpenAI aporta capa interpretativa para sintetizar resultados tecnicos en lenguaje natural.
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
