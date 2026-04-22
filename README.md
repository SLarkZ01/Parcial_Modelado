# 🎧 Guia completa del notebook JARVIS (Spotify + IA)

Esta guia explica, en lenguaje sencillo, que hace el notebook `main/jarvis.ipynb`, por que se hace, como se aplica cada concepto y como interpretar resultados. Esta pensada para alguien que ya programa un poco, pero que esta empezando en analisis de datos, machine learning y NLP. 📘

> [!NOTE]
> En esta guia, cada bloque importante incluye **donde se aplica en codigo** para que puedas ubicarlo rapido.

---

## 🎯 1) Objetivo del notebook

El notebook busca responder preguntas sobre musica de Spotify con datos reales 🎵:

- Como se distribuyen los generos.
- Como cambia el volumen de canciones por anio.
- Si hay relacion entre caracteristicas musicales (por ejemplo, bailabilidad y energia).
- Si un modelo puede predecir el genero musical.
- Si se puede usar texto (NLP) para clasificar genero.
- Como OpenAI puede ayudar a resumir hallazgos y responder preguntas.

> [!TIP]
> Piensa este notebook como una cadena: **entender datos -> visualizar -> modelar -> interpretar**.

---

## ✅ 2) Temas requeridos y donde se aplican

1. **AED (Analisis Exploratorio de Datos) 🔎**
   - Carga, calidad, limpieza y visualizaciones.

2. **Modelos de clasificacion 🤖**
   - Clasificacion supervisada de `genero_es`.

3. **Modelos de lenguaje / NLP 🧠**
   - TF-IDF + Regresion Logistica sobre texto.

4. **Modelos de IA OpenAI ✨**
   - Resumen automatico y Q&A.

### 📌 Dónde se aplica en código

```python
# AED - nulos y calidad
nulos = df.isna().sum().sort_values(ascending=False)

# Clasificacion
target = 'genero_es'
modelo_logreg = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', LogisticRegression(max_iter=1200)),
])

# NLP
modelo_nlp = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=70000)),
    ('clf', LogisticRegression(max_iter=1500)),
])

# OpenAI
client = OpenAI(api_key=api_key, timeout=30.0, max_retries=4)
```

Referencia: `main/build_jarvis_notebook.py:180`, `main/build_jarvis_notebook.py:566`, `main/build_jarvis_notebook.py:609`, `main/build_jarvis_notebook.py:764`, `main/build_jarvis_notebook.py:791`

---

## 🧭 3) Explicacion del flujo completo

### 🛠️ Paso A: Preparar entorno

Se instalan librerias como `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `openai`.

**Por que:** cada libreria cumple una funcion distinta:

- `pandas`: tablas y transformaciones.
- `matplotlib/seaborn`: graficas.
- `scikit-learn`: entrenamiento y evaluacion de modelos.
- `openai`: capa de interpretacion en lenguaje natural.

```python
!pip -q install pandas numpy matplotlib seaborn scikit-learn openai python-dotenv
```

Referencia: `main/build_jarvis_notebook.py:58`

### 📂 Paso B: Carga robusta de datos

El notebook intenta varias rutas de archivo (local, Colab, Drive).

```python
rutas_candidatas = [
    Path('/content/spotify_2015_2025_85k.csv'),
    Path('data/spotify_2015_2025_85k.csv'),
    Path('../data/spotify_2015_2025_85k.csv'),
]

ruta_csv = next((p for p in rutas_candidatas if p.exists()), None)
df = pd.read_csv(ruta_csv)
```

Referencia: `main/build_jarvis_notebook.py:132`

### 🔍 Paso C: Inspeccion inicial (AED basico)

Se revisan:

- columnas,
- tipos,
- nulos,
- uso de memoria.

```python
print(df.columns.tolist())
print(df.dtypes)
df.info(memory_usage='deep')

nulos = df.isna().sum().sort_values(ascending=False)
```

Referencia: `main/build_jarvis_notebook.py:166`, `main/build_jarvis_notebook.py:180`

### 🧹 Paso D: Limpieza y transformacion

Se hace:

- normalizacion de texto,
- conversion de fechas,
- creacion de `release_year` y `release_month`,
- eliminacion de duplicados,
- traduccion de genero al espanol (`genero_es`).

```python
df_limpio = df.copy()
df_limpio['release_date'] = pd.to_datetime(df_limpio['release_date'], errors='coerce')
df_limpio['release_year'] = df_limpio['release_date'].dt.year
df_limpio = df_limpio.drop_duplicates(subset=['track_id']).reset_index(drop=True)
```

Referencia: `main/build_jarvis_notebook.py:200`

### 📊 Paso E: AED visual

Se construyen graficas para entender patrones, distribuciones y relaciones.

```python
conteo_genero = df_limpio['genero_es'].value_counts().sort_values(ascending=True)
serie_anual = df_limpio.groupby('release_year').size().sort_index()
hb = ax.hexbin(sample_scatter['danceability'], sample_scatter['energy'], gridsize=35)
corr = num_df.corr(numeric_only=True)
```

Referencia: `main/build_jarvis_notebook.py:272`, `main/build_jarvis_notebook.py:311`, `main/build_jarvis_notebook.py:431`, `main/build_jarvis_notebook.py:491`

### 🤖 Paso F: Clasificacion tabular

Se predice `genero_es` con modelos de clasificacion y se comparan metricas.

```python
target = 'genero_es'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
resultados_df = pd.DataFrame(resultados).sort_values(by='f1_macro', ascending=False)
```

Referencia: `main/build_jarvis_notebook.py:566`, `main/build_jarvis_notebook.py:588`, `main/build_jarvis_notebook.py:629`

### 🧠 Paso G: NLP clasico

Se crea texto por pista y se clasifica el genero usando `TF-IDF`.

```python
df_nlp['texto'] = (
    df_nlp[texto_cols]
    .astype('string')
    .fillna('')
    .agg(' '.join, axis=1)
)

modelo_nlp = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=70000)),
    ('clf', LogisticRegression(max_iter=1500)),
])
```

Referencia: `main/build_jarvis_notebook.py:737`, `main/build_jarvis_notebook.py:764`

### ✨ Paso H: OpenAI

Se genera resumen automatico y respuestas a preguntas sobre los hallazgos.

```python
dotenv_paths = [Path('.env'), Path('../.env')]
client = OpenAI(api_key=api_key, timeout=30.0, max_retries=4)
resp = client.chat.completions.create(model='gpt-4o-mini', messages=[...])
```

Referencia: `main/build_jarvis_notebook.py:791`

---

## ⚙️ 4) Que significa lo que pasa "por detras"

### 4.1 Train/Test split 🧪

Divide datos en dos partes:

- entrenamiento: el modelo aprende,
- prueba: el modelo se evalua.

Esto evita evaluar con los mismos datos que uso para memorizar.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Referencia: `main/build_jarvis_notebook.py:588`

### 4.2 Pipeline + ColumnTransformer 🔗

Encadena pasos de preprocesamiento y modelo en una sola estructura reproducible.

```python
preprocess = ColumnTransformer(
    transformers=[
        ('num', pre_num, num_cols),
        ('cat', pre_cat, cat_cols),
    ]
)

modelo_rf = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)),
])
```

Referencia: `main/build_jarvis_notebook.py:602`, `main/build_jarvis_notebook.py:614`

### 4.3 F1 macro 📏

Metrica de clasificacion que promedia por clase y no favorece solo clases grandes.

```python
f1m = f1_score(y_test, pred, average='macro')
```

Referencia: `main/build_jarvis_notebook.py:635`

### 4.4 TF-IDF (NLP) 📝

Convierte texto a numeros ponderando palabras importantes:

- alta frecuencia en un documento,
- baja frecuencia global en el corpus.

```python
('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=70000))
```

Referencia: `main/build_jarvis_notebook.py:764`

---

## 🧾 5) Campos del dataset explicados (en lenguaje claro)

Como pediste especificamente estos dos, aqui van primero con detalle:

### 5.1 `danceability` (bailabilidad) 💃

- Rango: generalmente de **0.0 a 1.0**.
- Significado: que tan "bailable" es una cancion.
- No mide solo ritmo: combina elementos como pulso estable, tempo, fuerza de beat y regularidad.

**Interpretacion practica 👇:**

- **cerca de 0**: menos apta para baile.
- **cerca de 1**: mas apta para baile.

### 5.2 `energy` (energia) ⚡

- Rango: generalmente de **0.0 a 1.0**.
- Significado: intensidad percibida de la cancion.
- Se relaciona con fuerza sonora y actividad musical.

**Interpretacion practica 👇:**

- **cerca de 0**: cancion mas calmada/suave.
- **cerca de 1**: cancion mas intensa/potente.

### ❗ Importante: `danceability` y `energy` no son lo mismo

Una cancion puede:

- ser muy bailable y poco energetica,
- o ser muy energetica y no tan bailable.

```python
hb = ax.hexbin(
    sample_scatter['danceability'],
    sample_scatter['energy'],
    cmap='viridis',
    gridsize=35,
    mincnt=1,
)
```

Referencia: `main/build_jarvis_notebook.py:431`

> [!IMPORTANT]
> Si la correlacion entre `danceability` y `energy` sale cercana a 0, significa que **no hay relacion lineal fuerte**, no que "no tengan nada que ver" en todos los casos.

---

## 📈 6) Como leer las graficas clave

## 6.1 Distribucion de popularidad por genero (boxplot) 📦

**Que muestra:** como se reparte la popularidad dentro de cada genero.

**Como leer:**

- linea central de la caja: mediana,
- caja: rango intercuartil (25%-75%),
- bigotes: extension tipica.

```python
sns.boxplot(
    data=df_top10,
    x='genero_es',
    y='popularity',
    order=orden_mediana,
    showfliers=False,
)
```

Referencia: `main/build_jarvis_notebook.py:380`

## 6.2 Bailabilidad vs energia (hexbin) 🐝

**Que muestra:** densidad de canciones en combinaciones de bailabilidad (X) y energia (Y).

**Como leer:**

- color mas intenso = mas canciones en esa zona,
- linea de tendencia = direccion promedio,
- texto de correlacion = fuerza de relacion lineal.

```python
coef = np.polyfit(x, y, 1)
ax.plot(x_line, y_line, color='white', linewidth=2.0, label='Tendencia lineal')
corr_xy = np.corrcoef(x, y)[0, 1]
```

Referencia: `main/build_jarvis_notebook.py:445`

## 6.3 Matriz de correlacion (triangular) 🔗

**Que muestra:** relacion lineal entre pares de variables numericas.

**Escala:**

- +1: relacion positiva fuerte,
- 0: casi sin relacion,
- -1: relacion negativa fuerte.

```python
corr = num_df.corr(numeric_only=True)
sns.heatmap(corr_es, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1, annot=True)
```

Referencia: `main/build_jarvis_notebook.py:491`

## 6.4 Matriz de confusion normalizada (%) 🧭

**Que muestra:** porcentaje de aciertos/errores por clase real.

```python
cm_norm = confusion_matrix(y_test, pred_mejor, labels=labels, normalize='true')
```

Referencia: `main/build_jarvis_notebook.py:664`

## 6.5 Matriz de confusion absoluta (conteos) 🔢

**Que muestra:** cantidad real de ejemplos por celda (aciertos/errores).

```python
cm_abs = confusion_matrix(y_test, pred_mejor, labels=labels)
```

Referencia: `main/build_jarvis_notebook.py:691`

> [!TIP]
> En exposicion: primero explica la matriz normalizada (comparacion justa por clase), luego la absoluta (impacto en volumen real).

---

## 🗣️ 7) Como argumentar resultados en exposicion

Orden sugerido para explicar con claridad:

1. Calidad y preparacion de datos.
2. Distribuciones base (generos y tiempo).
3. Relaciones entre variables (`danceability`, `energy`, `popularidad`, `reproducciones`).
4. Modelos de clasificacion y metricas.
5. Matrices de confusion (primero %, luego conteos).
6. NLP como enfoque alternativo desde texto.
7. OpenAI como capa de sintesis e interpretacion.

> [!NOTE]
> Esta secuencia te da una narrativa academica: **contexto -> evidencia -> modelado -> interpretacion -> conclusion**.

---

## 📚 8) Mini glosario tecnico

- **AED:** exploracion inicial para entender el dataset.
- **Target:** variable que queremos predecir.
- **Overfitting:** cuando el modelo memoriza entrenamiento y falla en datos nuevos.
- **Validacion cruzada:** evaluar varias particiones para estimar robustez.
- **TF-IDF:** representacion numerica de texto basada en importancia de terminos.
- **NLP:** tecnicas para analizar lenguaje humano con computadoras.

---

## 🧩 9) Conclusiones de la guia

Este notebook no solo ejecuta codigo: construye una narrativa analitica completa que va desde limpieza y AED, hasta modelado clasico, NLP y apoyo con IA generativa. Las graficas fueron diseniadas para que sean explicables verbalmente, comparables y utiles para interpretar decisiones de analisis.

> [!SUCCESS]
> Si sigues esta guia en orden, puedes explicar de forma clara tanto el **que** se hizo como el **por que** y el **donde** en codigo.
