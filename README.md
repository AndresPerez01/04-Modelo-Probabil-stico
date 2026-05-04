# 04-BM25: Motor de Búsqueda con Modelo Probabilístico

Autor: Pérez Pineda Andrés Alejandro

## Problema del Ejercicio

TF-IDF tiene una debilidad estructural: no controla cuánto "premia" la repetición de un término ni ajusta por la longitud del documento más allá de la normalización del coseno. BM25 nace del **Modelo de Independencia Binaria (BIM)** y lo extiende con dos correcciones clave:

1. **Saturación de frecuencia de término:** La repetición de una palabra en un libro produce rendimientos decrecientes. Que "adventure" aparezca 500 veces no es 500 veces más relevante que si aparece 10 veces.
2. **Normalización explícita por longitud:** Un libro largo repite más palabras por razones estadísticas, no necesariamente porque sea más relevante. BM25 penaliza ese efecto con un parámetro dedicado.



## Arquitectura y Flujo del Código

### Paso 0: Carga del Corpus

El script verifica si los 1,000 libros de Project Gutenberg ya existen localmente. Si no, los descarga automáticamente con `requests`, guardando cada libro como `libro_N.txt`. Una vez disponibles, se cargan completos a memoria:

```python
with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
    texto = f.read()   # Lee el libro COMPLETO, no solo los primeros N caracteres
    corpus_gutenberg.append(texto)
```

> **Nota:** Leer los libros completos con `f.read()` aumenta la fidelidad del análisis, pero también incrementa la magnitud $||\vec{d}||$ de cada documento, lo cual diluye los puntajes de similitud coseno en TF-IDF. En BM25, este efecto queda controlado por el parámetro `b`.


### Paso 1: Construcción de la Matriz TF-IDF (Línea Base)

Se construye la matriz TF-IDF usando `TfidfVectorizer` de scikit-learn como punto de referencia:

```python
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
matriz_tfidf = vectorizer_tfidf.fit_transform(corpus_gutenberg)
```

Esto produce una matriz de **1,000 × 10,000** donde cada celda contiene el peso $w_{t,d}$ calculado internamente como:

$$w_{t,d} = tf_{t,d} \cdot \log\left(\frac{N}{df_t}\right)$$

La importación de `TfidfVectorizer` ya normaliza los vectores resultantes (norma L2), lo que equivale a dividir por $||\vec{d}||$ al calcular la similitud coseno.


### Paso 2: Ranking con TF-IDF y Similitud del Coseno

```python
vector_consulta = vectorizer_tfidf.transform([consulta])
similitudes_tfidf = cosine_similarity(vector_consulta, matriz_tfidf).flatten()
```

Se usa `transform()` (no `fit_transform()`) para proyectar la consulta al espacio vectorial ya establecido por los 1,000 libros. El ranking se basa en:

$$sim(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||}$$

El puntaje resultante está acotado entre **0 y 1**. Valores típicos sobre libros completos rondan **0.09–0.33**, ya que $||\vec{d}||$ es grande cuando el documento tiene cientos de miles de palabras.


### Paso 3: Implementación del Modelo Probabilístico BM25

A diferencia de TF-IDF, BM25 no se implementa con `fit_transform`. Se construye paso a paso desde sus fundamentos matemáticos.

#### 3.1 Tokenización con Frecuencias Puras

```python
vectorizer_tf = CountVectorizer(stop_words='english', max_features=10000)
matriz_tf = vectorizer_tf.fit_transform(corpus_gutenberg).toarray()
```

Se usa `CountVectorizer` en lugar de `TfidfVectorizer` porque BM25 necesita los conteos de frecuencia **crudos** ($tf_{t,d}$), no los pesos normalizados. A partir de aquí, el modelo construye su propia ponderación.

#### 3.2 Longitud de Documentos y Longitud Promedio

```python
longitudes_doc = matriz_tf.sum(axis=1)   # |d| para cada documento
avgdl = longitudes_doc.mean()            # avgdl del corpus
```

Estos dos valores son el corazón de la normalización por longitud de BM25. La relación $\frac{|d|}{avgdl}$ indica si un libro es más largo o más corto que el promedio de la colección.

#### 3.3 IDF con Corrección RSJ (Robertson-Sparck Jones)

El IDF estándar $\log(N/df_t)$ puede volverse negativo para términos muy frecuentes. BM25 aplica una variante robusta conocida como **corrección RSJ de 0.5**:

$$IDF(t) = \log\left(\frac{N - df_t + 0.5}{df_t + 0.5}\right)$$

donde $N$ es el número total de documentos y $df_t$ es el número de documentos que contienen el término $t$. Los valores del numerador y denominador se desplazan en 0.5 para:

- Evitar divisiones por cero cuando $df_t = 0$ o $df_t = N$.
- Suavizar los extremos estadísticos, especialmente en colecciones pequeñas.

En el código, se aplica un **piso en cero** para términos que aparecen en la mayoría de los documentos:

```python
idf = np.log((N - df + 0.5) / (df + 0.5))
idf[idf < 0] = 0  # Elimina penalizaciones por términos ultra-frecuentes
```

#### 3.4 Parámetros de Control: `k1` y `b`

```python
k1 = 1.5   # Controla la saturación de frecuencia de término
b  = 0.75  # Controla la normalización por longitud del documento
```

Estos son los dos parámetros que distinguen a BM25 del BIM original:

| Parámetro | Rango típico | Efecto |
|---|---|---|
| `k1` | [1.2, 2.0] | Si `k1 → 0`, el modelo se vuelve binario (solo presencia/ausencia). Si `k1` es grande, se aproxima al TF crudo sin saturación. |
| `b` | [0, 1] | Si `b = 0`, no hay normalización por longitud. Si `b = 1`, normalización máxima. |

Los valores usados (`k1 = 1.5`, `b = 0.75`) son los rangos típicos recomendados en la literatura (Manning et al., 2008).

#### 3.5 Fórmula Estándar BM25 (Okapi BM25)

La función de scoring combina todos los componentes anteriores:

$$BM25(d, q) = \sum_{t \in q} IDF(t) \cdot \frac{(k_1 + 1) \cdot tf_{t,d}}{k_1 \left((1 - b) + b \cdot \frac{|d|}{avgdl}\right) + tf_{t,d}}$$

Desglose de cada componente en el código:

```python
numerador   = (k1 + 1) * tf_td                                    # Saturación: crece pero se aplana
denominador = k1 * ((1 - b) + b * (longitudes_doc / avgdl)) + tf_td  # Penalización por longitud
score_termino = idf_t * (numerador / denominador)
scores += score_termino
```

- **El numerador** garantiza que el score crece con $tf_{t,d}$, pero con rendimientos decrecientes: duplicar la frecuencia no duplica el score.
- **El denominador** incorpora el factor de longitud. Si el libro es más largo que el promedio ($|d| > avgdl$), el denominador aumenta y el score disminuye.
- **El IDF** pondera cada término: los términos raros (bajo $df_t$) aportan más que los comunes.

> **Diferencia clave con TF-IDF:** El score BM25 **no está acotado entre 0 y 1**. Puede ser cualquier número positivo. Un puntaje de `7.99` en BM25 no es comparable directamente con `0.32` en TF-IDF — son escalas distintas.

### Paso 4: Comparación Visual

Para graficar ambos modelos juntos, el código normaliza cada escala a su propio máximo:

```python
scores_t_norm = [s / max(scores_t_top) for s in scores_t_top]  # TF-IDF: 0–1 → 0–1
scores_b_norm = [s / max(scores_b_top) for s in scores_b_top]  # BM25: 0–1.9085 → 0–1
```

Este paso es **necesario para la visualización** pero no modifica los rankings. La barra que llega al 100% en cada modelo representa simplemente al documento mejor rankeado por ese modelo. Lo que el gráfico revela es que ambos modelos tienen "opiniones" diferentes sobre qué documentos son relevantes, precisamente porque aplican lógicas de penalización distintas.
