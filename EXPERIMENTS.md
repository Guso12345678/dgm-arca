# EXPERIMENTS.md

Cuaderno de experimentos del proyecto (dominio: analista de riesgo de crédito para préstamos al consumo). Las entradas van de la más antigua a la más reciente.

---

### 2026-10-XX · Fase 1 · Elección de hiperparámetros del SFT

**Qué queríamos saber.** Con qué configuración entrenar el SFT por destilación para que el modelo aprenda el formato `<think>`/`<answer>` en nuestros cuatro tipos de problema (cuota, TAE, coste total y ratio) sin gastar más GPU de la necesaria.

**Qué hicimos.** Fijamos estos valores antes de lanzar el entrenamiento, y los iremos ajustando si algo no encaja:
- `--model Qwen/Qwen3-0.6B` (el valor por defecto). Es el alumno, distinto del profesor Qwen3-4B de `distill.py`. Tiene capacidad suficiente para aritmética sencilla con ayuda de `<think>` y es barato de entrenar y evaluar varias veces.
- `--lora-rank 8`. La tarea es estrecha (cuatro tipos de problema, un solo dominio), así que no hace falta mucha capacidad nueva; r=8 suele bastar y entrena más rápido. Si vemos underfitting en GRPO subimos a 16. No tiene sentido ir a 32 o más con tan pocos datos.
- `--epochs 3`. Con un dataset pequeño (esperábamos entre 300 y 700 trazas tras el filtro de `distill`, no las 510×4 iniciales), 2 épocas se quedaban cortas para fijar bien el formato; 3 es razonable sin pasarse hacia el sobreajuste.
- `--learning-rate 2e-4`, el valor estándar para LoRA.
- `--max-length 1024`. Las trazas son problemas numéricos con razonamiento corto (una fórmula y 2 a 4 pasos), no demostraciones largas, así que 1024 debería bastar.
- `--batch-size 8 --grad-accum 2`, que ajustamos según la memoria libre en cada momento.

**Qué pasó.** El SFT dejó un pass@1 de 0.278 sobre el conjunto de evaluación (25 de 90): coste total 15/19, ratio 10/27, cuota 0/23 y TAE 0/21. Los fallos son casi todos truncamientos: el modelo se queda sin presupuesto de tokens antes de cerrar `<answer>`. Según nuestras notas, el profesor tampoco consiguió trazas válidas para TAE (0 de 32) ni para cuota (0 de 508) por la misma razón, lo que explica por qué el alumno no las aprende.

**Qué concluimos.** El formato `<think>`/`<answer>` se aprende, pero el alumno no sabe resolver los tipos de problema cuyas trazas son largas. Esto es lo que debería arreglar GRPO con recompensas verificables; si falta capacidad, probaremos con rango 16.

---

### 2026-10-XX · Fase 1 · Preparación del GRPO: recompensas y falta de memoria

**Qué queríamos saber.** Cómo montar GRPO sobre el adaptador del SFT para que entre en los 16 GB de GPU que tenemos y dentro del límite de 8 horas por sesión de la DGX.

**Qué hicimos.** Usamos `rlm.train_grpo` partiendo de `rlm/weights/sft_lora`, con tres recompensas y pesos `[1.0, 2.0, 0.5]`. Con el tamaño de lote inicial nos quedamos sin memoria, así que añadimos micro-lotes (`--micro-batch`) y acumulamos gradiente (`--grad-accum 2`).

**Qué pasó.** Con micro-lotes el entrenamiento cabe. Cada paso tarda unos 152 s, así que 60 pasos son unas 2 h y media, y guardamos un checkpoint cada 20 para no perder el trabajo si se corta la sesión.

**Qué concluimos.** El diseño es viable en nuestra GPU. Los resultados del entrenamiento real (curvas de recompensa y pass@1 por tipo frente al SFT) irán en una entrada nueva cuando termine.

---

### 2026-10-XX · Fase 2 · Banco de primera llamada y ablaciones del prompt

**Qué queríamos saber.** Si Qwen3-1.7B elige bien la herramienta (`get_euribor`, `loan_metrics`, `save_credit_report`) y sus argumentos en la primera llamada, y qué parte del prompt hace el trabajo.

**Qué hicimos.** Un banco de 33 casos que mide dos cosas: acierto al elegir la herramienta y acierto en los argumentos. Probamos el prompt inicial (baseline), dos iteraciones posteriores (v2 y v3) y, sobre v3, tres ablaciones: sin ejemplos (`sin_fewshot`), sin reglas (`sin_reglas`) y con modo de pensamiento (`thinking`).

**Qué pasó.**

| Variante | Selección | Argumentos |
|---|---|---|
| baseline | 0.939 | 0.909 |
| v2 | 1.000 | 0.970 |
| v3 sin_fewshot | 1.000 | 1.000 |
| v3 sin_reglas | 0.970 | 0.970 |
| v3 thinking | 1.000 | 0.939 |

**Qué concluimos.** Un 1.0 en este banco hay que leerlo con cuidado: es el conjunto que usamos para iterar el prompt, así que es de desarrollo y sobreestima lo que haría con casos nuevos. Entre variantes las diferencias son de uno o dos casos sobre 33, dentro del ruido. Quitar los ejemplos no empeoró nada, y quitar las reglas lo empeoró muy poco.

---

### 2026-10-XX · Fase 2 · Con y sin herramientas sobre casos nuevos

**Qué queríamos saber.** Cuánto mejora el modelo con herramientas frente a sin ellas, sobre casos que no usamos para iterar.

**Qué hicimos.** `uv run python -m tool_use.compare --per-type 10`: 40 casos nuevos (10 por tipo) resueltos una vez sin herramientas y otra con ellas.

**Qué pasó.** Sin herramientas acierta el 10 % (4 de 40); con herramientas, el 68 % (27 de 40). Por tipo, sin y con herramientas: cuota 0 → 1.00, coste total 0.40 → 0.80, ratio 0 → 0.50, TAE 0 → 0.40. Cuando el modelo llama a la herramienta acierta siempre (27 de 27). Los 13 fallos tienen en común que no emite ninguna llamada: anuncia lo que va a hacer o razona en el texto, pero la lista de llamadas sale vacía. Un caso que vimos por separado con `curl` a `/tools`: omitió el argumento opcional `upfront_fee` y devolvió una TAE de 10.36 % en lugar de la correcta, 12.60 % (coste total 631.76).

**Qué concluimos.** Las herramientas son lo que hace que un modelo de este tamaño pueda calcular estas métricas, y el 0.68 es la cifra honesta porque sale de casos nuevos. El problema de la fase 2 no es elegir mal la herramienta ni pasar mal los argumentos, sino que a veces ni siquiera la llama, y que omite argumentos opcionales. Lo dejamos como está: afinar el modelo con SFT para tool use y construir un banco de cadenas de herramientas queda para el proyecto final.

---

### 2026-11-01 · Fase 3 · Corpus, gold set y distancias entre frases

**Qué queríamos saber.** Con qué textos trabajamos, cómo medimos si el retriever encuentra lo que toca, y si vale un umbral fijo de distancia para el chunking semántico con Qwen3-Embedding-0.6B.

**Qué hicimos.** El corpus son la Ley 16/2011 de crédito al consumo (110k caracteres) y la Directiva 2023/2225 (249k), bajados con `rag.fetch_corpus`. La guía de la EBA la dejamos fuera porque había que bajarla a mano, y el material del Banco de España venía con basura de la web pegada al texto. Con dos textos legales ya salen unos 500 fragmentos, nos pareció suficiente.

El gold set son 50 preguntas con una cita literal del corpus como evidencia (`rag/gold/gold.jsonl`). Las preguntas las sacamos con ayuda de un modelo, a partir de borradores y de búsquedas por palabras clave en el corpus, y luego las repasamos; la cita se copia del texto, así que casa siempre. `--check-gold` confirma que las 50 se encuentran con las cuatro estrategias de chunking. Para las distancias entre frases usamos `rag.plot_distances` (gráficas en `reports/distances_ley.png` y `reports/distances_directiva.png`).

**Qué pasó.** Con 400 frases por documento, la distancia media es 0.596 en la Ley y 0.497 en la Directiva. Un umbral de 0.6 corta el 51 % de las fronteras en la Ley pero solo el 29 % en la Directiva; con 0.7 es el 36 % y el 14 %. Los percentiles sí se comportan igual en los dos (p70 corta el 30 %, p80 el 20 %), aunque valen cosas distintas: 0.729 y 0.777 en la Ley, 0.587 y 0.657 en la Directiva. Las curvas son ruidosas y no tienen mesetas claras.

**Qué concluimos.** Con estos embeddings las distancias son altas y cambian de un documento a otro. Los umbrales típicos de 0.2 a 0.4 cortaban entre el 73 % y el 98 % de las fronteras, o sea casi cada frase. Nos quedamos con un umbral fijo de 0.7 y con los percentiles 70 y 80 para comparar. Sobre el gold set: lo generó un modelo y solo lo repasamos nosotros, las preguntas se parecen bastante al texto (solape de palabras de ~0.55), no hay ninguna que necesite dos fragmentos, y alguna no se entiende sin contexto ("¿Y si al contrato le queda un año o menos…?"). Hay que tenerlo en cuenta al leer los resultados.

---

### 2026-11-01 · Fase 3 · Chunking y retrievers (BM25, denso, híbrido)

**Qué queríamos saber.** Qué chunking y qué retriever dan mejor Recall@5 y MRR, y qué λ usar en el híbrido.

**Qué hicimos.** `uv run python -m rag.evaluate --gold rag/gold/gold.jsonl --k 5 --calibrate rag/gold/unanswerable.jsonl --generation` (log en `rag_eval.log`, resultados en `reports/phase3_eval.json`). Cuatro chunkings: recursivo de 800 caracteres con solape de 100, semántico con umbral 0.7, y semántico con percentil 70 y 80. Embeddings Qwen3-Embedding-0.6B con instrucción en la consulta. El híbrido mezcla las puntuaciones de BM25 y del denso con min-max y un peso λ para BM25; probamos λ de 0 a 1. Las 50 preguntas casan en las cuatro estrategias, así que la comparación es justa (no se pierde ninguna).

**Qué pasó.**

| Chunking | Chunks | Long. media | BM25 | Denso | Híbrido (mejor λ) |
|---|---|---|---|---|---|
| recursive-800 | 543 | 690 | 0.65 / 0.42 | 0.53 / 0.43 | 0.65 / 0.50 (0.7) |
| semantic-0.7 | 498 | 719 | 0.68 / 0.55 | 0.66 / 0.50 | 0.74 / 0.63 (0.6) |
| semantic-p70 | 605 | 592 | 0.66 / 0.53 | 0.62 / 0.40 | 0.74 / 0.58 (0.5) |
| semantic-p80 | 496 | 722 | 0.66 / 0.55 | 0.59 / 0.47 | 0.72 / 0.61 (0.5) |

(Cada celda es Recall@5 / MRR.)

**Qué concluimos.** El híbrido es el mejor, o empata con el mejor, en los cuatro chunkings. BM25 le gana al denso, lo cual tiene sentido porque nuestras preguntas comparten muchas palabras con el texto; no creemos que BM25 sea mejor en general, solo en este gold set. El chunking semántico sale mejor (0.74 frente a 0.65), pero no nos fiamos mucho: con 50 preguntas cada una pesa 0.02 y el error ronda ±0.06, los chunks semánticos son algo más largos (hasta 1500 caracteres frente a 799), lo que ayuda a que contengan la evidencia, y λ se elige con las mismas 50 preguntas que luego evaluamos. Aun así, para la API nos quedamos con semantic-0.7, híbrido y λ = 0.6, que es el mejor por poco y el más simple de los semánticos.

---

### 2026-11-01 · Fase 3 · Generación con citas y rechazo

**Qué queríamos saber.** Si el generador cita fragmentos que existen, si responde cuando el corpus tiene la información y si se niega cuando no la tiene.

**Qué hicimos.** Lanzamos el mismo comando dos veces, una con `recursive-800` (el que sale por defecto) y otra con `--chunking semantic-0.7`. Generador Qwen3-1.7B, k = 5, retriever híbrido. El umbral de similitud para rechazar se calibró con 8 preguntas que el corpus no contesta (`rag/gold/unanswerable.jsonl`). No tocamos el prompt entre una ejecución y otra; lo dejamos para el proyecto final. Para entender los fallos usamos un script pequeño (`diag_gen.py`) sobre `reports/phase3_eval.json`.

**Qué pasó.**

| | recursive-800 | semantic-0.7 |
|---|---|---|
| Sin respuesta en el corpus: se abstiene | 8/8 | 8/8 |
| Citas que no existen | 0 % | 0 % |
| Responde con alguna cita | 64 % | 70 % |
| Cita el fragmento de la evidencia | 32 % | 32 % |
| Se abstiene teniendo respuesta | 38 % | 32 % |
| Umbral: mínimo de las respondibles / máximo de las no respondibles | 0.585 / 0.577 | 0.570 / 0.568 |

Con recursive-800, de las 50 preguntas con respuesta: 22 las contesta con el fragmento correcto recuperado, 9 las contesta sin él, 11 se abstiene teniendo el fragmento correcto y 8 se abstiene sin tenerlo. Algunos ejemplos que miramos a mano:
- A "¿Hasta cuándo valen las cifras del cuadro de amortización?" da la respuesta correcta y luego añade "No hay información suficiente…", así que se contradice.
- A "¿De cuánto tiempo dispone un cliente para echarse atrás?" responde bien (14 días naturales) pero cita un fragmento vecino y no el de la evidencia, así que la métrica lo cuenta como fallo aunque la respuesta es correcta.
- A "¿Qué conceptos forman parte del coste total del crédito?" da una lista que no es la correcta (TAE, tipo deudor…) con citas que existen pero no respaldan lo que dice.
- En "¿Qué deben establecer los Estados…?" confunde el apartado 4 con el artículo 4.

**Qué concluimos.** Que no invente citas (0 %) no quiere decir que sea fiel al texto, el ejemplo del coste total lo deja claro. Para medir eso haría falta un juez o comprobar la evidencia, y no lo hemos hecho. El modelo es demasiado cauto: se niega en un tercio de las preguntas con respuesta, y más de la mitad de esas veces tenía el fragmento correcto delante. Además la métrica de "cita el fragmento relevante" se queda corta porque solo da por bueno un fragmento por pregunta. El margen del umbral de rechazo es mínimo (0.003, con solo 8 preguntas negativas), así que lo pusimos en 0.569, justo entre los dos valores, sin pensar que sea robusto.

Configuración que dejamos en la API: `ARCA_RAG_CHUNKING=semantic`, `ARCA_RAG_SEMANTIC_THRESHOLD=0.7`, `ARCA_RAG_LAMBDA=0.6`, `ARCA_RAG_MIN_SIMILARITY=0.569`. Para el proyecto final queda: ajustar el prompt de abstención, probar un reranker, añadir un juez de fidelidad, preguntas que necesiten dos fragmentos y un gold set más grande escrito a mano.

---

### 2026-11-01 · Fase 3 · Prueba del endpoint /rag

**Qué queríamos saber.** Si la API responde con la configuración final, con los tres retrievers y con una pregunta que el corpus no contesta.

**Qué hicimos.** Arrancamos `uvicorn api.app:app --port 8011` con `ARCA_RAG_CHUNKING=semantic`, `ARCA_RAG_SEMANTIC_THRESHOLD=0.7`, `ARCA_RAG_LAMBDA=0.6` y `ARCA_RAG_MIN_SIMILARITY=0.569`, y lanzamos con `curl` "¿Cuántos días tiene el consumidor para desistir del contrato de crédito?" con `dense`, `bm25` y `hybrid`, y "¿Cuál es el tipo de interés oficial del Banco Central Europeo hoy?" con `hybrid`.

**Qué pasó.** Los tres retrievers contestan bien (14 días naturales) y citan fragmentos que están entre los devueltos: el artículo 28 de la Ley y el artículo 26 de la Directiva, o el formulario normalizado de información. El denso pone primero el artículo 26 de la Directiva (0.87) y después el 28 de la Ley (0.85); BM25 pone primero el formulario (21.9) y el artículo 26 cuarto. En el híbrido la respuesta termina en "…como [4fa8f08b1a-0225] y.": el modelo citó otro fragmento y la comprobación de citas lo quitó, pero dejó la "y" colgando. La pregunta del BCE se rechaza con "No hay información suficiente en los documentos para responder a esa pregunta." y la lista de citas viene vacía.

**Qué concluimos.** El endpoint funciona de punta a punta y el rechazo también. El fallo de la "y" suelta es cosmético: la limpieza de citas inválidas no arregla la frase. Lo dejamos como está y lo anotamos para el proyecto final.