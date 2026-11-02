# EXPERIMENTS.md

Cuaderno de experimentos del proyecto (dominio: analista de riesgo de crédito para préstamos al consumo). Las entradas van de la más antigua a la más reciente.

---

### 2026-10-23 · Fase 1 · Elección de hiperparámetros del SFT

**Qué queríamos saber.** Con qué configuración entrenar el SFT por destilación para que el modelo aprenda el formato `<think>`/`<answer>` en nuestros cuatro tipos de problema (cuota, TAE, coste total y ratio) sin gastar más GPU de la necesaria.

**Qué hicimos.** Fijamos estos valores antes de lanzar el entrenamiento, y los iremos ajustando si algo no encaja:
- `--model Qwen/Qwen3-0.6B` (el valor por defecto). Es el alumno, distinto del profesor Qwen3-4B de `distill.py`. Tiene capacidad suficiente para aritmética sencilla con ayuda de `<think>` y es barato de entrenar y evaluar varias veces.
- `--lora-rank 8`. La tarea es estrecha (cuatro tipos de problema, un solo dominio), así que no hace falta mucha capacidad nueva; r=8 suele bastar y entrena más rápido. Si vemos underfitting en GRPO subimos a 16. No tiene sentido ir a 32 o más con tan pocos datos.
- `--epochs 3`. Con un dataset pequeño (esperábamos entre 300 y 700 trazas tras el filtro de `distill`, no las 510×4 iniciales), 2 épocas se quedaban cortas para fijar bien el formato; 3 es razonable sin pasarse hacia el sobreajuste.
- `--learning-rate 2e-4`, el valor estándar para LoRA.
- `--max-length 1024`. Las trazas son problemas numéricos con razonamiento corto (una fórmula y 2 a 4 pasos), no demostraciones largas, así que 1024 debería bastar.
- `--batch-size 8 --grad-accum 2`, que ajustamos según la memoria libre en cada momento.

**Qué pasó.** El SFT dejó un pass@1 de 0.278 sobre el conjunto de evaluación (25 de 90): coste total 15/19, ratio 10/27, cuota 0/23 y TAE 0/21. Esta medición la hicimos antes de fijar el protocolo común de evaluación (ver la entrada de resultados de GRPO), con otra configuración de generación, así que no es comparable con las tablas posteriores. Los fallos son casi todos truncamientos: el modelo se queda sin presupuesto de tokens antes de cerrar `<answer>`. Según nuestras notas, el profesor tampoco consiguió trazas válidas para TAE (0 de 32) ni para cuota (0 de 508) por la misma razón, lo que explica por qué el alumno no las aprende.

**Qué concluimos.** El formato `<think>`/`<answer>` se aprende, pero el alumno no sabe resolver los tipos de problema cuyas trazas son largas. Esto es lo que debería arreglar GRPO con recompensas verificables; si falta capacidad, probaremos con rango 16.

---

### 2026-10-23 · Fase 1 · Preparación del GRPO: recompensas y falta de memoria

**Qué queríamos saber.** Cómo montar GRPO sobre el adaptador del SFT para que entre en los 16 GB de GPU que tenemos y dentro del límite de 8 horas por sesión de la DGX.

**Qué hicimos.** Usamos `rlm.train_grpo` partiendo de `rlm/weights/sft_lora`, con tres recompensas y pesos `[1.0, 2.0, 0.5]`. Con el tamaño de lote inicial nos quedamos sin memoria, así que añadimos micro-lotes (`--micro-batch`) y acumulamos gradiente (`--grad-accum 2`).

**Qué pasó.** Con micro-lotes el entrenamiento cabe. En las primeras pruebas cada paso tardaba unos 152 s; en la ejecución final salieron unos 102 s por paso, así que los 100 pasos que lanzamos (`--steps 100`, `--learning-rate 2e-5`, `--grad-accum 2`) llevaron 2 h 50 min. Guardamos un checkpoint cada 20 pasos para no perder el trabajo si se cortaba la sesión.

**Qué concluimos.** El diseño es viable en nuestra GPU. Los resultados del entrenamiento real (curvas de recompensa y pass@1 por tipo frente al SFT) irán en una entrada nueva cuando termine.

---

### 2026-10-23 · Fase 2 · Banco de primera llamada y ablaciones del prompt

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

### 2026-10-24 · Fase 2 · Con y sin herramientas sobre casos nuevos

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

---

### 2026-11-01 · Fase 1 · Resultados de GRPO: base, SFT y GRPO sobre el test

**Qué queríamos saber.** Si GRPO mejora al SFT en el conjunto de test (90 problemas que el modelo no ha visto, 19 de coste total, 23 de cuota, 21 de TAE y 27 de ratio de endeudamiento) y en qué tipos, y entender qué hizo cada recompensa durante el entrenamiento.

**Qué hicimos.** Evaluamos los tres modelos con el mismo protocolo: `rlm.evaluate` sobre `rlm/data/rlm_data_test.jsonl`, generación voraz y `--max-new-tokens 1024`. El modelo base es Qwen3-0.6B sin adaptador, el SFT es `rlm/weights/sft_lora` y el GRPO es `rlm/weights/grpo_lora`, el adaptador que sale de 100 pasos de GRPO partiendo del SFT. Las curvas salen del log del entrenamiento (`rlm/report_phase1.py curves`) y los fallos de `rlm/report_phase1.py failures`.

**Qué pasó.**

| Tipo | Base | SFT | GRPO |
|---|---|---|---|
| coste total (19) | 4 (0.211) | 13 (0.684) | 16 (0.842) |
| cuota (23) | 0 | 0 | 0 |
| ratio de endeudamiento (27) | 0 | 0 | 7 (0.259) |
| TAE (21) | 0 | 0 | 0 |
| **Total (90)** | **4 (0.044)** | **13 (0.144)** | **23 (0.256)** |
| Formato válido | 0.31 | 0.21 | 0.41 |
| Tokens medios generados | 929 | 986 | 904 |

Durante el entrenamiento (100 pasos, comparando la media de los diez primeros con la de los diez últimos, porque un solo paso es muy ruidoso) la recompensa total pasó de 2.30 a 2.65; el máximo posible con nuestros pesos es 3.5. El formato subió de 0.72 a 0.93 y la recompensa de dominio de 0.72 a 0.93, **pero la exactitud casi no se movió (0.61 a 0.63, pendiente prácticamente cero)**. La longitud media de las completions bajó de 1146 a 914 tokens y la desviación de la recompensa dentro de cada grupo de 1.37 a 1.07. En solo 2 de los 100 pasos (el 9 y el 41) las 8 generaciones del grupo sacaron la recompensa máxima y la desviación fue 0.

**Qué pasó con los fallos.** Miramos ocho fallos de GRPO, que cubren los cuatro tipos:
- Cinco están truncados a 1024 tokens (coste total, cuota, ratio y dos de TAE). El modelo razona bien al principio pero se enreda en dudas ("Wait, no...") o en cuentas largas a mano. En el de ratio de endeudamiento el planteamiento era correcto y la suma de deudas también (1708.02), y se quedó sin tokens haciendo la división entre 3722.57.
- En TAE el modelo no sabe qué calcula: se pregunta si la TAE es el propio tipo nominal o si hay que sumar la comisión, y no llega a ninguna cuenta. La TAE exige resolver una ecuación de forma iterativa, algo que no puede hacer a mano en 1024 tokens.
- En cuota, uno de los fallos confunde "sistema de amortización francés" con "German amortization" y el otro, sin truncar, se inventa la fórmula ("préstamo más comisión entre meses, más intereses") y da 370.43 en vez de 448.27. No recuerda la fórmula de la cuota constante.
- Dos son casi correctos y los da por malos la comprobación estricta a dos decimales. En el coste total (1852.33 frente a 1851.85) el planteamiento es el bueno, incluida la comisión de apertura, pero multiplica mal 7.01 por 48 (336.96 en vez de 336.48). En el ratio (22.76 frente a 22.78) es un error de la división en el cuarto dígito.

**Qué concluimos.**
- GRPO mejora al SFT en 10 problemas (de 13 a 23 aciertos de 90). La mejora sale de dos tipos: coste total sube de 13 a 16 y el ratio de endeudamiento pasa de 0 a 7. Con 90 problemas y una sola ejecución, la diferencia es clara pero no hay manera de dar un intervalo fino, y no repetimos con otras semillas.
- Cuota y TAE siguen en 0 de 23 y 0 de 21 con los tres modelos. Lo explican dos cosas. El profesor no consiguió trazas válidas de esos tipos (0 de 508 y 0 de 32), así que el SFT no tiene de dónde aprenderlos. Y no entraron en GRPO: los 254 problemas de entrenamiento son solo de coste total (131) y de ratio de endeudamiento (123). GRPO ha podido mejorar los dos tipos que entrenó, que son justo los que mejoran (coste total de 13 a 16 y ratio de 0 a 7). Aunque cuota y TAE hubieran entrado, con el SFT en 0 aciertos ningún grupo tendría respuestas correctas, y en GRPO eso da ventaja cero y ningún gradiente de exactitud: solo quedarían las recompensas de formato y de dominio, que enseñan a cerrar bien la respuesta pero no a calcular. GRPO afina lo que el modelo ya resuelve a veces, no descubre lo que no sabe hacer nunca (esto último es nuestra interpretación, no lo hemos probado con un experimento aparte).
- El límite de generación pesa mucho. El 59 % de las respuestas de GRPO en el test no tiene el formato válido y la mayor parte de los fallos que miramos son truncamientos, así que buena parte del problema es de presupuesto y no de conocimiento. En el entrenamiento el formato llega a 0.93 y en el test solo a 0.41. El test incluye cuota y TAE, que el modelo no resuelve y en los que se corta, mientras que los 254 problemas de entrenamiento de GRPO son solo de coste total y de ratio. Por eso no comparamos la exactitud de entrenamiento (0.63) con el pass@1 del test (0.256).
- El SFT queda por debajo del base en formato válido (0.21 frente a 0.31), pese a acertar más. Creemos que es porque aprendió trazas largas del profesor y las alarga hasta el tope de tokens; no lo hemos comprobado.
- La mejora que vimos durante el entrenamiento es de formato y de longitud, no de exactitud: la exactitud en entrenamiento se queda plana (0.61 a 0.63). Lo más probable es que GRPO haya enseñado sobre todo a cerrar `<think>` y `<answer>` dentro del presupuesto y a pensar más corto, y que eso explique las 10 respuestas más que acierta en el test, pero no lo hemos separado con un experimento. Además la recompensa de dominio es casi idéntica a la de formato (coinciden en 98 de los 100 pasos, con diferencia máxima de 0.03): premia que el `<think>` tenga al menos tres números distintos y que la respuesta tenga dos decimales y ningún símbolo, y eso casi siempre se cumple cuando el formato es válido, así que aporta muy poca señal propia.
- La caída de la desviación de la recompensa dentro del grupo (de 1.37 a 1.07) indica que la política se va concentrando y que cada vez hay menos diferencias que aprender. No vimos ningún caso de *reward hacking* en los fallos que revisamos, pero no hicimos una búsqueda dirigida.

**Qué decidimos.** Dejamos GRPO como adaptador final de la fase 1 (`ARCA_RLM_ADAPTER=rlm/weights/grpo_lora`). No relanzamos la evaluación con más tokens por falta de tiempo: queda para el proyecto final, junto con generar trazas del profesor para cuota y TAE con más presupuesto, probar GRPO con ejemplos de esos tipos que el modelo sí resuelva alguna vez y una recompensa que no penalice un error en el último dígito igual que un planteamiento equivocado.

---

### 2026-11-01 · Fase 4 · Diseño del bucle ReAct, la memoria y el banco de tareas

**Qué queríamos saber.** Cómo montar el agente para que sea robusto con modelos pequeños y cómo medirlo de forma reproducible, antes de lanzar ninguna evaluación.

**Qué hicimos.** El bucle está en `agent/react_agent.py`. Decisiones que tomamos y por qué:
- Una acción por paso. Si el modelo escribe varias llamadas en el mismo mensaje solo se ejecuta la primera y se le avisa. En la fase 2 vimos que el modelo pequeño escribe a la vez el Euribor y el préstamo que depende de él, e inventa el valor del Euribor.
- Todo lo que falla es una observación y no una excepción: JSON mal formado, argumentos inválidos, herramienta desconocida, herramienta que lanza un error, que tarda demasiado o que pide una confirmación que nadie dio. Cada herramienta tiene un tiempo máximo (20 s, y 300 s para la base de conocimiento porque la primera llamada puede cargar el índice).
- Si la misma llamada se repite, la segunda no se ejecuta (se recuerda el resultado) y la tercera termina la tarea como fallida, para no gastar todos los pasos.
- Si el modelo contesta con texto en vez de llamar a `final_answer`, lo aceptamos como respuesta cuando ya usó alguna herramienta, y marcamos el paso. Si todavía no había usado ninguna le recordamos una vez que debe usarlas, porque contestar un cálculo de memoria es justo lo que queremos evitar.
- Las herramientas con efectos (`save_credit_report`) pasan por una confirmación. Como `/agent` es una sola llamada sin interacción, la política por defecto es que la propia petición del usuario es la confirmación (igual que en la fase 2); con `ARCA_AGENT_CONFIRM=deny` se bloquean.
- Memoria (`agent/memory.py`): la tarea original queda fijada y nunca se resume, las observaciones largas se recortan antes de resumir, el corte nunca separa una observación de la llamada que la produjo, y el resumen se hace por extracción (qué herramienta, qué argumentos, qué volvió) sin llamar al modelo; hay una variante con el modelo (`ARCA_AGENT_SUMMARY=llm`) para compararlas. Solo se comprime cuando se supera el presupuesto de tokens.
- Los tres cerebros: `base` y `rlm` comparten un Qwen3-0.6B en memoria (el adaptador LoRA se apaga para `base`) y `thinking` es un Qwen3-1.7B con el modo de pensamiento activado. Se cargan la primera vez que se usan.
- `search_knowledge_base` usa los recuperadores de la fase 3 (híbrido por defecto) sin cargar el generador de RAG, porque la respuesta la escribe el propio cerebro del agente.

El banco (`agent/bench/tasks.jsonl`) tiene 23 tareas: 21 encadenan entre dos y tres herramientas (Euribor, base de conocimiento, `loan_metrics`, `save_credit_report`) y 2 son imposibles (el Euribor de 2030 y el saldo bancario de un cliente) y solo se aciertan si el agente lo dice. Las respuestas numéricas se comprueban con tolerancia de 0.02 y los esperados se recalculan en un test con la misma herramienta. Como el `get_euribor` real lee el BCE y cambia cada mes, para evaluar lo sustituimos por una tabla fija (`ARCA_AGENT_FIXTURE=bench`) con el mismo nombre y esquema. Además de la tasa de éxito medimos qué fracción de tareas usa de verdad las herramientas requeridas, para distinguir un acierto razonado de uno de memoria.

**Qué pasó.** Todavía no hemos evaluado ningún cerebro: la GPU estaba ocupada con el GRPO. Probamos el bucle con un cerebro simulado que reproduce respuestas guionizadas (no necesita GPU) y los tests cubren la cadena Euribor, préstamo y respuesta, la recuperación tras errores, el tiempo máximo, la detección de bucles, la confirmación, la compresión de memoria y la comparación con el agente de código de smolagents.

**Qué concluimos.** Dejamos la evaluación de los tres cerebros y la comparación entre agente JSON y agente de código para cuando termine el GRPO. Una cosa que esperamos ver: el cerebro `rlm` está entrenado sobre Qwen3-0.6B, que es más pequeño que el 1.7B de las fases 2 y 3, así que la comparación mezcla el efecto de entrenar con el del tamaño.

---

### 2026-11-01 · Fase 4 · Evaluación de los tres cerebros y corrección del bucle

**Qué queríamos saber.** Cuántas de las 23 tareas del banco resuelve el agente con cada cerebro (`base`, `rlm` y `thinking`), y si los fallos son del modelo o de nuestro bucle.

**Qué hicimos.** Lanzamos `agent.evaluate` con Euribor fijo (`ARCA_AGENT_FIXTURE=bench`), generación voraz, 10 pasos como máximo y el adaptador de GRPO (`ARCA_RLM_ADAPTER=rlm/weights/grpo_lora`). La primera ejecución (v1) dio resultados muy bajos incluso con el cerebro de 1.7B, que en la fase 2 elegía bien las herramientas, así que sacamos la salida cruda de dos tareas (`raw4.py`, un script de depuración que no está en el repositorio) y encontramos tres fallos nuestros:
- El modelo imita la transcripción ReAct y escribe también el `Observation:` con un valor inventado, y seguía con un `final_answer` basado en él. Ahora todo lo que va después de un `Observation:` escrito por el modelo se descarta.
- Escribía sumas dentro del JSON (`"annual_rate_pct": 2.5 + 3`). Esa llamada era inválida, y nuestro bucle saltaba a la siguiente llamada válida, que era el `final_answer` inventado. Ahora una suma simple se calcula, y si la primera llamada sigue mal formada no se ejecuta ninguna posterior; el modelo recibe el error.
- Escribía `Action: {json}` en vez de `<tool_call>`, y lo tratábamos como texto. Ahora se acepta ese formato.
Añadimos cinco tests con las salidas reales y relanzamos con el mismo banco (v2).

**Qué pasó.**

| Cerebro | Éxito v1 | Éxito v2 | Herramientas requeridas v1 → v2 | Pasos v2 | Tokens v2 |
|---|---|---|---|---|---|
| base (0.6B) | 0.09 (2/23) | 0.04 (1/23) | 0.09 → 0.09 | 7.3 | 17625 |
| rlm (0.6B + GRPO) | 0.09 (2/23) | 0.09 (2/23) | 0.13 → 0.13 | 5.6 | 13319 |
| thinking (1.7B) | 0.26 (6/23) | 0.57 (13/23) | 0.26 → 0.70 | 2.8 | 7506 |

Por tipo de tarea, `thinking` pasó de 3/8 a 6/8 en Euribor y préstamo, de 1/3 a 2/3 en base de conocimiento y guardar informe, de 1/3 a 2/3 en cálculo y guardar, y de 0/2 a 2/2 en la cadena de tres herramientas. En las tareas con base de conocimiento y cálculo se queda en 0/5 en las dos ejecuciones, y en las dos tareas imposibles saca 1/2.

**Qué miramos en los fallos de `thinking` (10 de 23).**
- Dos de Euribor y préstamo suman mal el diferencial: el Euribor a 1 mes valía 2.0 y para "más 5 puntos" escribió `annual_rate_pct: 2.05`; el de 3 meses valía 2.1 y con "más 6 puntos" escribió 2.16. Entendió "5 puntos" como 0.05.
- Las cinco tareas que mezclan la ley y un cálculo fallan cada una de una forma. Una busca con `top_k` igual a 1, recibe el apartado del 0.5 % (el de menos de un año) cuando la tarea pedía el de más de un año, y lo usa como comisión: es el recall@1 bajo de la fase 3 visto desde el agente. Dos buscan y contestan solo el dato legal sin llegar a llamar a `loan_metrics`. Otra no busca, se inventa el 3 % y llama a `loan_metrics` con argumentos que no existen. La última se rinde diciendo que falta el importe, sin usar la base de conocimiento.
- En el ratio de endeudamiento no pasó `monthly_income`, la herramienta devolvió `debt_ratio_pct` nulo y el modelo lo calculó a mano ignorando la cuota (300 entre 2200, 13.64 en vez de 27.79); además no guardó el informe que pedía la tarea.
- En la tarea imposible ("el Euribor del 1 de enero de 2030") llamó a `get_euribor` y dio la tasa actual, con una frase de cautela, en vez de decir que no se puede saber.

Los cerebros de 0.6B no fallan por el bucle. En `base` 12 de 22 fallos no llaman a ninguna herramienta, y muchos terminan los 10 pasos repitiendo `Action: loan_metrics` y recibiendo el error "tool_call JSON has no 'name' field" sin corregirlo, por eso sube de 5.3 a 7.3 pasos de media. `rlm` inventa cifras redondas (`1000.00`, `1200.00`) o razona sobre un resultado que no ha obtenido. El adaptador de la fase 1 se entrenó para producir `<think>` y `<answer>`, no llamadas a herramientas, y no transfiere.

**Qué concluimos.**
- Más de la mitad de la distancia entre `thinking` v1 y v2 (de 0.26 a 0.57) era un fallo de nuestro bucle, no del modelo. Con los tres arreglos pasa de usar bien las herramientas en 0.26 de las tareas a 0.70.
- Para un agente de varios pasos el tamaño del modelo pesa más que la especialización: `thinking` (1.7B) supera con claridad a los dos de 0.6B, y GRPO sobre aritmética no hace de `rlm` un agente. La comparación mezcla tamaño y entrenamiento (es otro modelo base), así que no podemos atribuir la diferencia solo a una de las dos cosas.
- Limitaciones: son 23 tareas, una ejecución y decodificación voraz. Los arreglos del bucle los diseñamos mirando las dos primeras tareas de este mismo banco, así que el 0.57 es una cifra de desarrollo y probablemente optimista. El banco no distingue un fallo de recuperación de uno de razonamiento más que leyendo las trazas, como hicimos aquí.
- Para el proyecto final: SFT de uso de herramientas para los cerebros pequeños, un `top_k` mayor por defecto en `search_knowledge_base`, validar los argumentos obligatorios antes de ejecutar (por ejemplo, avisar de que el ratio necesita `monthly_income`) y un banco más grande con una partición que no se use para iterar.

## Fase 4 · Agente JSON frente a agente de código (smolagents)
**Fecha:** 2026-11-01
**Hipótesis:** un agente que escribe código Python (CodeAgent) es más flexible que uno que escribe una llamada JSON por paso, pero con un modelo de 1.7B el formato de código es más frágil.
**Configuración:** mismo cerebro (thinking, Qwen3-1.7B), mismas herramientas (adaptadas con `to_smolagents_tool`), Euribor fijo del banco, primeras 8 tareas, máximo 8 pasos.
**Resultado:**
| agente | éxito | herramientas requeridas | pasos | tokens |
|---|---|---|---|---|
| JSON | 0.62 (5/8) | 0.88 | 3.0 | 8091 |
| código | 0.38 (3/8) | 0.50 | 4.4 | 32744 |
**Interpretación:** el agente de código gasta unas 4 veces más tokens y acierta menos. Fallos vistos en el log: bloque `<code>` sin cerrar porque el `<think>` se come los tokens (error de parseo y reintento hasta el máximo de pasos), sumar un número al diccionario que devuelve `get_euribor`, y llamar a `loan_metrics` con argumentos inventados (`monthly_payment=0`, `monthly_income=0`). Solo 8 tareas: es una tendencia, no una medida firme. Tardó más de una hora por los reintentos.
**Pendiente (proyecto final):** repetir con más tareas y sin modo de pensamiento para separar formato de modelo.