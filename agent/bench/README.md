# Banco de tareas de la fase 4

`tasks.jsonl` tiene 23 tareas de analista de riesgo de crédito. Cada línea trae la tarea, las
herramientas que hay que encadenar (`required_tools`), una comprobación automática de la
respuesta final (`check`) y el tipo de cadena (`kind`).

| Cadena | Tareas | Ejemplo |
|---|---|---|
| Euribor → `loan_metrics` | 8 | cuota de un préstamo a "Euribor 12M + 3 puntos" |
| Base de conocimiento → `loan_metrics` | 5 | usar como comisión el porcentaje máximo de compensación que fija la ley |
| Base de conocimiento → `save_credit_report` | 3 | buscar el plazo de desistimiento y guardar un informe que lo cite |
| `loan_metrics` → `save_credit_report` | 3 | calcular una TAE y guardar el informe |
| Euribor → `loan_metrics` → `save_credit_report` | 2 | tres herramientas en cadena |
| Imposibles | 2 | el Euribor de 2030, el saldo bancario de un cliente |

Las 21 primeras necesitan al menos dos herramientas. Las dos últimas no se pueden resolver con
lo que tiene el agente: se acierta si lo dice claramente en vez de inventar una cifra.

## Cómo se comprueba

* Tareas numéricas: `check.type = "number"` con tolerancia de 0.02. El valor esperado sale de
  `loan_metrics` con los datos del campo `calc` (los tests lo recalculan, así que el banco no
  se puede desviar de las herramientas sin que falle un test). Las tareas piden "números con
  punto decimal y sin separador de miles" porque el comprobador entregado solo lee ese formato.
* Tareas con la base de conocimiento y el informe: `check.type = "regex"` sobre el dato que
  debe citarse (14 días, 1 por 100, 75.000 euros).
* Imposibles: `regex` que acepta las fórmulas habituales de "no puedo / no dispongo / no es
  posible".
* `evaluate.py` también calcula `required_tools_rate`: en qué fracción de las tareas el agente
  usó de verdad las herramientas necesarias. Un modelo puede acertar un número de memoria; esta
  métrica lo distingue.

## Euribor congelado

El `get_euribor` real lee el BCE y su valor cambia cada mes, así que un banco con respuestas
numéricas dejaría de ser válido. Con `ARCA_AGENT_FIXTURE=bench` el agente sustituye la
herramienta por una tabla fija (1M 2.0, 3M 2.1, 6M 2.2, 12M 2.5 %) con el mismo nombre,
descripción y esquema; el modelo no puede distinguirla. La API en producción usa el dato real.

## Cómo se ejecuta

```bash
export ARCA_AGENT_FIXTURE=bench
export ARCA_REPORTS_DIR=/tmp/arca_reports        # los informes de prueba no van al repo
# misma configuración de RAG que la API:
export ARCA_RAG_CHUNKING=semantic ARCA_RAG_SEMANTIC_THRESHOLD=0.7 ARCA_RAG_LAMBDA=0.6

uv run python -m agent.evaluate --bench agent/bench/tasks.jsonl --brains base rlm thinking
uv run python -m agent.compare_codeagent --bench agent/bench/tasks.jsonl --brains thinking --limit 8
```
