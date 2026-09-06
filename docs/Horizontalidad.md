# Horizontalización — Personal AGI: conocimiento jerárquico, memoria persistente y agencia asistida

> Transformar el sistema de retrieval vertical en un Personal AGI con identidad persistente, memoria episódica, navegación jerárquica multi-hop asistida por planner, tool calling vía MCP, y tier system para escalabilidad del LLM. Diseñado para funcionar con Qwen3.5-9B EXL3 3.0bpw como validación y escalar automáticamente cuando se integre un modelo más capaz.

---

## Contexto y motivación

### Problema 1: Retrieval vertical no escala

El sistema actual hace retrieval **vertical**: query → similitud vectorial → chunks parecidos → fin. Esto funciona para RAG básico pero no escala para una Personal AGI que necesita **razonar a través del conocimiento**:

- **Horizontal:** encontrar chunks del mismo tópico en otros documentos
- **Drill-down:** categoría → tópico → chunks específicos
- **Multi-hop:** tópico → tópicos relacionados → más chunks
- **Evolución temporal:** cómo cambió un tópico en el tiempo

Con 100k documentos, el sistema actual manda todos al LLM para curación (~11 horas de inferencia). Esto es inescalable.

### Problema 2: El agente no es persistente

Hoy cada interfaz define su propia personalidad:

- **Deep Dive**: system prompt ad-hoc que cambia cada vez que lo editamos
- **Reporter (redacción)**: otro prompt distinto, estricto
- **CLI**: no tiene identidad
- **Dashboard**: no tiene identidad

No hay un núcleo compartido. El agente no recuerda quién es Valen, no recuerda conversaciones anteriores, no sabe qué tópicos ya discutieron. Cada sesión empieza desde cero. Esto no es un Personal AGI — es una colección de chatbots amnésicos.

### Problema 3: El LLM pierde el hilo

El Qwen3.5-9B 3.0bpw tiene limitaciones reales que observamos en producción:

- **Multi-hop**: pierde el hilo cuando necesita razonar через más de 2 saltos
- **Conversaciones largas**: degrada — mezcla idiomas, repite frases, genera texto garabateado
- **Tool calling encadenado**: más de 2-3 tools seguidas se pierde
- **Consolidación de memoria**: no es bueno resumiendo 20 conversaciones y extrayendo insights

Estas limitaciones no se arreglan con un mejor prompt. Se arreglan con **andamiaje externo** que le dé al modelo exactamente lo que necesita en cada paso, sin pedirle que mantenga estado él solo.

### Problema 4: Think mode off

Estamos usando el modelo en `think=False` (ChatML no-think) por restricciones de VRAM y tiempo. Esto significa que el modelo no razona internamente antes de responder. El razonamiento tiene que venir del planner externo.

---

## Arquitectura propuesta

### Visión general

```
Usuario → Planner (determinístico, mantiene estado)
            ├→ Identidad del agente (system prompt base, compartido)
            ├→ RAG retrieval (memoria episódica + conocimiento jerárquico)
            ├→ Tool execution (MCP: search_corpus, get_topic, list_topics, ...)
            ├→ State tracking (qué hizo, qué falta, qué tools ya llamó)
            └→ LLM (síntesis + decisión de próximo paso)
                   ↑↓
              Output cleanup (validación post-generación)
```

El LLM es el cerebro. El planner es el andamiaje que lo sostiene. El modelo no necesita mantener todo en contexto porque el planner le da exactamente lo que necesita en cada paso. Esto es lo que hace que un 9B 3.0bpw pueda funcionar — no le pedimos que haga lo que un 70B hace solo, le damos estructura.

### Los 5 pilares

```
Pilar 1: IDENTIDAD — quién es el agente (compartido por todas las interfaces)
Pilar 2: MEMORIA — qué sabe del mundo (semántica) + qué conversamos (episódica)
Pilar 3: AGENCIA — planner + RAG asistido + MCP tools + output cleanup
Pilar 4: ESCALABILIDAD — tier system para que el LLM no procese todo
Pilar 5: USER MODEL — qué sabe, qué quiere y qué evita el usuario sobre cada tópico
```

---

## Pilar 1: Identidad persistente

---

## Pilar 1: Identidad persistente

### Problema

Cada interfaz inventa su personalidad. No hay continuidad. El agente del Deep Dive no es el mismo que el del Reporter.

### Solución

Un archivo de identidad base que todas las interfaces de **interacción** importan. El Reporter mantiene su propio prompt estricto (para simetría de reportes), pero todo lo que involucre conversación con Valen usa la identidad base.

### Archivos nuevos

- `configs/agent_identity.yaml` — identidad base del Personal AGI

### Estructura

```yaml
# configs/agent_identity.yaml
name: "Personal AGI"
user: "Valen"
language: "español"
persona: |
  Sos el Personal AGI de Valen — una inteligencia general con curiosidad
  insaciable y capacidad de sintetizar cualquier tema. Respondés en español
  claro y natural. Tu conocimiento previo es una herramienta poderosa — lo
  usás libremente para explicar, contextualizar, conectar ideas y profundizar.
  Nunca rechazás evidencia porque contradiga tu conocimiento previo. Si la
  evidencia dice que algo existe o pasó, lo aceptás y construís desde ahí.
  Combinás evidencia + conocimiento previo para dar la respuesta más completa
  y útil posible.
principles:
  - "La evidencia [n] es el ancla factual. Citá [n] para hechos de los documentos."
  - "Tu conocimiento previo enriquece y contextualiza — no lo suprimas."
  - "Si algo no está en la evidencia pero lo sabés, aportalo igual."
  - "No inventes cifras ni citas específicas que no estén en la evidencia."
  - "Si la evidencia no alcanza, dilo explícitamente."
capabilities:
  - search_corpus
  - get_topic_info
  - list_topics
  - recall_conversation
```

### Cómo se usa

Cada interfaz de interacción carga `agent_identity.yaml` al construir el system prompt. Las interfaces que necesitan contexto adicional (Deep Dive: evidencia [n], CLI: tools disponibles) **extienden** el prompt base, no lo reemplazan.

**Deep Dive**: `identidad base` + `evidencia recuperada` + `memoria episódica relevante`
**CLI**: `identidad base` + `tools disponibles` + `memoria episódica relevante`
**Reporter (redacción)**: NO usa identidad base — mantiene su prompt estricto para simetría de reportes

### Archivos a modificar

- `src/res023_lab/reporter_deep_dive.py` — cargar identidad base en lugar de prompt hardcodeado
- `src/res023_lab/agent_identity.py` — NUEVO, loader de `agent_identity.yaml`

---

## Pilar 2: Memoria

### Dos tipos de memoria

```
Memoria semántica — qué sabe del mundo
  → Estructura jerárquica de tópicos, categorías, relaciones
  → Es el conocimiento del corpus + la navegación horizontal
  → Vive en LanceDB (vectores) + SQLite/PostgreSQL (metadata)

Memoria episódica — qué conversamos
  → Registro de cada turno de cada conversación
  → Vinculado a tópicos del conocimiento semántico
  → Vive en SQLite (poco volumen, una persona)
```

### Memoria semántica: los 3 layers de retrieval

```
Layer 1: LEXICAL (puerta de entrada — ya existe)
  BM25 + Tantivy → match exacto de términos, CVEs, nombres propios
  "CVE-2024-3094" → encuentra los chunks exactos

Layer 2: VECTORIAL (fallback semántico — ya existe)
  LanceDB + BGE-M3 → similitud semántica cuando no hay match léxico
  "supply chain attack" → encuentra chunks semánticamente similares

Layer 3: JERÁRQUICO (navegación — NUEVO)
  topic_cluster_id + parent_category_id + topic_links
  Navegación horizontal, drill-down y multi-hop desde cualquier punto
```

### Flujo de retrieval completo

```
1. Punto de entrada (lexical)
   "xz-utils backdoor" → BM25/Tantivy → chunks exactos

2. Fallback semántico (vectorial, si lexical no encuentra)
   LanceDB hybrid → chunks semánticamente similares

3. Navegación horizontal (jerárquico)
   esos chunks → topic_cluster_id → mismo tópico en otros docs

4. Drill-down (jerarquía)
   tópico → parent_category_id → categoría completa

5. Multi-hop (grafo)
   categoría → topic_links → tópicos relacionados → más chunks
```

### Memoria episódica: tablas nuevas

**SQLite (suficiente para una persona, migrable a PostgreSQL después):**

```sql
-- Registro crudo de cada turno de cada conversación
CREATE TABLE agent_episodes (
    episode_id    TEXT PRIMARY KEY,      -- UUID
    interface     TEXT NOT NULL,         -- deep_dive / cli / dashboard / futuro
    session_id    TEXT NOT NULL,         -- agrupa turnos de una sesión
    role          TEXT NOT NULL,         -- user / assistant / tool
    content       TEXT NOT NULL,         -- texto del turno
    topic_cluster_id TEXT,              -- FK → topic_clusters (a qué tópico se refería)
    tool_calls    TEXT,                  -- JSON: tools llamadas en este turno
    created_at    TEXT NOT NULL          -- ISO timestamp
);

CREATE INDEX idx_episodes_session ON agent_episodes(session_id);
CREATE INDEX idx_episodes_topic ON agent_episodes(topic_cluster_id);
CREATE INDEX idx_episodes_created ON agent_episodes(created_at);

-- Memoria consolidada (extractos de conversaciones pasadas)
-- Períodoicamente el LLM lee episodios recientes y extrae lo que vale la pena recordar
-- Esto evita que la memoria crezca infinitamente
CREATE TABLE agent_memory_extracts (
    extract_id     TEXT PRIMARY KEY,     -- UUID
    source_episodes TEXT NOT NULL,       -- JSON: lista de episode_ids que lo generaron
    summary        TEXT NOT NULL,        -- resumen de lo aprendido/decidido
    topic_cluster_id TEXT,              -- FK → topic_clusters
    created_at     TEXT NOT NULL
);

CREATE INDEX idx_extracts_topic ON agent_memory_extracts(topic_cluster_id);
```

### Cómo se conecta la memoria episódica con la semántica

El `topic_cluster_id` en `agent_episodes` es el puente. Cuando el agente conversa sobre GPT-6 Astra, ese episodio se vincula al tópico correspondiente en la jerarquía. Después, si en otra sesión le preguntás algo relacionado, puede:

1. Recuperar episodios previos por tópico (no por similitud de texto)
2. Navegar horizontalmente a tópicos relacionados
3. Traer la memoria consolidada de ese tema

No es una infraestructura paralela — es una extensión natural del mismo grafo de conocimiento. Los episodios cuelgan de los mismos tópicos que los chunks. La memoria episódica y la semántica comparten la misma jerarquía.

### Carga de memoria al iniciar sesión

Antes de responder una query, el agente:

1. Determina el `topic_cluster_id` relevante (via embedding similarity de la query contra topic_centroids)
2. Recupera los últimos 5-10 episodios vinculados a ese tópico
3. Recupera extractos consolidados de ese tópico
4. Todo eso entra como contexto adicional al system prompt

**No se mandan 20 conversaciones al contexto.** Se hace RAG sobre la memoria episódica — mismo patrón que ya usamos para chunks, solo que ahora también recupera de `agent_episodes`.

### Consolidación de memoria

Periódicamente (job manual o automático):

1. Seleccionar episodios no consolidados (sin extract_id asociado)
2. Agrupar por topic_cluster_id
3. Para cada grupo, mandar al LLM: "Leé estos N episodios sobre el tópico X y extraé los puntos clave que vale la pena recordar"
4. Guardar el resumen en `agent_memory_extracts`
5. Marcar los episodios como consolidados

**Honestidad sobre el 9B 3.0bpw**: los resúmenes que produzca van a ser burdos, perderán matices. Un 9B 3.0bpw no es ideal para "leí 20 conversaciones y extraé los insights clave". Pero la estructura está lista — cuando se integre un modelo más capaz, la consolidación mejora sin reescribir nada. Mientras tanto, la consolidación puede ser manual (Valen revisa y edita) o semi-automática (el LLM propone, Valen aprueba).

### Archivos nuevos

- `src/res023_lab/agent_memory.py` — CRUD de `agent_episodes` y `agent_memory_extracts`
- `src/res023_lab/agent_identity.py` — loader de identidad + carga de memoria relevante

### Archivos a modificar

- `src/res023_lab/reporter_deep_dive.py` — escribir episodios después de cada turno, cargar memoria al iniciar
- `scripts/web_dashboard.py` — pasar conversation_history a agent_memory, persistir turnos

---

## Pilar 3: Agencia — Planner + RAG asistido + MCP + Output cleanup

### 3A: Planner determinístico para multi-hop

#### Problema

El 9B 3.0bpw no razona 5 hops solo. Pierde el hilo. Y estamos en think mode off — el modelo no razona internamente antes de responder.

#### Solución

Un **planner determinístico** descompone la consulta en sub-queries, ejecuta cada una contra el retrieval, y alimenta los resultados al LLM como contexto estructurado. El LLM no hace el hop — el planner lo hace y el LLM sintetiza el resultado.

#### Patrón

```
Usuario: "¿Cómo se relaciona el backdoor de xz-utils con los ataques a la supply chain de SolarWinds?"

Planner:
  Step 1: search_corpus("xz-utils backdoor") → chunks [A, B, C]
  Step 2: get_topic_info(topic de A) → topic_cluster_id = T1
  Step 3: get_related_topics(T1) → [T2 (supply chain attacks), T3 (SolarWinds)]
  Step 4: search_corpus("SolarWinds supply chain", topic=T2) → chunks [D, E, F]
  Step 5: search_corpus("SolarWinds supply chain", topic=T3) → chunks [G, H]

  Contexto al LLM:
    "Evidencia sobre xz-utils: [A][B][C]
     Evidencia sobre supply chain attacks (tópico relacionado): [D][E][F]
     Evidencia sobre SolarWinds (tópico relacionado): [G][H]
     Pregunta: ¿Cómo se relacionan?"

  LLM sintetiza la respuesta final.
```

El LLM no necesita mantener 5 pasos en contexto. El planner le entrega el resultado de los 5 pasos como un solo contexto estructurado.

#### Archivos

- `src/res023_lab/reporter_planner.py` — ya existe como esqueleto, expandir con:
  - `plan_query(query) → list[SubQuery]`
  - `execute_step(step, state) → StepResult`
  - `build_context(results) → str`
  - State tracking: qué steps se ejecutaron, qué devolvieron, qué falta

- `src/res023_lab/reporter_retrieval.py` — soporte para filtros por `topic_cluster_id` y `parent_category_id`

### 3B: MCP Tools

#### Problema

El agente necesita poder ejecutar acciones: buscar en el corpus, obtener info de un tópico, listar tópicos disponibles, recordar conversaciones.

#### Solución

MCP (Model Context Protocol) con 3-5 herramientas simples. El 9B 3.0bpw maneja bien schemas simples — 2-3 tools por turno, no más.

#### Tools iniciales

```python
# Tool 1: Buscar en el corpus
{
    "name": "search_corpus",
    "description": "Buscar chunks en el corpus por query. Retorna chunks con score y source.",
    "parameters": {
        "query": {"type": "string", "description": "Consulta de búsqueda"},
        "topic_cluster_id": {"type": "string", "description": "Filtrar por tópico (opcional)"},
        "top_k": {"type": "integer", "description": "Máximo resultados", "default": 5}
    }
}

# Tool 2: Obtener info de un tópico
{
    "name": "get_topic_info",
    "description": "Obtener metadata de un tópico: label, descripción, categoría padre, documentos.",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "ID del tópico"}
    }
}

# Tool 3: Listar tópicos disponibles
{
    "name": "list_topics",
    "description": "Listar todos los tópicos del corpus, opcionalmente filtrados por categoría.",
    "parameters": {
        "parent_category_id": {"type": "string", "description": "Filtrar por categoría (opcional)"}
    }
}

# Tool 4: Recordar conversación
{
    "name": "recall_conversation",
    "description": "Recuperar episodios de conversaciones previas sobre un tópico.",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "Tópico de interés"},
        "limit": {"type": "integer", "description": "Máximo episodios", "default": 5}
    }
}

# Tool 5: Obtener tópicos relacionados
{
    "name": "get_related_topics",
    "description": "Obtener tópicos relacionados vía topic_links (multi-hop).",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "Tópico de origen"},
        "max_hops": {"type": "integer", "description": "Profundidad del hop", "default": 2}
    }
}
```

#### Tool calling encadenado

El planner mantiene el estado de qué tools se ejecutaron y qué devolvieron. El LLM no tiene que recordar 5 pasos — el planner le pasa el resultado del paso anterior como contexto. Es un patrón ReAct básico: el estado vive en el planner, no en el contexto del modelo.

```
Turno 1: LLM decide llamar search_corpus("xz-utils")
         Planner ejecuta, guarda resultado en state
Turno 2: Planner le pasa resultado al LLM
         LLM decide llamar get_topic_info(T1)
         Planner ejecuta, guarda resultado en state
Turno 3: Planner le pasa resultado al LLM
         LLM decide llamar get_related_topics(T1)
         Planner ejecuta, guarda resultado en state
Turno 4: Planner le pasa todos los resultados al LLM
         LLM sintetiza respuesta final
```

El LLM solo decide "¿qué tool llamar ahora?" basado en lo que ya tiene. No mantiene estado.

#### Archivos

- `src/res023_lab/mcp_server.py` — ya existe, expandir con las 5 tools
- `src/res023_lab/agent_tools.py` — NUEVO, implementación de las tools como funciones Python
- `src/res023_lab/agent_planner.py` — NUEVO, planner ReAct con state tracking

### 3C: Output cleanup

#### Problema

El 9B 3.0bpw degrada en conversaciones largas: mezcla idiomas, repite frases, genera texto garabateado (vimos "Essen isimerkizi" en una respuesta).

#### Solución

Un pase de validación/cleanup post-generación. No es perfecto pero suaviza el problema.

#### Chequeos

1. **Detección de idioma mezclado**: identificar tokens no españoles (excepto términos técnicos/nombres propios) y marcarlos
2. **Detección de repetición**: si una frase de 5+ palabras se repite 3+ veces, colapsar
3. **Detección de tokens garabateados**: secuencias de caracteres que no forman palabras válidas en ningún idioma
4. **Coherencia con turnos anteriores**: si la respuesta introduce un tema completamente nuevo que no estaba ni en la query ni en la evidencia ni en el contexto, marcarlo como sospechoso
5. **Validación de citas**: verificar que cada [n] en la respuesta corresponde a un chunk real

#### Implementación

```python
# src/res023_lab/agent_cleanup.py (NUEVO)
def clean_output(text: str, evidence_chunks: list, conversation_history: list) -> str:
    """Post-process LLM output to fix degradation artifacts."""
    text = _fix_mixed_languages(text)
    text = _collapse_repetitions(text)
    text = _remove_garbled_tokens(text)
    text = _validate_citations(text, evidence_chunks)
    return text
```

No reescribe la respuesta — solo limpia artefactos obvios. Si el cleanup detecta degradación severa (más de 30% del texto afectado), marca la respuesta para re-generación con contexto reducido.

#### Archivos

- `src/res023_lab/agent_cleanup.py` — NUEVO
- `src/res023_lab/reporter_deep_dive.py` — aplicar cleanup después de generar

---

## Pilar 4: Escalabilidad — Tier system para el LLM

### Problema actual

- Con 82 documentos: 82 calls × 300 tokens = 24,600 tokens → ~12 min
- Con 100k documentos: 100,000 calls × 300 tokens = 30M tokens → ~11 horas

### Solución: clasificación en 3 tiers

```
Tier 1: Determinístico (gratis, instantáneo)
  → keyword matching, hash dedup, quality_score, fecha
  → descarta obvios: duplicados, irrelevantes por fecha/keywords
  → ~40-60% de los docs se descartan aquí

Tier 2: Embeddings (barato, BGE-M3 batch)
  → similitud centroide ↔ intereses
  → descarta claramente irrelevantes (score < 0.2)
  → aprueba claramente relevantes (score > 0.8)
  → ~20-30% adicional se resuelve aquí

Tier 3: LLM (caro, solo borderline)
  → solo docs con score 0.2-0.8 (típicamente 10-20% del total)
  → 100k docs → ~10-20k al LLM en vez de 100k
  → tiempo: ~1-2 horas en vez de ~11 horas
```

### Estado actual

El Tier 1 y Tier 2 ya están implementados en `reporter_curation.py`:
- Tier 1: deduplicación por URL, hash, fecha, quality_score
- Tier 2: cosine similarity con embeddings de LanceDB para relevance y novelty
- Heurísticas mejoradas para source_quality, impact, depth, actionability

Lo que falta es el **Tier 3 condicional**: solo mandar al LLM los docs con scores borderline (0.2-0.8 en relevance).

### Para labeling de tópicos

```
Actual: label_many(25 grupos) → 25 calls al LLM
Con 100k docs: ~5,000 grupos → 5,000 calls al LLM

Solución:
  1. Clustering jerárquico sobre centroides → agrupa grupos similares
  2. Labelar solo categorías padre (3-8) con LLM
  3. Sub-tópicos heredan label del padre + keywords determinísticas
  4. Solo labelar tópicos NUEVOS (no existentes en runs anteriores)
     → match_topic_continuity ya hace algo de esto
```

---

## Pilar 5: User Model — perfil del usuario

### Problema

El agente no sabe quién es Valen más allá del nombre. No sabe qué temas domina, cuáles le interesan, cuáles evita, ni con qué profundidad discutió cada uno. Cada respuesta es genérica porque el agente trata al usuario como un desconocido.

El `TUTOR_AGENT_DESIGN.md` plantea un "learner model" con estados `unknown → exposed → understood → applied → mastered`. Eso es un caso específico de algo más general: **un modelo del usuario que aplica a cualquier interacción, no solo a tutoría**.

### Solución

Un **user model** vinculado al grafo de tópicos que tracking qué sabe, qué quiere y qué evita el usuario sobre cada tema. No es un inventario estático — se construye desde la memoria episódica y se actualiza con cada interacción.

### Estados del usuario por tópico

```
unknown        no hay evidencia — el usuario nunca mencionó este tópico
exposed        el usuario lo vio o leyó (apareció en un reporte, conversación)
familiar       lo discutió con competencia (puede seguir el hilo)
practiced      lo aplicó en un proyecto o decisión real
expert         lo domina — no necesita explicación básica
interested     quiere profundizar o seguir aprendiendo
avoid          no le interesa o le incomoda
misconception  cree algo incorrecto sobre este tópico
```

Los estados `familiar`, `practiced`, `expert` reemplazan a `understood`, `applied`, `mastered` del learner model. Son más generales: aplican a cualquier dominio, no solo al pedagógico.

### Tabla principal

```sql
-- Estado del usuario por tópico
CREATE TABLE user_topic_records (
    topic_cluster_id  TEXT PRIMARY KEY,      -- FK → topic_clusters
    user_status       TEXT NOT NULL,          -- unknown/exposed/familiar/practiced/expert/interested/avoid/misconception
    expertise_level   REAL DEFAULT 0.0,       -- 0.0-1.0, inferido de evidencia
    interest_level    REAL DEFAULT 0.5,       -- 0.0-1.0, inferido de interacciones
    last_discussed    TEXT,                   -- ISO timestamp de la última conversación
    episode_count     INTEGER DEFAULT 0,      -- cuántas conversaciones tocaron este tópico
    notes             TEXT,                   -- extracto libre: "trabaja en security, entiende CVEs pero no SBOM"
    evidence_json     TEXT NOT NULL,          -- JSON: lista de evidence_ids que respaldan el estado
    updated_at        TEXT NOT NULL
);

CREATE INDEX idx_user_topic_status ON user_topic_records(user_status);
```

### Evidencia

El user model no se inventa — se construye desde evidencia persistente. Cada inferencia debe estar respaldada por:

```sql
-- Registro de evidencia que respalda el user model
CREATE TABLE user_evidence (
    evidence_id     TEXT PRIMARY KEY,         -- UUID
    topic_cluster_id TEXT NOT NULL,           -- FK → topic_clusters
    source_type     TEXT NOT NULL,            -- episode / assessment / tool_result / user_statement / observed_action
    source_id       TEXT,                     -- FK → agent_episodes.episode_id u otro
    evidence_text   TEXT NOT NULL,            -- qué se observó
    inferred_status TEXT,                     -- qué estado sugiere esta evidencia
    confidence      REAL DEFAULT 0.5,         -- 0.0-1.0
    created_at      TEXT NOT NULL
);

CREATE INDEX idx_evidence_topic ON user_evidence(topic_cluster_id);
```

### Cómo se construye el user model

```
agent_episodes (lo que conversaron)
    ↓ consolidación
agent_memory_extracts (lo que vale la pena recordar)
    ↓ vinculación a tópicos
topic_clusters (a qué tópicos se refiere cada episodio)
    ↓ inferencia con evidencia
user_topic_records (qué sabe / qué quiere / qué evita el usuario sobre cada tópico)
```

El user model es la **capa de inferencia** sobre la memoria episódica. No reemplaza a los episodios — los sintetiza en un estado usable.

### Cómo se usa el user model al responder

Antes de generar una respuesta, el agente consulta el user model del tópico relevante:

```
tópico: supply-chain-attacks
  user_status: familiar
  expertise_level: 0.6
  interest_level: 0.8
  notes: "trabaja en security, entiende CVEs pero no profundizó en SBOM"
```

Y ajusta:

- **expert** → no explicar basics, ir directo al detalle técnico
- **familiar** → asumir conocimiento medio, profundizar donde hay gaps
- **exposed** → explicar contexto pero no desde cero
- **unknown** → explicar desde cero, contextualizar
- **interested** → ofrecer profundización, fuentes adicionales
- **avoid** → no insistir con el tema salvo que el usuario lo pida
- **misconception** → corregir con evidencia, no ignorar

### Inferencia del user model

El LLM **no infiere el user model solo**. La inferencia se hace con evidencia persistente:

1. **Automática**: después de cada conversación, un job analiza los episodios y actualiza `user_topic_records` basándose en `user_evidence`
2. **Semi-automática**: el LLM propone un estado, Valen aprueba o corrige
3. **Manual**: Valen edita su perfil directamente desde el dashboard

La confianza numérica (`expertise_level`, `interest_level`) solo se actualiza cuando hay suficiente evidencia — no es un valor inventado por el LLM.

### Generalización del learner model

El `TUTOR_AGENT_DESIGN.md` define:

| Learner model (Tutor) | User model (Personal AGI) |
|---|---|
| `MasteryRecord` | `UserTopicRecord` |
| `LearningGoal` | `Goal` (cualquier objetivo, no solo pedagógico) |
| `AssessmentAttempt` | `EvidenceRecord` (cualquier evidencia, no solo evaluaciones) |
| `understood` | `familiar` |
| `applied` | `practiced` |
| `mastered` | `expert` |
| `misconception` | `misconception` (sin cambio) |

El Tutor Agent pasa a ser **un modo del Personal AGI** que usa el user model con policies pedagógicas. No tiene su propio learner model separado — usa el mismo `user_topic_records` con interpretación orientada a enseñanza.

### Perfil global del usuario

Además del estado por tópico, hay un perfil global que no depende de tópicos específicos:

```sql
-- Perfil global del usuario (no vinculado a tópicos)
CREATE TABLE user_profile (
    key         TEXT PRIMARY KEY,             -- ej: "profession", "language_preference", "communication_style"
    value       TEXT NOT NULL,                -- ej: "security analyst", "español", "directo técnico"
    confidence  REAL DEFAULT 0.5,
    source      TEXT NOT NULL,                -- user_statement / inferred / manual
    updated_at  TEXT NOT NULL
);
```

Esto guarda cosas como:

- Profesión / área de trabajo
- Idioma preferido
- Estilo de comunicación (directo, técnico, conversacional)
- Zona horaria
- Preferencias de profundidad
- Cosas que el usuario dijo explícitamente sobre sí mismo

### Archivos nuevos

- `src/res023_lab/user_model.py` — CRUD de `user_topic_records`, `user_evidence`, `user_profile`
- `src/res023_lab/user_model_inference.py` — inferencia del user model desde episodios

### Archivos a modificar

- `src/res023_lab/reporter_deep_dive.py` — consultar user model antes de responder
- `src/res023_lab/agent_memory.py` — después de consolidar episodios, actualizar user model
- `scripts/web_dashboard.py` — endpoint para ver/editar el user model
- `web/static/app.js` — panel de perfil de usuario

---

## División de tecnologías

### LanceDB (vectorial)

```
Tabla: chunks (ya existe + columnas nuevas)
  vector[1024]           ← BGE-M3 (ya existe)
  document_id            ← (ya existe)
  chunk_id               ← (ya existe)
  text                   ← (ya existe)
  sparse_json            ← (ya existe)
  topic_cluster_id       ← NUEVO: tópico al que pertenece
  parent_category_id     ← NUEVO: categoría padre
  temporal_bucket        ← NUEVO: semana/mes para evolución

Tabla: topic_centroids (NUEVA — solo vectores)
  cluster_id             ← PK
  centroid_vector[1024]  ← centroide del tópico para búsqueda vectorial
  label                  ← label del tópico
  parent_category_id     ← categoría padre
  doc_count              ← cuántos docs
  chunk_count            ← cuántos chunks
```

### SQLite (relacional — tablas existentes + nuevas)

```
Tablas existentes (se mantienen):
  document_store, scrape_history, dashboard, reporter decisions

Tablas nuevas (memoria episódica):
  agent_episodes         ← registro crudo de conversaciones
  agent_memory_extracts  ← memoria consolidada

Tablas nuevas (user model):
  user_topic_records     ← estado del usuario por tópico
  user_evidence          ← evidencia que respalda el user model
  user_profile           ← perfil global (profesión, idioma, estilo)

Tablas nuevas (jerárquicas — mientras PG no se necesite):
  topic_clusters         ← metadata de tópicos
  topic_links            ← grafo de relaciones
  topic_evolution        ← tracking temporal
```

### PostgreSQL (relacional — migración futura cuando se necesite escala)

```
Migrar cuando:
  - topic_clusters supere 10k registros
  - agent_episodes supere 50k registros
  - user_topic_records supere 5k registros
  - Se necesite acceso concurrente multi-agente

Las tablas existentes (document_store, scrape_history, dashboard)
se mantienen en SQLite. Solo las tablas jerárquicas y de memoria
migran a PostgreSQL.
```

| Aspecto | SQLite | PostgreSQL |
|---|---|---|
| Recursive CTEs (jerarquía) | ✅ pero lento | ✅ optimizado |
| Multi-hop graph traversal | ⚠️ sin índices | ✅ con índices |
| Multi-agente concurrente | ❌ 1 writer | ✅ MVCC |
| JSONB metadata | ❌ JSON text | ✅ indexado |
| Escala 100k+ tópicos | ⚠️ se degrada | ✅ sin problema |

---

## Implementación por etapas

### Etapa 1: Identidad persistente

**Archivos nuevos:**

- `configs/agent_identity.yaml` — identidad base del Personal AGI
- `src/res023_lab/agent_identity.py` — loader de identidad

**Archivos a modificar:**

- `src/res023_lab/reporter_deep_dive.py` — cargar identidad base en lugar de prompt hardcodeado

**Qué hace:**

1. Define la identidad del agente en un archivo YAML compartido
2. `agent_identity.py` carga el YAML y construye el system prompt base
3. El Deep Dive usa `load_identity()` en lugar del string hardcodeado
4. El Reporter NO cambia — mantiene su prompt estricto

**Validación:**

- El Deep Dive responde con la personalidad definida en el YAML
- Cambiar el YAML cambia el comportamiento sin tocar código
- El Reporter sigue siendo estricto

### Etapa 2: Memoria episódica

**Archivos nuevos:**

- `src/res023_lab/agent_memory.py` — CRUD de episodios y extractos

**Archivos a modificar:**

- `src/res023_lab/reporter_deep_dive.py` — escribir episodios después de cada turno, cargar memoria al iniciar
- `scripts/web_dashboard.py` — persistir turnos en agent_episodes, pasar history a deep_dive

**Qué hace:**

1. Crea tablas `agent_episodes` y `agent_memory_extracts` en SQLite
2. Después de cada turno (user + assistant), escribe un episodio
3. Al iniciar una sesión, carga episodios previos relevantes al tópico de la query
4. `agent_memory_extracts` se llena manualmente o semi-automáticamente por ahora

**Validación:**

- Después de una conversación sobre X, iniciar una nueva sesión y preguntar sobre X → el agente recuerda
- Los episodios se vinculan a topic_cluster_id cuando es posible
- La tabla no crece infinitamente porque los extractos consolidan

### Etapa 3: Clustering jerárquico sobre centroides existentes

**Archivos a modificar:**

- `src/res023_lab/lancedb_index.py` — agregar `_compute_topic_clusters()`, `document_embeddings()` ya existe
- `src/res023_lab/document_store.py` — agregar tabla `document_topic_assignments`
- `scripts/run_fast_path.py` — llamar `_compute_topic_clusters()` después de `_compute_centroids()`

**Qué hace:**

1. Toma los centroides ya computados por `_compute_centroids()`
2. Hace agglomerative clustering (scipy `linkage`) sobre los vectores centroide
3. Threshold de similitud → define tópicos
4. Guarda `topic_cluster_id` en cada chunk de LanceDB
5. Computa centroides de tópicos → guarda en tabla `topic_centroids` en LanceDB

**Validación:**

- `_compute_topic_clusters()` produce cluster_ids consistentes para chunks del mismo tópico
- `document_embeddings()` ya funciona (implementado en etapa anterior)

### Etapa 4: Categorías padre y grafo de tópicos

**Archivos a modificar:**

- `src/res023_lab/reporter_topics.py` — `group_topics_into_categories()` usa centroides en vez de LLM
- `src/res023_lab/lancedb_index.py` — agregar `_compute_parent_categories()`
- `src/res023_lab/reporter_pipeline.py` — integrar categorías jerárquicas

**Qué hace:**

1. Segundo nivel de clustering sobre centroides de tópicos → categorías padre
2. Guarda `parent_category_id` en cada chunk de LanceDB
3. LLM solo se usa para labelar las 3-8 categorías padre (no los 25 tópicos)
4. Sub-tópicos usan `_fallback_label()` determinístico + label heredado del padre
5. Crea tabla `topic_links` con relaciones entre tópicos

**Validación:**

- `parent_category_id` agrupa tópicos en 3-8 categorías coherentes
- Ninguna categoría contiene más del 50% de los tópicos

### Etapa 5: MCP Tools + Planner ReAct

**Archivos nuevos:**

- `src/res023_lab/agent_tools.py` — implementación de las 5 tools
- `src/res023_lab/agent_planner.py` — planner ReAct con state tracking
- `src/res023_lab/agent_cleanup.py` — output cleanup post-generación

**Archivos a modificar:**

- `src/res023_lab/mcp_server.py` — registrar las 5 tools
- `src/res023_lab/reporter_deep_dive.py` — integrar planner + cleanup
- `src/res023_lab/reporter_planner.py` — expandir con plan_query, execute_step, build_context
- `src/res023_lab/reporter_retrieval.py` — soporte para filtros por `topic_cluster_id`

**Qué hace:**

1. Implementa las 5 tools: `search_corpus`, `get_topic_info`, `list_topics`, `recall_conversation`, `get_related_topics`
2. El planner descompone queries complejas en sub-queries
3. Ejecuta cada sub-query contra el retrieval o las tools
4. Mantiene estado de qué se ejecutó y qué devolvió
5. Al final, construye un contexto estructurado y se lo pasa al LLM
6. Después de generar, aplica output cleanup

**Validación:**

- Query multi-hop: "¿Cómo se relaciona X con Y?" → planner ejecuta 3-4 steps → LLM sintetiza
- Tool calling: el LLM decide llamar `search_corpus` → planner ejecuta → LLM usa el resultado
- Output cleanup: detecta idioma mezclado y lo limpia

### Etapa 6: Deep dive con navegación jerárquica

**Archivos a modificar:**

- `src/res023_lab/reporter_deep_dive.py` — navegación horizontal y multi-hop
- `src/res023_lab/reporter_retrieval.py` — filtros por `topic_cluster_id` y `parent_category_id`
- `src/res023_lab/reporter_planner.py` — planear queries multi-hop

**Qué hace:**

1. Después del retrieval léxico + vectorial, obtiene `topic_cluster_id` de los hits
2. Navegación horizontal: `WHERE topic_cluster_id = X` → chunks del mismo tópico
3. Drill-down: `WHERE parent_category_id = X` → categoría completa
4. Multi-hop: `topic_links` → tópicos relacionados → más chunks
5. Evolución temporal: `temporal_bucket` → cómo cambió el tópico

**Validación:**

- Deep dive encuentra chunks relacionados vía navegación jerárquica
- Multi-hop: tópico A → relacionado B → chunks de B aparecen en la respuesta

### Etapa 7: Evolución temporal y continuidad

**Archivos a modificar:**

- `src/res023_lab/reporter_topics.py` — `match_topic_continuity` usa `cluster_id` en vez de similitud léxica
- `src/res023_lab/reporter_store.py` — tabla `topic_evolution`

**Qué hace:**

1. Cuando un tópico ya existe de runs anteriores, reusa el `cluster_id`
2. Trackea `first_seen`, `last_seen`, `evolution` (new/continuing/merged/split)
3. El LLM solo labela tópicos nuevos
4. Tópicos existentes heredan label + description del run anterior

**Validación:**

- Tópicos existentes se reconocen y reusan label en runs siguientes
- `topic_evolution` trackea cambios entre períodos

### Etapa 8: Consolidación de memoria episódica

**Archivos a modificar:**

- `src/res023_lab/agent_memory.py` — función `consolidate_episodes()`
- `scripts/run_memory_consolidation.py` — NUEVO, job periódico

**Qué hace:**

1. Selecciona episodios no consolidados agrupados por `topic_cluster_id`
2. Para cada grupo, manda al LLM: "Leé estos N episodios sobre el tópico X y extraé los puntos clave"
3. Guarda el resumen en `agent_memory_extracts`
4. Marca los episodios como consolidados

**Honestidad**: el 9B 3.0bpw no va a producir extractos de alta calidad. Esta etapa es para validar el flujo — la calidad mejora cuando se integre un modelo más capaz. Mientras tanto, la consolidación puede ser manual o semi-automática.

**Validación:**

- Después de consolidar, `recall_conversation` devuelve el extracto en lugar de los episodios crudos
- Los episodios consolidados se marcan como tales

### Etapa 9: User Model — perfil del usuario

**Archivos nuevos:**

- `src/res023_lab/user_model.py` — CRUD de `user_topic_records`, `user_evidence`, `user_profile`
- `src/res023_lab/user_model_inference.py` — inferencia del user model desde episodios consolidados

**Archivos a modificar:**

- `src/res023_lab/reporter_deep_dive.py` — consultar user model antes de responder, ajustar nivel de explicación
- `src/res023_lab/agent_memory.py` — después de consolidar episodios, disparar inferencia del user model
- `scripts/web_dashboard.py` — endpoint para ver/editar el user model
- `web/static/app.js` — panel de perfil de usuario

**Qué hace:**

1. Crea tablas `user_topic_records`, `user_evidence`, `user_profile` en SQLite
2. Después de cada conversación, registra evidencia en `user_evidence`
3. Un job periódico (o manual) infiere el `user_status` por tópico desde la evidencia acumulada
4. El agente consulta el user model antes de responder y ajusta el nivel de explicación
5. Valen puede ver y editar su perfil desde el dashboard

**Cómo se ajusta la respuesta según el user model:**

- `expert` → no explicar basics, ir directo al detalle técnico
- `familiar` → asumir conocimiento medio, profundizar donde hay gaps
- `exposed` → explicar contexto pero no desde cero
- `unknown` → explicar desde cero, contextualizar
- `interested` → ofrecer profundización, fuentes adicionales
- `avoid` → no insistir con el tema salvo que el usuario lo pida
- `misconception` → corregir con evidencia, no ignorar

**Inferencia semi-automática**: el LLM propone un estado basándose en evidencia, Valen aprueba o corrige. La confianza numérica solo se actualiza con suficiente evidencia — no es un valor inventado.

**Validación:**

- Después de conversar sobre un tópico N veces, el `user_topic_records` refleja el estado correcto
- El agente ajusta el nivel de explicación según el `user_status`
- Valen puede editar su perfil desde el dashboard
- El Tutor Agent usa el mismo user model (no tiene learner model separado)

### Etapa 10: PostgreSQL (migración futura)

**Archivos nuevos:**

- `src/res023_lab/pg_store.py` — adapter PostgreSQL para `topic_clusters`, `topic_links`, `topic_evolution`, `agent_episodes`, `agent_memory_extracts`, `user_topic_records`, `user_evidence`, `user_profile`
- `configs/pg.yaml` — configuración de conexión

**Archivos a modificar:**

- `src/res023_lab/reporter_store.py` — opcionalmente migrar tablas jerárquicas a PG
- `src/res023_lab/agent_memory.py` — soporte dual SQLite/PG

**Qué hace:**

1. Crear tablas en PostgreSQL
2. Adapter `pg_store.py` con misma interfaz que los adapters SQLite
3. Queries recursivas para jerarquía y multi-hop
4. Migración gradual: lo nuevo va a PG, lo existente se mantiene en SQLite

**Cuándo migrar:**

- `topic_clusters` > 10k registros
- `agent_episodes` > 50k registros
- Se necesite acceso concurrente multi-agente

**Validación:**

- PostgreSQL responde queries recursivas (jerarquía) y multi-hop (graph traversal)
- Migración no pierde datos

---

## Resiliencia del Reporter frente a timeouts

Las llamadas al LLM no se ejecutan como un único batch monolítico. Se procesan en lotes independientes y cada lote se persiste al terminar:

```text
Lote normal (curación: 2 documentos; labels: 2 tópicos)
    ↓ timeout/error
Reintento dividido a la mitad
    ↓ si vuelve a fallar
Procesamiento tópico/documento por tópico
    ↓ si falla el elemento individual
Fallback determinístico solo para ese elemento
```

Después de cada lote se ejecuta `reset_generator()` para limpiar el estado del generador y evitar que un timeout invalide toda la fase. La respuesta se valida por cardinalidad, JSON válido y correspondencia con los elementos de entrada.

Para categorías padre se aplican validaciones estructurales:

- entre 3 y 8 categorías cuando hay más de 12 tópicos;
- ninguna categoría puede contener más del 50% de los tópicos;
- si el resultado no cumple, se usa el agrupamiento determinístico seguro;
- el fallback evita keywords genéricas para no producir mega-categorías.

Esto permite que un fallo parcial degrade únicamente un lote o tópico, en lugar de perder el reporte completo.

---

## Honestidad sobre el modelo actual (Qwen3.5-9B EXL3 3.0bpw)

### Lo que puede

- Recordar quién sos y qué te interesa — es un system prompt, no razonamiento
- Tool calling básico — Qwen3.5 está entrenado para function calling, incluso cuantizado a 3.0bpw mantiene la estructura JSON
- Recuperar memoria previa y usarla como contexto — siempre que la memoria esté bien estructurada, el modelo solo necesita leerla
- Síntesis de 2-3 fragmentos de evidencia — hasta ahí llega bien
- MCP simple — 3-5 herramientas con schemas claros

### Lo que no puede bien (y cómo lo mitigamos)

| Limitación | Mitigación |
|---|---|
| Multi-hop complejo | Planner determinístico que descompone y ejecuta los hops |
| Conversaciones largas | Output cleanup + reducir contexto a 4 turnos |
| Tool calling encadenado | Planner mantiene estado, LLM solo decide próximo paso |
| Consolidación de memoria | Estructura lista, calidad mejora con modelo mejor. Mientras tanto: manual o semi-auto |
| Think mode off | Planner externo suple el razonamiento que el modelo no hace internamente |

### El techo real

La arquitectura no depende del modelo. Las tablas de memoria, la identidad, el MCP, la jerarquía de tópicos, el planner — todo eso es estructura. Cuando se integre un modelo más capaz (32B, 70B, o lo que venga), todo ya está listo y el agente mejora sin reescribir nada.

El 9B 3.0bpw es suficiente para **validar que la arquitectura funciona**. No para tener un agente de alta calidad, pero sí para tener uno mínimamente capaz que te conoce, recuerda de qué hablaron, sabe buscar en la base, y puede ejecutar herramientas básicas.

La pregunta no es "¿es buen agente?" sino "¿la arquitectura está bien diseñada para que cuando pongas un modelo mejor ahí, todo mejore automáticamente?" Si la respuesta es sí, vale la pena construirla ahora con el 9B como validación.

---

## Archivos a crear/modificar (resumen)

### Archivos nuevos

| Archivo | Etapa | Propósito |
|---|---|---|
| `configs/agent_identity.yaml` | 1 | Identidad base del Personal AGI |
| `src/res023_lab/agent_identity.py` | 1 | Loader de identidad + carga de memoria relevante |
| `src/res023_lab/agent_memory.py` | 2, 8 | CRUD de episodios y extractos de memoria |
| `src/res023_lab/agent_tools.py` | 5 | Implementación de las 5 MCP tools |
| `src/res023_lab/agent_planner.py` | 5 | Planner ReAct con state tracking |
| `src/res023_lab/agent_cleanup.py` | 5 | Output cleanup post-generación |
| `src/res023_lab/user_model.py` | 9 | CRUD de user_topic_records, user_evidence, user_profile |
| `src/res023_lab/user_model_inference.py` | 9 | Inferencia del user model desde episodios |
| `scripts/run_memory_consolidation.py` | 8 | Job periódico de consolidación |
| `src/res023_lab/pg_store.py` | 10 | Adapter PostgreSQL |
| `configs/pg.yaml` | 10 | Configuración PG |

### Archivos a modificar

| Archivo | Etapa | Cambio |
|---|---|---|
| `src/res023_lab/reporter_deep_dive.py` | 1, 2, 5, 6, 9 | Identidad base, memoria episódica, planner, cleanup, navegación jerárquica, user model |
| `scripts/web_dashboard.py` | 2, 9 | Persistir turnos en agent_episodes, endpoint de user model |
| `web/static/app.js` | 9 | Panel de perfil de usuario |
| `src/res023_lab/lancedb_index.py` | 3, 4 | `_compute_topic_clusters()`, `_compute_parent_categories()`, columnas nuevas |
| `src/res023_lab/document_store.py` | 3 | Tabla `document_topic_assignments` |
| `scripts/run_fast_path.py` | 3 | Llamar `_compute_topic_clusters()` después de centroids |
| `src/res023_lab/reporter_topics.py` | 4, 7 | `group_topics_into_categories()` usa centroides, `match_topic_continuity` usa cluster_id |
| `src/res023_lab/reporter_pipeline.py` | 4 | Integrar categorías jerárquicas |
| `src/res023_lab/mcp_server.py` | 5 | Registrar las 5 tools |
| `src/res023_lab/reporter_planner.py` | 5, 6 | Expandir con plan_query, execute_step, build_context, multi-hop |
| `src/res023_lab/reporter_retrieval.py` | 6 | Filtros por `topic_cluster_id` y `parent_category_id` |
| `src/res023_lab/reporter_store.py` | 7, 9 | Tabla `topic_evolution`, migración opcional a PG |

---

## Verification

- [ ] Etapa 1: El Deep Dive usa `agent_identity.yaml` — cambiar el YAML cambia el comportamiento
- [ ] Etapa 2: Después de una conversación, iniciar nueva sesión y preguntar lo mismo → el agente recuerda
- [ ] Etapa 2: Los episodios se vinculan a `topic_cluster_id` cuando es posible
- [ ] Etapa 3: `_compute_topic_clusters()` produce cluster_ids consistentes para chunks del mismo tópico
- [ ] Etapa 4: `parent_category_id` agrupa tópicos en 3-8 categorías coherentes
- [ ] Etapa 5: Query multi-hop → planner ejecuta 3-4 steps → LLM sintetiza respuesta coherente
- [ ] Etapa 5: Tool calling → LLM decide llamar `search_corpus` → planner ejecuta → LLM usa resultado
- [ ] Etapa 5: Output cleanup detecta y limpia idioma mezclado
- [ ] Etapa 6: Deep dive encuentra chunks relacionados vía navegación jerárquica
- [ ] Etapa 7: Tópicos existentes se reconocen y reusan label en runs siguientes
- [ ] Etapa 8: Después de consolidar, `recall_conversation` devuelve el extracto en lugar de episodios crudos
- [ ] Etapa 9: Después de conversar sobre un tópico N veces, `user_topic_records` refleja el estado correcto
- [ ] Etapa 9: El agente ajusta el nivel de explicación según `user_status` (expert vs unknown)
- [ ] Etapa 9: Valen puede ver y editar su perfil desde el dashboard
- [ ] Etapa 9: El Tutor Agent usa el mismo user model (no tiene learner model separado)
- [ ] Etapa 10: PostgreSQL responde queries recursivas (jerarquía) y multi-hop (graph traversal)
- [ ] Tests: `test_reporter.py` valida clustering, jerarquía y continuidad
- [ ] Tests: `test_agent.py` valida identidad, memoria episódica, tools y planner
- [ ] Tests: `test_user_model.py` valida inferencia de estados, evidencia y ajuste de respuesta
- [ ] Benchmarks: comparar tiempo de inferencia LLM antes/después del tier system
- [ ] Benchmarks: comparar calidad de respuesta con/sin planner en queries multi-hop

---

## Risks / Considerations

- **PostgreSQL dependency:** agregar requerimiento de servidor PG. Etapa 9 es opcional hasta que se necesite escala real. SQLite maneja bien miles de episodios de una persona.
- **Clustering quality:** agglomerative clustering puede producir tópicos poco coherentes si el threshold no está bien calibrado. Necesita tuning.
- **Backward compatibility:** los chunks existentes en LanceDB no tendrán `topic_cluster_id`. Hay que re-indexar o hacer migration.
- **LLM timeout:** el `group_topics` con timeout de 300s ya está implementado. El tier system reduce el riesgo pero no lo elimina.
- **Complejidad:** el sistema pasa de 3 índices planos a 3 índices + jerarquía + grafo + memoria episódica + planner + MCP. Más potente pero más complejo de mantener.
- **Calidad del 9B 3.0bpw:** la consolidación de memoria y la síntesis multi-hop van a ser de calidad limitada. La arquitectura está diseñada para que esto mejore automáticamente con un modelo mejor, sin reescribir nada.
- **Memoria infinita:** sin consolidación, `agent_episodes` crece sin límite. La consolidación manual o semi-automática es necesaria desde el inicio, incluso si es de baja calidad.
- **Privacidad:** la memoria episódica contiene conversaciones personales. Debe vivir localmente, nunca enviarse a servicios externos. PostgreSQL, si se usa, debe ser local.
- **Identidad vs Reporter:** el Reporter mantiene su prompt estricto para simetría de reportes. La identidad base solo aplica a interfaces de interacción. No mezclar.
