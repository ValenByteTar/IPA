# HorizontalizaciÃ³n â€” Personal AGI: conocimiento jerÃ¡rquico, memoria persistente y agencia asistida

> Transformar el sistema de retrieval vertical en un Personal AGI con identidad persistente, memoria episÃ³dica, navegaciÃ³n jerÃ¡rquica multi-hop asistida por planner, tool calling vÃ­a MCP, y tier system para escalabilidad del LLM. DiseÃ±ado para funcionar con Qwen3.5-9B EXL3 3.0bpw como validaciÃ³n y escalar automÃ¡ticamente cuando se integre un modelo mÃ¡s capaz.

---

## Contexto y motivaciÃ³n

### Problema 1: Retrieval vertical no escala

El sistema actual hace retrieval **vertical**: query â†’ similitud vectorial â†’ chunks parecidos â†’ fin. Esto funciona para RAG bÃ¡sico pero no escala para una Personal AGI que necesita **razonar a travÃ©s del conocimiento**:

- **Horizontal:** encontrar chunks del mismo tÃ³pico en otros documentos
- **Drill-down:** categorÃ­a â†’ tÃ³pico â†’ chunks especÃ­ficos
- **Multi-hop:** tÃ³pico â†’ tÃ³picos relacionados â†’ mÃ¡s chunks
- **EvoluciÃ³n temporal:** cÃ³mo cambiÃ³ un tÃ³pico en el tiempo

Con 100k documentos, el sistema actual manda todos al LLM para curaciÃ³n (~11 horas de inferencia). Esto es inescalable.

### Problema 2: El agente no es persistente

Hoy cada interfaz define su propia personalidad:

- **Deep Dive**: system prompt ad-hoc que cambia cada vez que lo editamos
- **Reporter (redacciÃ³n)**: otro prompt distinto, estricto
- **CLI**: no tiene identidad
- **Dashboard**: no tiene identidad

No hay un nÃºcleo compartido. El agente no recuerda quiÃ©n es Valen, no recuerda conversaciones anteriores, no sabe quÃ© tÃ³picos ya discutieron. Cada sesiÃ³n empieza desde cero. Esto no es un Personal AGI â€” es una colecciÃ³n de chatbots amnÃ©sicos.

### Problema 3: El LLM pierde el hilo

El Qwen3.5-9B 3.0bpw tiene limitaciones reales que observamos en producciÃ³n:

- **Multi-hop**: pierde el hilo cuando necesita razonar Ñ‡ÐµÑ€ÐµÐ· mÃ¡s de 2 saltos
- **Conversaciones largas**: degrada â€” mezcla idiomas, repite frases, genera texto garabateado
- **Tool calling encadenado**: mÃ¡s de 2-3 tools seguidas se pierde
- **ConsolidaciÃ³n de memoria**: no es bueno resumiendo 20 conversaciones y extrayendo insights

Estas limitaciones no se arreglan con un mejor prompt. Se arreglan con **andamiaje externo** que le dÃ© al modelo exactamente lo que necesita en cada paso, sin pedirle que mantenga estado Ã©l solo.

### Problema 4: Think mode off

Estamos usando el modelo en `think=False` (ChatML no-think) por restricciones de VRAM y tiempo. Esto significa que el modelo no razona internamente antes de responder. El razonamiento tiene que venir del planner externo.

---

## Arquitectura propuesta

### VisiÃ³n general

```
Usuario â†’ Planner (determinÃ­stico, mantiene estado)
            â”œâ†’ Identidad del agente (system prompt base, compartido)
            â”œâ†’ RAG retrieval (memoria episÃ³dica + conocimiento jerÃ¡rquico)
            â”œâ†’ Tool execution (MCP: search_corpus, get_topic, list_topics, ...)
            â”œâ†’ State tracking (quÃ© hizo, quÃ© falta, quÃ© tools ya llamÃ³)
            â””â†’ LLM (sÃ­ntesis + decisiÃ³n de prÃ³ximo paso)
                   â†‘â†“
              Output cleanup (validaciÃ³n post-generaciÃ³n)
```

El LLM es el cerebro. El planner es el andamiaje que lo sostiene. El modelo no necesita mantener todo en contexto porque el planner le da exactamente lo que necesita en cada paso. Esto es lo que hace que un 9B 3.0bpw pueda funcionar â€” no le pedimos que haga lo que un 70B hace solo, le damos estructura.

### Los 5 pilares

```
Pilar 1: IDENTIDAD â€” quiÃ©n es el agente (compartido por todas las interfaces)
Pilar 2: MEMORIA â€” quÃ© sabe del mundo (semÃ¡ntica) + quÃ© conversamos (episÃ³dica)
Pilar 3: AGENCIA â€” planner + RAG asistido + MCP tools + output cleanup
Pilar 4: ESCALABILIDAD â€” tier system para que el LLM no procese todo
Pilar 5: USER MODEL â€” quÃ© sabe, quÃ© quiere y quÃ© evita el usuario sobre cada tÃ³pico
```

---

## Pilar 1: Identidad persistente

---

## Pilar 1: Identidad persistente

### Problema

Cada interfaz inventa su personalidad. No hay continuidad. El agente del Deep Dive no es el mismo que el del Reporter.

### SoluciÃ³n

Un archivo de identidad base que todas las interfaces de **interacciÃ³n** importan. El Reporter mantiene su propio prompt estricto (para simetrÃ­a de reportes), pero todo lo que involucre conversaciÃ³n con Valen usa la identidad base.

### Archivos nuevos

- `configs/agent_identity.yaml` â€” identidad base del Personal AGI

### Estructura

```yaml
# configs/agent_identity.yaml
name: "Personal AGI"
user: "Valen"
language: "espaÃ±ol"
persona: |
  Sos el Personal AGI de Valen â€” una inteligencia general con curiosidad
  insaciable y capacidad de sintetizar cualquier tema. RespondÃ©s en espaÃ±ol
  claro y natural. Tu conocimiento previo es una herramienta poderosa â€” lo
  usÃ¡s libremente para explicar, contextualizar, conectar ideas y profundizar.
  Nunca rechazÃ¡s evidencia porque contradiga tu conocimiento previo. Si la
  evidencia dice que algo existe o pasÃ³, lo aceptÃ¡s y construÃ­s desde ahÃ­.
  CombinÃ¡s evidencia + conocimiento previo para dar la respuesta mÃ¡s completa
  y Ãºtil posible.
principles:
  - "La evidencia [n] es el ancla factual. CitÃ¡ [n] para hechos de los documentos."
  - "Tu conocimiento previo enriquece y contextualiza â€” no lo suprimas."
  - "Si algo no estÃ¡ en la evidencia pero lo sabÃ©s, aportalo igual."
  - "No inventes cifras ni citas especÃ­ficas que no estÃ©n en la evidencia."
  - "Si la evidencia no alcanza, dilo explÃ­citamente."
capabilities:
  - search_corpus
  - get_topic_info
  - list_topics
  - recall_conversation
```

### CÃ³mo se usa

Cada interfaz de interacciÃ³n carga `agent_identity.yaml` al construir el system prompt. Las interfaces que necesitan contexto adicional (Deep Dive: evidencia [n], CLI: tools disponibles) **extienden** el prompt base, no lo reemplazan.

**Deep Dive**: `identidad base` + `evidencia recuperada` + `memoria episÃ³dica relevante`
**CLI**: `identidad base` + `tools disponibles` + `memoria episÃ³dica relevante`
**Reporter (redacciÃ³n)**: NO usa identidad base â€” mantiene su prompt estricto para simetrÃ­a de reportes

### Archivos a modificar

- `src/ipa/reporter/reporter_deep_dive.py` â€” cargar identidad base en lugar de prompt hardcodeado
- `src/ipa/agentic/agent_identity.py` â€” NUEVO, loader de `agent_identity.yaml`

---

## Pilar 2: Memoria

### Dos tipos de memoria

```
Memoria semÃ¡ntica â€” quÃ© sabe del mundo
  â†’ Estructura jerÃ¡rquica de tÃ³picos, categorÃ­as, relaciones
  â†’ Es el conocimiento del corpus + la navegaciÃ³n horizontal
  â†’ Vive en LanceDB (vectores) + SQLite/PostgreSQL (metadata)

Memoria episÃ³dica â€” quÃ© conversamos
  â†’ Registro de cada turno de cada conversaciÃ³n
  â†’ Vinculado a tÃ³picos del conocimiento semÃ¡ntico
  â†’ Vive en SQLite (poco volumen, una persona)
```

### Memoria semÃ¡ntica: los 3 layers de retrieval

```
Layer 1: LEXICAL (puerta de entrada â€” ya existe)
  BM25 + Tantivy â†’ match exacto de tÃ©rminos, CVEs, nombres propios
  "CVE-2024-3094" â†’ encuentra los chunks exactos

Layer 2: VECTORIAL (fallback semÃ¡ntico â€” ya existe)
  LanceDB + BGE-M3 â†’ similitud semÃ¡ntica cuando no hay match lÃ©xico
  "supply chain attack" â†’ encuentra chunks semÃ¡nticamente similares

Layer 3: JERÃRQUICO (navegaciÃ³n â€” NUEVO)
  topic_cluster_id + parent_category_id + topic_links
  NavegaciÃ³n horizontal, drill-down y multi-hop desde cualquier punto
```

### Flujo de retrieval completo

```
1. Punto de entrada (lexical)
   "xz-utils backdoor" â†’ BM25/Tantivy â†’ chunks exactos

2. Fallback semÃ¡ntico (vectorial, si lexical no encuentra)
   LanceDB hybrid â†’ chunks semÃ¡nticamente similares

3. NavegaciÃ³n horizontal (jerÃ¡rquico)
   esos chunks â†’ topic_cluster_id â†’ mismo tÃ³pico en otros docs

4. Drill-down (jerarquÃ­a)
   tÃ³pico â†’ parent_category_id â†’ categorÃ­a completa

5. Multi-hop (grafo)
   categorÃ­a â†’ topic_links â†’ tÃ³picos relacionados â†’ mÃ¡s chunks
```

### Memoria episÃ³dica: tablas nuevas

**SQLite (suficiente para una persona, migrable a PostgreSQL despuÃ©s):**

```sql
-- Registro crudo de cada turno de cada conversaciÃ³n
CREATE TABLE agent_episodes (
    episode_id    TEXT PRIMARY KEY,      -- UUID
    interface     TEXT NOT NULL,         -- deep_dive / cli / dashboard / futuro
    session_id    TEXT NOT NULL,         -- agrupa turnos de una sesiÃ³n
    role          TEXT NOT NULL,         -- user / assistant / tool
    content       TEXT NOT NULL,         -- texto del turno
    topic_cluster_id TEXT,              -- FK â†’ topic_clusters (a quÃ© tÃ³pico se referÃ­a)
    tool_calls    TEXT,                  -- JSON: tools llamadas en este turno
    created_at    TEXT NOT NULL          -- ISO timestamp
);

CREATE INDEX idx_episodes_session ON agent_episodes(session_id);
CREATE INDEX idx_episodes_topic ON agent_episodes(topic_cluster_id);
CREATE INDEX idx_episodes_created ON agent_episodes(created_at);

-- Memoria consolidada (extractos de conversaciones pasadas)
-- PerÃ­odoicamente el LLM lee episodios recientes y extrae lo que vale la pena recordar
-- Esto evita que la memoria crezca infinitamente
CREATE TABLE agent_memory_extracts (
    extract_id     TEXT PRIMARY KEY,     -- UUID
    source_episodes TEXT NOT NULL,       -- JSON: lista de episode_ids que lo generaron
    summary        TEXT NOT NULL,        -- resumen de lo aprendido/decidido
    topic_cluster_id TEXT,              -- FK â†’ topic_clusters
    created_at     TEXT NOT NULL
);

CREATE INDEX idx_extracts_topic ON agent_memory_extracts(topic_cluster_id);
```

### CÃ³mo se conecta la memoria episÃ³dica con la semÃ¡ntica

El `topic_cluster_id` en `agent_episodes` es el puente. Cuando el agente conversa sobre GPT-6 Astra, ese episodio se vincula al tÃ³pico correspondiente en la jerarquÃ­a. DespuÃ©s, si en otra sesiÃ³n le preguntÃ¡s algo relacionado, puede:

1. Recuperar episodios previos por tÃ³pico (no por similitud de texto)
2. Navegar horizontalmente a tÃ³picos relacionados
3. Traer la memoria consolidada de ese tema

No es una infraestructura paralela â€” es una extensiÃ³n natural del mismo grafo de conocimiento. Los episodios cuelgan de los mismos tÃ³picos que los chunks. La memoria episÃ³dica y la semÃ¡ntica comparten la misma jerarquÃ­a.

### Carga de memoria al iniciar sesiÃ³n

Antes de responder una query, el agente:

1. Determina el `topic_cluster_id` relevante (via embedding similarity de la query contra topic_centroids)
2. Recupera los Ãºltimos 5-10 episodios vinculados a ese tÃ³pico
3. Recupera extractos consolidados de ese tÃ³pico
4. Todo eso entra como contexto adicional al system prompt

**No se mandan 20 conversaciones al contexto.** Se hace RAG sobre la memoria episÃ³dica â€” mismo patrÃ³n que ya usamos para chunks, solo que ahora tambiÃ©n recupera de `agent_episodes`.

### ConsolidaciÃ³n de memoria

PeriÃ³dicamente (job manual o automÃ¡tico):

1. Seleccionar episodios no consolidados (sin extract_id asociado)
2. Agrupar por topic_cluster_id
3. Para cada grupo, mandar al LLM: "LeÃ© estos N episodios sobre el tÃ³pico X y extraÃ© los puntos clave que vale la pena recordar"
4. Guardar el resumen en `agent_memory_extracts`
5. Marcar los episodios como consolidados

**Honestidad sobre el 9B 3.0bpw**: los resÃºmenes que produzca van a ser burdos, perderÃ¡n matices. Un 9B 3.0bpw no es ideal para "leÃ­ 20 conversaciones y extraÃ© los insights clave". Pero la estructura estÃ¡ lista â€” cuando se integre un modelo mÃ¡s capaz, la consolidaciÃ³n mejora sin reescribir nada. Mientras tanto, la consolidaciÃ³n puede ser manual (Valen revisa y edita) o semi-automÃ¡tica (el LLM propone, Valen aprueba).

### Archivos nuevos

- `src/ipa/agentic/agent_memory.py` â€” CRUD de `agent_episodes` y `agent_memory_extracts`
- `src/ipa/agentic/agent_identity.py` â€” loader de identidad + carga de memoria relevante

### Archivos a modificar

- `src/ipa/reporter/reporter_deep_dive.py` â€” escribir episodios despuÃ©s de cada turno, cargar memoria al iniciar
- `scripts/web_dashboard.py` â€” pasar conversation_history a agent_memory, persistir turnos

---

## Pilar 3: Agencia â€” Planner + RAG asistido + MCP + Output cleanup

### 3A: Planner determinÃ­stico para multi-hop

#### Problema

El 9B 3.0bpw no razona 5 hops solo. Pierde el hilo. Y estamos en think mode off â€” el modelo no razona internamente antes de responder.

#### SoluciÃ³n

Un **planner determinÃ­stico** descompone la consulta en sub-queries, ejecuta cada una contra el retrieval, y alimenta los resultados al LLM como contexto estructurado. El LLM no hace el hop â€” el planner lo hace y el LLM sintetiza el resultado.

#### PatrÃ³n

```
Usuario: "Â¿CÃ³mo se relaciona el backdoor de xz-utils con los ataques a la supply chain de SolarWinds?"

Planner:
  Step 1: search_corpus("xz-utils backdoor") â†’ chunks [A, B, C]
  Step 2: get_topic_info(topic de A) â†’ topic_cluster_id = T1
  Step 3: get_related_topics(T1) â†’ [T2 (supply chain attacks), T3 (SolarWinds)]
  Step 4: search_corpus("SolarWinds supply chain", topic=T2) â†’ chunks [D, E, F]
  Step 5: search_corpus("SolarWinds supply chain", topic=T3) â†’ chunks [G, H]

  Contexto al LLM:
    "Evidencia sobre xz-utils: [A][B][C]
     Evidencia sobre supply chain attacks (tÃ³pico relacionado): [D][E][F]
     Evidencia sobre SolarWinds (tÃ³pico relacionado): [G][H]
     Pregunta: Â¿CÃ³mo se relacionan?"

  LLM sintetiza la respuesta final.
```

El LLM no necesita mantener 5 pasos en contexto. El planner le entrega el resultado de los 5 pasos como un solo contexto estructurado.

#### Archivos

- `src/ipa/agentic/reporter_planner.py` â€” ya existe como esqueleto, expandir con:
  - `plan_query(query) â†’ list[SubQuery]`
  - `execute_step(step, state) â†’ StepResult`
  - `build_context(results) â†’ str`
  - State tracking: quÃ© steps se ejecutaron, quÃ© devolvieron, quÃ© falta

- `src/ipa/agentic/reporter_retrieval.py` â€” soporte para filtros por `topic_cluster_id` y `parent_category_id`

### 3B: MCP Tools

#### Problema

El agente necesita poder ejecutar acciones: buscar en el corpus, obtener info de un tÃ³pico, listar tÃ³picos disponibles, recordar conversaciones.

#### SoluciÃ³n

MCP (Model Context Protocol) con 3-5 herramientas simples. El 9B 3.0bpw maneja bien schemas simples â€” 2-3 tools por turno, no mÃ¡s.

#### Tools iniciales

```python
# Tool 1: Buscar en el corpus
{
    "name": "search_corpus",
    "description": "Buscar chunks en el corpus por query. Retorna chunks con score y source.",
    "parameters": {
        "query": {"type": "string", "description": "Consulta de bÃºsqueda"},
        "topic_cluster_id": {"type": "string", "description": "Filtrar por tÃ³pico (opcional)"},
        "top_k": {"type": "integer", "description": "MÃ¡ximo resultados", "default": 5}
    }
}

# Tool 2: Obtener info de un tÃ³pico
{
    "name": "get_topic_info",
    "description": "Obtener metadata de un tÃ³pico: label, descripciÃ³n, categorÃ­a padre, documentos.",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "ID del tÃ³pico"}
    }
}

# Tool 3: Listar tÃ³picos disponibles
{
    "name": "list_topics",
    "description": "Listar todos los tÃ³picos del corpus, opcionalmente filtrados por categorÃ­a.",
    "parameters": {
        "parent_category_id": {"type": "string", "description": "Filtrar por categorÃ­a (opcional)"}
    }
}

# Tool 4: Recordar conversaciÃ³n
{
    "name": "recall_conversation",
    "description": "Recuperar episodios de conversaciones previas sobre un tÃ³pico.",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "TÃ³pico de interÃ©s"},
        "limit": {"type": "integer", "description": "MÃ¡ximo episodios", "default": 5}
    }
}

# Tool 5: Obtener tÃ³picos relacionados
{
    "name": "get_related_topics",
    "description": "Obtener tÃ³picos relacionados vÃ­a topic_links (multi-hop).",
    "parameters": {
        "topic_cluster_id": {"type": "string", "description": "TÃ³pico de origen"},
        "max_hops": {"type": "integer", "description": "Profundidad del hop", "default": 2}
    }
}
```

#### Tool calling encadenado

El planner mantiene el estado de quÃ© tools se ejecutaron y quÃ© devolvieron. El LLM no tiene que recordar 5 pasos â€” el planner le pasa el resultado del paso anterior como contexto. Es un patrÃ³n ReAct bÃ¡sico: el estado vive en el planner, no en el contexto del modelo.

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

El LLM solo decide "Â¿quÃ© tool llamar ahora?" basado en lo que ya tiene. No mantiene estado.

#### Archivos

- `src/ipa/mcp/mcp_server.py` â€” ya existe, expandir con las 5 tools
- `src/ipa/agentic/agent_tools.py` â€” NUEVO, implementaciÃ³n de las tools como funciones Python
- `src/ipa/agentic/agent_planner.py` â€” NUEVO, planner ReAct con state tracking

### 3C: Output cleanup

#### Problema

El 9B 3.0bpw degrada en conversaciones largas: mezcla idiomas, repite frases, genera texto garabateado (vimos "Essen isimerkizi" en una respuesta).

#### SoluciÃ³n

Un pase de validaciÃ³n/cleanup post-generaciÃ³n. No es perfecto pero suaviza el problema.

#### Chequeos

1. **DetecciÃ³n de idioma mezclado**: identificar tokens no espaÃ±oles (excepto tÃ©rminos tÃ©cnicos/nombres propios) y marcarlos
2. **DetecciÃ³n de repeticiÃ³n**: si una frase de 5+ palabras se repite 3+ veces, colapsar
3. **DetecciÃ³n de tokens garabateados**: secuencias de caracteres que no forman palabras vÃ¡lidas en ningÃºn idioma
4. **Coherencia con turnos anteriores**: si la respuesta introduce un tema completamente nuevo que no estaba ni en la query ni en la evidencia ni en el contexto, marcarlo como sospechoso
5. **ValidaciÃ³n de citas**: verificar que cada [n] en la respuesta corresponde a un chunk real

#### ImplementaciÃ³n

```python
# src/ipa/agentic/agent_cleanup.py (NUEVO)
def clean_output(text: str, evidence_chunks: list, conversation_history: list) -> str:
    """Post-process LLM output to fix degradation artifacts."""
    text = _fix_mixed_languages(text)
    text = _collapse_repetitions(text)
    text = _remove_garbled_tokens(text)
    text = _validate_citations(text, evidence_chunks)
    return text
```

No reescribe la respuesta â€” solo limpia artefactos obvios. Si el cleanup detecta degradaciÃ³n severa (mÃ¡s de 30% del texto afectado), marca la respuesta para re-generaciÃ³n con contexto reducido.

#### Archivos

- `src/ipa/agentic/agent_cleanup.py` â€” NUEVO
- `src/ipa/reporter/reporter_deep_dive.py` â€” aplicar cleanup despuÃ©s de generar

---

## Pilar 4: Escalabilidad â€” Tier system para el LLM

### Problema actual

- Con 82 documentos: 82 calls Ã— 300 tokens = 24,600 tokens â†’ ~12 min
- Con 100k documentos: 100,000 calls Ã— 300 tokens = 30M tokens â†’ ~11 horas

### SoluciÃ³n: clasificaciÃ³n en 3 tiers

```
Tier 1: DeterminÃ­stico (gratis, instantÃ¡neo)
  â†’ keyword matching, hash dedup, quality_score, fecha
  â†’ descarta obvios: duplicados, irrelevantes por fecha/keywords
  â†’ ~40-60% de los docs se descartan aquÃ­

Tier 2: Embeddings (barato, BGE-M3 batch)
  â†’ similitud centroide â†” intereses
  â†’ descarta claramente irrelevantes (score < 0.2)
  â†’ aprueba claramente relevantes (score > 0.8)
  â†’ ~20-30% adicional se resuelve aquÃ­

Tier 3: LLM (caro, solo borderline)
  â†’ solo docs con score 0.2-0.8 (tÃ­picamente 10-20% del total)
  â†’ 100k docs â†’ ~10-20k al LLM en vez de 100k
  â†’ tiempo: ~1-2 horas en vez de ~11 horas
```

### Estado actual

El Tier 1 y Tier 2 ya estÃ¡n implementados en `reporter_curation.py`:
- Tier 1: deduplicaciÃ³n por URL, hash, fecha, quality_score
- Tier 2: cosine similarity con embeddings de LanceDB para relevance y novelty
- HeurÃ­sticas mejoradas para source_quality, impact, depth, actionability

Lo que falta es el **Tier 3 condicional**: solo mandar al LLM los docs con scores borderline (0.2-0.8 en relevance).

### Para labeling de tÃ³picos

```
Actual: label_many(25 grupos) â†’ 25 calls al LLM
Con 100k docs: ~5,000 grupos â†’ 5,000 calls al LLM

SoluciÃ³n:
  1. Clustering jerÃ¡rquico sobre centroides â†’ agrupa grupos similares
  2. Labelar solo categorÃ­as padre (3-8) con LLM
  3. Sub-tÃ³picos heredan label del padre + keywords determinÃ­sticas
  4. Solo labelar tÃ³picos NUEVOS (no existentes en runs anteriores)
     â†’ match_topic_continuity ya hace algo de esto
```

---

## Pilar 5: User Model â€” perfil del usuario

### Problema

El agente no sabe quiÃ©n es Valen mÃ¡s allÃ¡ del nombre. No sabe quÃ© temas domina, cuÃ¡les le interesan, cuÃ¡les evita, ni con quÃ© profundidad discutiÃ³ cada uno. Cada respuesta es genÃ©rica porque el agente trata al usuario como un desconocido.

El `TUTOR_AGENT_DESIGN.md` plantea un "learner model" con estados `unknown â†’ exposed â†’ understood â†’ applied â†’ mastered`. Eso es un caso especÃ­fico de algo mÃ¡s general: **un modelo del usuario que aplica a cualquier interacciÃ³n, no solo a tutorÃ­a**.

### SoluciÃ³n

Un **user model** vinculado al grafo de tÃ³picos que tracking quÃ© sabe, quÃ© quiere y quÃ© evita el usuario sobre cada tema. No es un inventario estÃ¡tico â€” se construye desde la memoria episÃ³dica y se actualiza con cada interacciÃ³n.

### Estados del usuario por tÃ³pico

```
unknown        no hay evidencia â€” el usuario nunca mencionÃ³ este tÃ³pico
exposed        el usuario lo vio o leyÃ³ (apareciÃ³ en un reporte, conversaciÃ³n)
familiar       lo discutiÃ³ con competencia (puede seguir el hilo)
practiced      lo aplicÃ³ en un proyecto o decisiÃ³n real
expert         lo domina â€” no necesita explicaciÃ³n bÃ¡sica
interested     quiere profundizar o seguir aprendiendo
avoid          no le interesa o le incomoda
misconception  cree algo incorrecto sobre este tÃ³pico
```

Los estados `familiar`, `practiced`, `expert` reemplazan a `understood`, `applied`, `mastered` del learner model. Son mÃ¡s generales: aplican a cualquier dominio, no solo al pedagÃ³gico.

### Tabla principal

```sql
-- Estado del usuario por tÃ³pico
CREATE TABLE user_topic_records (
    topic_cluster_id  TEXT PRIMARY KEY,      -- FK â†’ topic_clusters
    user_status       TEXT NOT NULL,          -- unknown/exposed/familiar/practiced/expert/interested/avoid/misconception
    expertise_level   REAL DEFAULT 0.0,       -- 0.0-1.0, inferido de evidencia
    interest_level    REAL DEFAULT 0.5,       -- 0.0-1.0, inferido de interacciones
    last_discussed    TEXT,                   -- ISO timestamp de la Ãºltima conversaciÃ³n
    episode_count     INTEGER DEFAULT 0,      -- cuÃ¡ntas conversaciones tocaron este tÃ³pico
    notes             TEXT,                   -- extracto libre: "trabaja en security, entiende CVEs pero no SBOM"
    evidence_json     TEXT NOT NULL,          -- JSON: lista de evidence_ids que respaldan el estado
    updated_at        TEXT NOT NULL
);

CREATE INDEX idx_user_topic_status ON user_topic_records(user_status);
```

### Evidencia

El user model no se inventa â€” se construye desde evidencia persistente. Cada inferencia debe estar respaldada por:

```sql
-- Registro de evidencia que respalda el user model
CREATE TABLE user_evidence (
    evidence_id     TEXT PRIMARY KEY,         -- UUID
    topic_cluster_id TEXT NOT NULL,           -- FK â†’ topic_clusters
    source_type     TEXT NOT NULL,            -- episode / assessment / tool_result / user_statement / observed_action
    source_id       TEXT,                     -- FK â†’ agent_episodes.episode_id u otro
    evidence_text   TEXT NOT NULL,            -- quÃ© se observÃ³
    inferred_status TEXT,                     -- quÃ© estado sugiere esta evidencia
    confidence      REAL DEFAULT 0.5,         -- 0.0-1.0
    created_at      TEXT NOT NULL
);

CREATE INDEX idx_evidence_topic ON user_evidence(topic_cluster_id);
```

### CÃ³mo se construye el user model

```
agent_episodes (lo que conversaron)
    â†“ consolidaciÃ³n
agent_memory_extracts (lo que vale la pena recordar)
    â†“ vinculaciÃ³n a tÃ³picos
topic_clusters (a quÃ© tÃ³picos se refiere cada episodio)
    â†“ inferencia con evidencia
user_topic_records (quÃ© sabe / quÃ© quiere / quÃ© evita el usuario sobre cada tÃ³pico)
```

El user model es la **capa de inferencia** sobre la memoria episÃ³dica. No reemplaza a los episodios â€” los sintetiza en un estado usable.

### CÃ³mo se usa el user model al responder

Antes de generar una respuesta, el agente consulta el user model del tÃ³pico relevante:

```
tÃ³pico: supply-chain-attacks
  user_status: familiar
  expertise_level: 0.6
  interest_level: 0.8
  notes: "trabaja en security, entiende CVEs pero no profundizÃ³ en SBOM"
```

Y ajusta:

- **expert** â†’ no explicar basics, ir directo al detalle tÃ©cnico
- **familiar** â†’ asumir conocimiento medio, profundizar donde hay gaps
- **exposed** â†’ explicar contexto pero no desde cero
- **unknown** â†’ explicar desde cero, contextualizar
- **interested** â†’ ofrecer profundizaciÃ³n, fuentes adicionales
- **avoid** â†’ no insistir con el tema salvo que el usuario lo pida
- **misconception** â†’ corregir con evidencia, no ignorar

### Inferencia del user model

El LLM **no infiere el user model solo**. La inferencia se hace con evidencia persistente:

1. **AutomÃ¡tica**: despuÃ©s de cada conversaciÃ³n, un job analiza los episodios y actualiza `user_topic_records` basÃ¡ndose en `user_evidence`
2. **Semi-automÃ¡tica**: el LLM propone un estado, Valen aprueba o corrige
3. **Manual**: Valen edita su perfil directamente desde el dashboard

La confianza numÃ©rica (`expertise_level`, `interest_level`) solo se actualiza cuando hay suficiente evidencia â€” no es un valor inventado por el LLM.

### GeneralizaciÃ³n del learner model

El `TUTOR_AGENT_DESIGN.md` define:

| Learner model (Tutor) | User model (Personal AGI) |
|---|---|
| `MasteryRecord` | `UserTopicRecord` |
| `LearningGoal` | `Goal` (cualquier objetivo, no solo pedagÃ³gico) |
| `AssessmentAttempt` | `EvidenceRecord` (cualquier evidencia, no solo evaluaciones) |
| `understood` | `familiar` |
| `applied` | `practiced` |
| `mastered` | `expert` |
| `misconception` | `misconception` (sin cambio) |

El Tutor Agent pasa a ser **un modo del Personal AGI** que usa el user model con policies pedagÃ³gicas. No tiene su propio learner model separado â€” usa el mismo `user_topic_records` con interpretaciÃ³n orientada a enseÃ±anza.

### Perfil global del usuario

AdemÃ¡s del estado por tÃ³pico, hay un perfil global que no depende de tÃ³picos especÃ­ficos:

```sql
-- Perfil global del usuario (no vinculado a tÃ³picos)
CREATE TABLE user_profile (
    key         TEXT PRIMARY KEY,             -- ej: "profession", "language_preference", "communication_style"
    value       TEXT NOT NULL,                -- ej: "security analyst", "espaÃ±ol", "directo tÃ©cnico"
    confidence  REAL DEFAULT 0.5,
    source      TEXT NOT NULL,                -- user_statement / inferred / manual
    updated_at  TEXT NOT NULL
);
```

Esto guarda cosas como:

- ProfesiÃ³n / Ã¡rea de trabajo
- Idioma preferido
- Estilo de comunicaciÃ³n (directo, tÃ©cnico, conversacional)
- Zona horaria
- Preferencias de profundidad
- Cosas que el usuario dijo explÃ­citamente sobre sÃ­ mismo

### Archivos nuevos

- `src/ipa/agentic/user_model.py` â€” CRUD de `user_topic_records`, `user_evidence`, `user_profile`
- `src/ipa/agentic/user_model_inference.py` â€” inferencia del user model desde episodios

### Archivos a modificar

- `src/ipa/reporter/reporter_deep_dive.py` â€” consultar user model antes de responder
- `src/ipa/agentic/agent_memory.py` â€” despuÃ©s de consolidar episodios, actualizar user model
- `scripts/web_dashboard.py` â€” endpoint para ver/editar el user model
- `web/static/app.js` â€” panel de perfil de usuario

---

## DivisiÃ³n de tecnologÃ­as

### LanceDB (vectorial)

```
Tabla: chunks (ya existe + columnas nuevas)
  vector[1024]           â† BGE-M3 (ya existe)
  document_id            â† (ya existe)
  chunk_id               â† (ya existe)
  text                   â† (ya existe)
  sparse_json            â† (ya existe)
  topic_cluster_id       â† NUEVO: tÃ³pico al que pertenece
  parent_category_id     â† NUEVO: categorÃ­a padre
  temporal_bucket        â† NUEVO: semana/mes para evoluciÃ³n

Tabla: topic_centroids (NUEVA â€” solo vectores)
  cluster_id             â† PK
  centroid_vector[1024]  â† centroide del tÃ³pico para bÃºsqueda vectorial
  label                  â† label del tÃ³pico
  parent_category_id     â† categorÃ­a padre
  doc_count              â† cuÃ¡ntos docs
  chunk_count            â† cuÃ¡ntos chunks
```

### SQLite (relacional â€” tablas existentes + nuevas)

```
Tablas existentes (se mantienen):
  document_store, scrape_history, dashboard, reporter decisions

Tablas nuevas (memoria episÃ³dica):
  agent_episodes         â† registro crudo de conversaciones
  agent_memory_extracts  â† memoria consolidada

Tablas nuevas (user model):
  user_topic_records     â† estado del usuario por tÃ³pico
  user_evidence          â† evidencia que respalda el user model
  user_profile           â† perfil global (profesiÃ³n, idioma, estilo)

Tablas nuevas (jerÃ¡rquicas â€” mientras PG no se necesite):
  topic_clusters         â† metadata de tÃ³picos
  topic_links            â† grafo de relaciones
  topic_evolution        â† tracking temporal
```

### PostgreSQL (relacional â€” migraciÃ³n futura cuando se necesite escala)

```
Migrar cuando:
  - topic_clusters supere 10k registros
  - agent_episodes supere 50k registros
  - user_topic_records supere 5k registros
  - Se necesite acceso concurrente multi-agente

Las tablas existentes (document_store, scrape_history, dashboard)
se mantienen en SQLite. Solo las tablas jerÃ¡rquicas y de memoria
migran a PostgreSQL.
```

| Aspecto | SQLite | PostgreSQL |
|---|---|---|
| Recursive CTEs (jerarquÃ­a) | âœ… pero lento | âœ… optimizado |
| Multi-hop graph traversal | âš ï¸ sin Ã­ndices | âœ… con Ã­ndices |
| Multi-agente concurrente | âŒ 1 writer | âœ… MVCC |
| JSONB metadata | âŒ JSON text | âœ… indexado |
| Escala 100k+ tÃ³picos | âš ï¸ se degrada | âœ… sin problema |

---

## ImplementaciÃ³n por etapas

### Etapa 1: Identidad persistente

**Archivos nuevos:**

- `configs/agent_identity.yaml` â€” identidad base del Personal AGI
- `src/ipa/agentic/agent_identity.py` â€” loader de identidad

**Archivos a modificar:**

- `src/ipa/reporter/reporter_deep_dive.py` â€” cargar identidad base en lugar de prompt hardcodeado

**QuÃ© hace:**

1. Define la identidad del agente en un archivo YAML compartido
2. `agent_identity.py` carga el YAML y construye el system prompt base
3. El Deep Dive usa `load_identity()` en lugar del string hardcodeado
4. El Reporter NO cambia â€” mantiene su prompt estricto

**ValidaciÃ³n:**

- El Deep Dive responde con la personalidad definida en el YAML
- Cambiar el YAML cambia el comportamiento sin tocar cÃ³digo
- El Reporter sigue siendo estricto

### Etapa 2: Memoria episÃ³dica

**Archivos nuevos:**

- `src/ipa/agentic/agent_memory.py` â€” CRUD de episodios y extractos

**Archivos a modificar:**

- `src/ipa/reporter/reporter_deep_dive.py` â€” escribir episodios despuÃ©s de cada turno, cargar memoria al iniciar
- `scripts/web_dashboard.py` â€” persistir turnos en agent_episodes, pasar history a deep_dive

**QuÃ© hace:**

1. Crea tablas `agent_episodes` y `agent_memory_extracts` en SQLite
2. DespuÃ©s de cada turno (user + assistant), escribe un episodio
3. Al iniciar una sesiÃ³n, carga episodios previos relevantes al tÃ³pico de la query
4. `agent_memory_extracts` se llena manualmente o semi-automÃ¡ticamente por ahora

**ValidaciÃ³n:**

- DespuÃ©s de una conversaciÃ³n sobre X, iniciar una nueva sesiÃ³n y preguntar sobre X â†’ el agente recuerda
- Los episodios se vinculan a topic_cluster_id cuando es posible
- La tabla no crece infinitamente porque los extractos consolidan

### Etapa 3: Clustering jerÃ¡rquico sobre centroides existentes

**Archivos a modificar:**

- `src/ipa/indexes/lancedb_index.py` â€” agregar `_compute_topic_clusters()`, `document_embeddings()` ya existe
- `src/ipa/storage/document_store.py` â€” agregar tabla `document_topic_assignments`
- `scripts/cli/run_fast_path.py` â€” llamar `_compute_topic_clusters()` despuÃ©s de `_compute_centroids()`

**QuÃ© hace:**

1. Toma los centroides ya computados por `_compute_centroids()`
2. Hace agglomerative clustering (scipy `linkage`) sobre los vectores centroide
3. Threshold de similitud â†’ define tÃ³picos
4. Guarda `topic_cluster_id` en cada chunk de LanceDB
5. Computa centroides de tÃ³picos â†’ guarda en tabla `topic_centroids` en LanceDB

**ValidaciÃ³n:**

- `_compute_topic_clusters()` produce cluster_ids consistentes para chunks del mismo tÃ³pico
- `document_embeddings()` ya funciona (implementado en etapa anterior)

### Etapa 4: CategorÃ­as padre y grafo de tÃ³picos

**Archivos a modificar:**

- `src/ipa/reporter/reporter_topics.py` â€” `group_topics_into_categories()` usa centroides en vez de LLM
- `src/ipa/indexes/lancedb_index.py` â€” agregar `_compute_parent_categories()`
- `src/ipa/reporter/reporter_pipeline.py` â€” integrar categorÃ­as jerÃ¡rquicas

**QuÃ© hace:**

1. Segundo nivel de clustering sobre centroides de tÃ³picos â†’ categorÃ­as padre
2. Guarda `parent_category_id` en cada chunk de LanceDB
3. LLM solo se usa para labelar las 3-8 categorÃ­as padre (no los 25 tÃ³picos)
4. Sub-tÃ³picos usan `_fallback_label()` determinÃ­stico + label heredado del padre
5. Crea tabla `topic_links` con relaciones entre tÃ³picos

**ValidaciÃ³n:**

- `parent_category_id` agrupa tÃ³picos en 3-8 categorÃ­as coherentes
- Ninguna categorÃ­a contiene mÃ¡s del 50% de los tÃ³picos

### Etapa 5: MCP Tools + Planner ReAct

**Archivos nuevos:**

- `src/ipa/agentic/agent_tools.py` â€” implementaciÃ³n de las 5 tools
- `src/ipa/agentic/agent_planner.py` â€” planner ReAct con state tracking
- `src/ipa/agentic/agent_cleanup.py` â€” output cleanup post-generaciÃ³n

**Archivos a modificar:**

- `src/ipa/mcp/mcp_server.py` â€” registrar las 5 tools
- `src/ipa/reporter/reporter_deep_dive.py` â€” integrar planner + cleanup
- `src/ipa/agentic/reporter_planner.py` â€” expandir con plan_query, execute_step, build_context
- `src/ipa/agentic/reporter_retrieval.py` â€” soporte para filtros por `topic_cluster_id`

**QuÃ© hace:**

1. Implementa las 5 tools: `search_corpus`, `get_topic_info`, `list_topics`, `recall_conversation`, `get_related_topics`
2. El planner descompone queries complejas en sub-queries
3. Ejecuta cada sub-query contra el retrieval o las tools
4. Mantiene estado de quÃ© se ejecutÃ³ y quÃ© devolviÃ³
5. Al final, construye un contexto estructurado y se lo pasa al LLM
6. DespuÃ©s de generar, aplica output cleanup

**ValidaciÃ³n:**

- Query multi-hop: "Â¿CÃ³mo se relaciona X con Y?" â†’ planner ejecuta 3-4 steps â†’ LLM sintetiza
- Tool calling: el LLM decide llamar `search_corpus` â†’ planner ejecuta â†’ LLM usa el resultado
- Output cleanup: detecta idioma mezclado y lo limpia

### Etapa 6: Deep dive con navegaciÃ³n jerÃ¡rquica

**Archivos a modificar:**

- `src/ipa/reporter/reporter_deep_dive.py` â€” navegaciÃ³n horizontal y multi-hop
- `src/ipa/agentic/reporter_retrieval.py` â€” filtros por `topic_cluster_id` y `parent_category_id`
- `src/ipa/agentic/reporter_planner.py` â€” planear queries multi-hop

**QuÃ© hace:**

1. DespuÃ©s del retrieval lÃ©xico + vectorial, obtiene `topic_cluster_id` de los hits
2. NavegaciÃ³n horizontal: `WHERE topic_cluster_id = X` â†’ chunks del mismo tÃ³pico
3. Drill-down: `WHERE parent_category_id = X` â†’ categorÃ­a completa
4. Multi-hop: `topic_links` â†’ tÃ³picos relacionados â†’ mÃ¡s chunks
5. EvoluciÃ³n temporal: `temporal_bucket` â†’ cÃ³mo cambiÃ³ el tÃ³pico

**ValidaciÃ³n:**

- Deep dive encuentra chunks relacionados vÃ­a navegaciÃ³n jerÃ¡rquica
- Multi-hop: tÃ³pico A â†’ relacionado B â†’ chunks de B aparecen en la respuesta

### Etapa 7: EvoluciÃ³n temporal y continuidad

**Archivos a modificar:**

- `src/ipa/reporter/reporter_topics.py` â€” `match_topic_continuity` usa `cluster_id` en vez de similitud lÃ©xica
- `src/ipa/reporter/reporter_store.py` â€” tabla `topic_evolution`

**QuÃ© hace:**

1. Cuando un tÃ³pico ya existe de runs anteriores, reusa el `cluster_id`
2. Trackea `first_seen`, `last_seen`, `evolution` (new/continuing/merged/split)
3. El LLM solo labela tÃ³picos nuevos
4. TÃ³picos existentes heredan label + description del run anterior

**ValidaciÃ³n:**

- TÃ³picos existentes se reconocen y reusan label en runs siguientes
- `topic_evolution` trackea cambios entre perÃ­odos

### Etapa 8: ConsolidaciÃ³n de memoria episÃ³dica

**Archivos a modificar:**

- `src/ipa/agentic/agent_memory.py` â€” funciÃ³n `consolidate_episodes()`
- `scripts/run_memory_consolidation.py` â€” NUEVO, job periÃ³dico

**QuÃ© hace:**

1. Selecciona episodios no consolidados agrupados por `topic_cluster_id`
2. Para cada grupo, manda al LLM: "LeÃ© estos N episodios sobre el tÃ³pico X y extraÃ© los puntos clave"
3. Guarda el resumen en `agent_memory_extracts`
4. Marca los episodios como consolidados

**Honestidad**: el 9B 3.0bpw no va a producir extractos de alta calidad. Esta etapa es para validar el flujo â€” la calidad mejora cuando se integre un modelo mÃ¡s capaz. Mientras tanto, la consolidaciÃ³n puede ser manual o semi-automÃ¡tica.

**ValidaciÃ³n:**

- DespuÃ©s de consolidar, `recall_conversation` devuelve el extracto en lugar de los episodios crudos
- Los episodios consolidados se marcan como tales

### Etapa 9: User Model â€” perfil del usuario

**Archivos nuevos:**

- `src/ipa/agentic/user_model.py` â€” CRUD de `user_topic_records`, `user_evidence`, `user_profile`
- `src/ipa/agentic/user_model_inference.py` â€” inferencia del user model desde episodios consolidados

**Archivos a modificar:**

- `src/ipa/reporter/reporter_deep_dive.py` â€” consultar user model antes de responder, ajustar nivel de explicaciÃ³n
- `src/ipa/agentic/agent_memory.py` â€” despuÃ©s de consolidar episodios, disparar inferencia del user model
- `scripts/web_dashboard.py` â€” endpoint para ver/editar el user model
- `web/static/app.js` â€” panel de perfil de usuario

**QuÃ© hace:**

1. Crea tablas `user_topic_records`, `user_evidence`, `user_profile` en SQLite
2. DespuÃ©s de cada conversaciÃ³n, registra evidencia en `user_evidence`
3. Un job periÃ³dico (o manual) infiere el `user_status` por tÃ³pico desde la evidencia acumulada
4. El agente consulta el user model antes de responder y ajusta el nivel de explicaciÃ³n
5. Valen puede ver y editar su perfil desde el dashboard

**CÃ³mo se ajusta la respuesta segÃºn el user model:**

- `expert` â†’ no explicar basics, ir directo al detalle tÃ©cnico
- `familiar` â†’ asumir conocimiento medio, profundizar donde hay gaps
- `exposed` â†’ explicar contexto pero no desde cero
- `unknown` â†’ explicar desde cero, contextualizar
- `interested` â†’ ofrecer profundizaciÃ³n, fuentes adicionales
- `avoid` â†’ no insistir con el tema salvo que el usuario lo pida
- `misconception` â†’ corregir con evidencia, no ignorar

**Inferencia semi-automÃ¡tica**: el LLM propone un estado basÃ¡ndose en evidencia, Valen aprueba o corrige. La confianza numÃ©rica solo se actualiza con suficiente evidencia â€” no es un valor inventado.

**ValidaciÃ³n:**

- DespuÃ©s de conversar sobre un tÃ³pico N veces, el `user_topic_records` refleja el estado correcto
- El agente ajusta el nivel de explicaciÃ³n segÃºn el `user_status`
- Valen puede editar su perfil desde el dashboard
- El Tutor Agent usa el mismo user model (no tiene learner model separado)

### Etapa 10: PostgreSQL (migraciÃ³n futura)

**Archivos nuevos:**

- `src/ipa/storage/pg_store.py` â€” adapter PostgreSQL para `topic_clusters`, `topic_links`, `topic_evolution`, `agent_episodes`, `agent_memory_extracts`, `user_topic_records`, `user_evidence`, `user_profile`
- `configs/pg.yaml` â€” configuraciÃ³n de conexiÃ³n

**Archivos a modificar:**

- `src/ipa/reporter/reporter_store.py` â€” opcionalmente migrar tablas jerÃ¡rquicas a PG
- `src/ipa/agentic/agent_memory.py` â€” soporte dual SQLite/PG

**QuÃ© hace:**

1. Crear tablas en PostgreSQL
2. Adapter `pg_store.py` con misma interfaz que los adapters SQLite
3. Queries recursivas para jerarquÃ­a y multi-hop
4. MigraciÃ³n gradual: lo nuevo va a PG, lo existente se mantiene en SQLite

**CuÃ¡ndo migrar:**

- `topic_clusters` > 10k registros
- `agent_episodes` > 50k registros
- Se necesite acceso concurrente multi-agente

**ValidaciÃ³n:**

- PostgreSQL responde queries recursivas (jerarquÃ­a) y multi-hop (graph traversal)
- MigraciÃ³n no pierde datos

---

## Resiliencia del Reporter frente a timeouts

Las llamadas al LLM no se ejecutan como un Ãºnico batch monolÃ­tico. Se procesan en lotes independientes y cada lote se persiste al terminar:

```text
Lote normal (curaciÃ³n: 2 documentos; labels: 2 tÃ³picos)
    â†“ timeout/error
Reintento dividido a la mitad
    â†“ si vuelve a fallar
Procesamiento tÃ³pico/documento por tÃ³pico
    â†“ si falla el elemento individual
Fallback determinÃ­stico solo para ese elemento
```

DespuÃ©s de cada lote se ejecuta `reset_generator()` para limpiar el estado del generador y evitar que un timeout invalide toda la fase. La respuesta se valida por cardinalidad, JSON vÃ¡lido y correspondencia con los elementos de entrada.

Para categorÃ­as padre se aplican validaciones estructurales:

- entre 3 y 8 categorÃ­as cuando hay mÃ¡s de 12 tÃ³picos;
- ninguna categorÃ­a puede contener mÃ¡s del 50% de los tÃ³picos;
- si el resultado no cumple, se usa el agrupamiento determinÃ­stico seguro;
- el fallback evita keywords genÃ©ricas para no producir mega-categorÃ­as.

Esto permite que un fallo parcial degrade Ãºnicamente un lote o tÃ³pico, en lugar de perder el reporte completo.

---

## Honestidad sobre el modelo actual (Qwen3.5-9B EXL3 3.0bpw)

### Lo que puede

- Recordar quiÃ©n sos y quÃ© te interesa â€” es un system prompt, no razonamiento
- Tool calling bÃ¡sico â€” Qwen3.5 estÃ¡ entrenado para function calling, incluso cuantizado a 3.0bpw mantiene la estructura JSON
- Recuperar memoria previa y usarla como contexto â€” siempre que la memoria estÃ© bien estructurada, el modelo solo necesita leerla
- SÃ­ntesis de 2-3 fragmentos de evidencia â€” hasta ahÃ­ llega bien
- MCP simple â€” 3-5 herramientas con schemas claros

### Lo que no puede bien (y cÃ³mo lo mitigamos)

| LimitaciÃ³n | MitigaciÃ³n |
|---|---|
| Multi-hop complejo | Planner determinÃ­stico que descompone y ejecuta los hops |
| Conversaciones largas | Output cleanup + reducir contexto a 4 turnos |
| Tool calling encadenado | Planner mantiene estado, LLM solo decide prÃ³ximo paso |
| ConsolidaciÃ³n de memoria | Estructura lista, calidad mejora con modelo mejor. Mientras tanto: manual o semi-auto |
| Think mode off | Planner externo suple el razonamiento que el modelo no hace internamente |

### El techo real

La arquitectura no depende del modelo. Las tablas de memoria, la identidad, el MCP, la jerarquÃ­a de tÃ³picos, el planner â€” todo eso es estructura. Cuando se integre un modelo mÃ¡s capaz (32B, 70B, o lo que venga), todo ya estÃ¡ listo y el agente mejora sin reescribir nada.

El 9B 3.0bpw es suficiente para **validar que la arquitectura funciona**. No para tener un agente de alta calidad, pero sÃ­ para tener uno mÃ­nimamente capaz que te conoce, recuerda de quÃ© hablaron, sabe buscar en la base, y puede ejecutar herramientas bÃ¡sicas.

La pregunta no es "Â¿es buen agente?" sino "Â¿la arquitectura estÃ¡ bien diseÃ±ada para que cuando pongas un modelo mejor ahÃ­, todo mejore automÃ¡ticamente?" Si la respuesta es sÃ­, vale la pena construirla ahora con el 9B como validaciÃ³n.

---

## Archivos a crear/modificar (resumen)

### Archivos nuevos

| Archivo | Etapa | PropÃ³sito |
|---|---|---|
| `configs/agent_identity.yaml` | 1 | Identidad base del Personal AGI |
| `src/ipa/agentic/agent_identity.py` | 1 | Loader de identidad + carga de memoria relevante |
| `src/ipa/agentic/agent_memory.py` | 2, 8 | CRUD de episodios y extractos de memoria |
| `src/ipa/agentic/agent_tools.py` | 5 | ImplementaciÃ³n de las 5 MCP tools |
| `src/ipa/agentic/agent_planner.py` | 5 | Planner ReAct con state tracking |
| `src/ipa/agentic/agent_cleanup.py` | 5 | Output cleanup post-generaciÃ³n |
| `src/ipa/agentic/user_model.py` | 9 | CRUD de user_topic_records, user_evidence, user_profile |
| `src/ipa/agentic/user_model_inference.py` | 9 | Inferencia del user model desde episodios |
| `scripts/run_memory_consolidation.py` | 8 | Job periÃ³dico de consolidaciÃ³n |
| `src/ipa/storage/pg_store.py` | 10 | Adapter PostgreSQL |
| `configs/pg.yaml` | 10 | ConfiguraciÃ³n PG |

### Archivos a modificar

| Archivo | Etapa | Cambio |
|---|---|---|
| `src/ipa/reporter/reporter_deep_dive.py` | 1, 2, 5, 6, 9 | Identidad base, memoria episÃ³dica, planner, cleanup, navegaciÃ³n jerÃ¡rquica, user model |
| `scripts/web_dashboard.py` | 2, 9 | Persistir turnos en agent_episodes, endpoint de user model |
| `web/static/app.js` | 9 | Panel de perfil de usuario |
| `src/ipa/indexes/lancedb_index.py` | 3, 4 | `_compute_topic_clusters()`, `_compute_parent_categories()`, columnas nuevas |
| `src/ipa/storage/document_store.py` | 3 | Tabla `document_topic_assignments` |
| `scripts/cli/run_fast_path.py` | 3 | Llamar `_compute_topic_clusters()` despuÃ©s de centroids |
| `src/ipa/reporter/reporter_topics.py` | 4, 7 | `group_topics_into_categories()` usa centroides, `match_topic_continuity` usa cluster_id |
| `src/ipa/reporter/reporter_pipeline.py` | 4 | Integrar categorÃ­as jerÃ¡rquicas |
| `src/ipa/mcp/mcp_server.py` | 5 | Registrar las 5 tools |
| `src/ipa/agentic/reporter_planner.py` | 5, 6 | Expandir con plan_query, execute_step, build_context, multi-hop |
| `src/ipa/agentic/reporter_retrieval.py` | 6 | Filtros por `topic_cluster_id` y `parent_category_id` |
| `src/ipa/reporter/reporter_store.py` | 7, 9 | Tabla `topic_evolution`, migraciÃ³n opcional a PG |

---

## Verification

- [ ] Etapa 1: El Deep Dive usa `agent_identity.yaml` â€” cambiar el YAML cambia el comportamiento
- [ ] Etapa 2: DespuÃ©s de una conversaciÃ³n, iniciar nueva sesiÃ³n y preguntar lo mismo â†’ el agente recuerda
- [ ] Etapa 2: Los episodios se vinculan a `topic_cluster_id` cuando es posible
- [ ] Etapa 3: `_compute_topic_clusters()` produce cluster_ids consistentes para chunks del mismo tÃ³pico
- [ ] Etapa 4: `parent_category_id` agrupa tÃ³picos en 3-8 categorÃ­as coherentes
- [ ] Etapa 5: Query multi-hop â†’ planner ejecuta 3-4 steps â†’ LLM sintetiza respuesta coherente
- [ ] Etapa 5: Tool calling â†’ LLM decide llamar `search_corpus` â†’ planner ejecuta â†’ LLM usa resultado
- [ ] Etapa 5: Output cleanup detecta y limpia idioma mezclado
- [ ] Etapa 6: Deep dive encuentra chunks relacionados vÃ­a navegaciÃ³n jerÃ¡rquica
- [ ] Etapa 7: TÃ³picos existentes se reconocen y reusan label en runs siguientes
- [ ] Etapa 8: DespuÃ©s de consolidar, `recall_conversation` devuelve el extracto en lugar de episodios crudos
- [ ] Etapa 9: DespuÃ©s de conversar sobre un tÃ³pico N veces, `user_topic_records` refleja el estado correcto
- [ ] Etapa 9: El agente ajusta el nivel de explicaciÃ³n segÃºn `user_status` (expert vs unknown)
- [ ] Etapa 9: Valen puede ver y editar su perfil desde el dashboard
- [ ] Etapa 9: El Tutor Agent usa el mismo user model (no tiene learner model separado)
- [ ] Etapa 10: PostgreSQL responde queries recursivas (jerarquÃ­a) y multi-hop (graph traversal)
- [ ] Tests: `test_reporter.py` valida clustering, jerarquÃ­a y continuidad
- [ ] Tests: `test_agent.py` valida identidad, memoria episÃ³dica, tools y planner
- [ ] Tests: `test_user_model.py` valida inferencia de estados, evidencia y ajuste de respuesta
- [ ] Benchmarks: comparar tiempo de inferencia LLM antes/despuÃ©s del tier system
- [ ] Benchmarks: comparar calidad de respuesta con/sin planner en queries multi-hop

---

## Risks / Considerations

- **PostgreSQL dependency:** agregar requerimiento de servidor PG. Etapa 9 es opcional hasta que se necesite escala real. SQLite maneja bien miles de episodios de una persona.
- **Clustering quality:** agglomerative clustering puede producir tÃ³picos poco coherentes si el threshold no estÃ¡ bien calibrado. Necesita tuning.
- **Backward compatibility:** los chunks existentes en LanceDB no tendrÃ¡n `topic_cluster_id`. Hay que re-indexar o hacer migration.
- **LLM timeout:** el `group_topics` con timeout de 300s ya estÃ¡ implementado. El tier system reduce el riesgo pero no lo elimina.
- **Complejidad:** el sistema pasa de 3 Ã­ndices planos a 3 Ã­ndices + jerarquÃ­a + grafo + memoria episÃ³dica + planner + MCP. MÃ¡s potente pero mÃ¡s complejo de mantener.
- **Calidad del 9B 3.0bpw:** la consolidaciÃ³n de memoria y la sÃ­ntesis multi-hop van a ser de calidad limitada. La arquitectura estÃ¡ diseÃ±ada para que esto mejore automÃ¡ticamente con un modelo mejor, sin reescribir nada.
- **Memoria infinita:** sin consolidaciÃ³n, `agent_episodes` crece sin lÃ­mite. La consolidaciÃ³n manual o semi-automÃ¡tica es necesaria desde el inicio, incluso si es de baja calidad.
- **Privacidad:** la memoria episÃ³dica contiene conversaciones personales. Debe vivir localmente, nunca enviarse a servicios externos. PostgreSQL, si se usa, debe ser local.
- **Identidad vs Reporter:** el Reporter mantiene su prompt estricto para simetrÃ­a de reportes. La identidad base solo aplica a interfaces de interacciÃ³n. No mezclar.
