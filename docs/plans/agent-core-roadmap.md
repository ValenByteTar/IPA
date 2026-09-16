# Agent core roadmap — pseudo-tutor + horizontalidad (contract-first)

> Construir el núcleo de un agente personal local omnipresente (identidad + memoria +
> planner + tools + sesiones) con arquitectura por contratos, y montar sobre él el
> pseudo-tutor como primer rol. La horizontalidad (búsqueda horizontal, multi-hop,
> jerarquía de tópicos, web research) enriquece al núcleo por fases, cada una con un
> consumidor real que valida la anterior.

Este plan fusiona y reordena `docs/Horizontalidad.md` (los 5 pilares y 10 etapas),
`docs/plans/agent-runtime-migration.md` (el primer slice ya validado por EXP-003) y
el diseño del Tutor (`docs/architecture/tutor.md`) alrededor de un **núcleo agentivo
omnipresente**, no anclado al dashboard.

---

## Modelo de tres capas

```text
NÚCLEO (ipa/agent/)          — runtime, no UI
  identidad + memoria (episódica + user model) + planner + tools + sesiones
        ↑ abre sesiones
SUPERFICIES (thin clients)
  CLI  |  dashboard  | ventanas futuras
        ↑
ROLES = policy + prompt + scope de estado
  Tutor (primero) · asistente general · (Reporter redacción queda batch/estricto)
```

Reglas del modelo:

- **Omnipresencia es del estado y la identidad, no del deployment**: local-first,
  in-process, una persona, una GPU con slots. Sin servidor de agente con red/auth.
- **Un rol no es solo un prompt**: es un scope de estado. El Tutor dueña del estado
  de aprendizaje; el chat general solo genera episodios; la redacción del Reporter
  ni siquiera es conversacional (prompt estricto, simetría de reportes).
- El dashboard es "a local control room, not a second domain runtime"
  (`docs/architecture/dashboard.md`) — deja de ser el ancla del agente.

## Disciplina contract-first

Los contratos en `contracts/` son autoridad; el runtime es un adapter. Cada fase
arranca congelando sus schemas (jsonschema + `contract_vocabulary` + validador en
`scripts/validation/`), luego implementa el runtime, y los tests validan comportamiento
real contra schema. Los contratos evolucionan por `supersedes`, nunca mutando.

**Regla anti-parálisis**: un contrato se congela solo cuando su consumidor de la
misma fase existe. Nada de big-design-up-front para fases futuras.

### Inventario de contratos

| Contract | Estado | Fase |
|---|---|---|
| `learning_goal`, `concept`, `roadmap`, `assessment_result` | existe | 2 |
| `research_request` | existe | 1 |
| `topic_link` | existe | 3 |
| `tutor_common` (provenance + approval) | existe | todas |
| `agent_common` (provenance agentiva, session refs) | **nuevo** | 0 |
| `agent_session` (turnos, interface, rol activo) | **nuevo** | 0 |
| `agent_episode` (episodio crudo) | **nuevo** | 0 |
| `tool_call` / `tool_result` | **nuevo** | 1 |
| `web_source` (source_url, fetched_at, content_hash, license, trust_label) | **nuevo** | 1 |
| `user_topic_record` / `user_evidence` | **nuevo** | 2 |
| `topic_cluster` (jerarquía, padre, centroides) | **nuevo** | 3 |

---

## Fase 0 — Núcleo del agente (identidad + sesiones + memoria)

**Contratos a congelar**: `agent_common`, `agent_session`, `agent_episode`.

- DEC-002: ownership del user model unificado y boundaries del agent core
  (cierra el gap de RES-002).
- `ipa/agent/`: identity loader (`configs/agent_identity.yaml`) + session manager
  (sesiones durables, campo `interface`) + memoria episódica (episodios +
  extractos manuales; consolidación automática NO en esta fase).
- Superficie CLI fina (`scripts/cli/agent.py`): la misma sesión, el mismo recuerdo.

**Gate**:

- [ ] la sesión sobrevive entre procesos (CLI ↔ CLI y CLI ↔ proceso del dashboard);
- [ ] cambiar el YAML cambia el comportamiento sin tocar código;
- [ ] episodios vinculados a `topic_cluster_id` cuando existe (nullable hasta Fase 3).

## Fase 1 — Agencia: tools + web research (temprano)

**Contratos a congelar**: `tool_call`, `tool_result`, `web_source`.
(`research_request` ya existe.)

- Tools determinísticas en el núcleo, sin LLM: `search_corpus`,
  `list_topics`/`get_topic_info`, `recall_conversation` — queries sobre
  store/topics/episodios que emiten `tool_call`/`tool_result`.
- `research_topic` (ejecuta el contract `research_request` existente):
  `search_web` (SearXNG local o DuckDuckGo, sin nube) → scraper existente
  (E4: RSS/Playwright/arxiv/OCR) → Landing → ingestion → retrieval con citas.
  Acotado: presupuesto de URLs/tiempo; provenance obligatoria vía `web_source`
  (`source_url`, `fetched_at`, `content_hash`, license, trust_label).
  Regla PAT-003: lo traído de la web es representación derivada etiquetada,
  **nunca autoridad canónica**.
- Dashboard como segunda superficie: el Deep Dive migrado a sesión del agente
  (el streaming ya existe; el núcleo expone la misma interfaz).

**Gates**:

- test de omnipresencia: CLI ↔ dashboard comparten memoria de sesión;
- un `ResearchRequest` end-to-end con citas verificables (`web_source` hashes).

## Fase 2 — Pseudo-Tutor v1 (el primer rol)

**Contratos congelados**: `user_topic_record`, `user_evidence`.
(los pedagógicos ya existen: `learning_goal`, `concept`, `roadmap`,
`assessment_result`). Los consumidores actuales son `TutorStore` y
`TutorSession` en `src/ipa/tutor/tutor_runtime.py`.

- **Rol Tutor**: persona pedagógica (extensión de la identidad base) + scope de
  estado propio. La redacción del Reporter NO es un rol conversacional: mantiene
  su prompt estricto por simetría de reportes.
- **Diagnóstico**: desde `user_topic_records` + episodios — andamiaje
  determinístico; el LLM solo clasifica con JSON estructurado (validez 0.90
  medida en BM-006). El diagnóstico es la skill débil del 9B (0.67-0.70): el
  andamiaje es la compensación, no el prompt.
- **Roadmap**: ✅ implementado — `TutorSession.propose_roadmap()` (LLM propone
  3-7 unidades desde conceptos disponibles, el andamiaje valida y da forma de
  contrato) → `approve_roadmap()`/`reject_roadmap()` (gate humano explícito) →
  `activate_roadmap()` (solo desde approved). Versionado por `supersedes`
  (versión >1 exige `previous_roadmap_id` + `change_reason`). Verificado con
  el modelo real: propuesta de 5 unidades con reasons pedagógicas, gate
  proposed→active bloqueado sin aprobación, contrato Roadmap schema-validado.
- **Lección**: sesión del agente con policy pedagógica; si el corpus no alcanza
  → `ResearchRequest` web (Fase 1).
- **Assessment**: structured JSON + abstención (0.81-0.89 medidos). Modo
  independent — la deliberación está descartada por evidencia (daño neto).
- **Mastery update**: `user_topic_records` + `user_evidence` — el assessment ES
  la evidencia. Sin learner model paralelo: el user model unificado es el
  mastery store (DEC-002).

**Estado actual**: loop funcional verificado con el modelo estrella
(diagnóstico determinístico → lección con policy → assessment JSON con
abstención → mastery update trazable) y 536 tests.

**Gate formal: CUMPLIDO (2026-09-08)**. Corrida `tutor-fase2-qwen35-9b-3.0`
contra `pedagogical_v1`: 720/720 generaciones, 0 failures, 0 OOM,
quality 0.9124, diagnosis_accuracy 0.6628 (rango 9B), next_step_accuracy 0.75
(rubrica t12 corregida), json_valid_rate 0.9028, abstention_accuracy 0.8125.
Evidencia: `EXP-004` (sección Gate Fase 2) y
`small-model-deliberation/engine_benchmark/results/tutor-fase2-qwen35-9b-3.0/`.

## Fase 3 — Horizontalidad profunda (enriquece al núcleo)

**Contratos congelados**: `topic_cluster` (consumidor: TopicClusterStore +
TopicNavigator + MemoryConsolidator, todos en la misma fase). (`topic_link`
ya existe.)

- **Clustering**: ✅ implementado — agglomerative determinístico sobre
  centroides BGE-M3 existentes, threshold-based (sin K fijo, mismo enfoque
  validado en Reporter discover_topics). Sobre el corpus E12: 103 docs →
  9 clusters emergentes con jerarquía padre/hijo, coherencia 0.72-0.96,
  contratos schema-validados. Store: `outputs/agent/topic_clusters.db`.
- **Navegación horizontal + multi-hop**: ✅ implementado — `TopicNavigator`
  con vertical-first, cobertura medida por **diversidad de documentos** (no
  conteo de chunks), máximo 2 hops, presupuesto por hop (PAT-004), trace
  auditable por hop.
- **Gate experimental multi-hop vs vertical**: ✅ medido
  (`outputs/experiments/E13-multihop-vs-vertical.json`) — recall promedio
  vertical 0.290 vs multi-hop 0.306 (Δ +0.016) en 3 queries del corpus real;
  multi-hop aporta recall adicional a costo de latencia despreciable (~80ms).
  La ganancia es modesta en este corpus (los clusters son pequeños); el
  mecanismo queda disponible y medido, su valor crece con el corpus.
- **Consolidación de memoria (etapa 8)**: ✅ implementado —
  `MemoryConsolidator` propone consolidados de episodios (nunca borra
  originales, append-only), `UserModelInference` propone mastery updates
  desde evidencia acumulada (nunca propone regresión). Ambos quedan
  `pending` hasta aprobación humana (invariante:
  `memory_consolidation_requires_human_approval`).
- **Inferencia del user model (etapa 9)**: ✅ implementada como
  `UserModelInference` — job determinístico que propone actualizaciones de
  `user_topic_records` desde evidencia acumulada; Valen aprueba o corrige.

**Gates**:
- [x] multi-hop vs vertical medido (E13, recall +1.6pp promedio)
- [x] memoria vs sin memoria medido: el mastery context cambia estructuralmente
  el payload de lección (estado "desconocido" → "applied (score 0.9, N intentos,
  N evidencias)") — verificado con el modelo estrella real
- [x] planner determinístico vs sin planner medido (E13-planner): Δ=+0.000 en
  este corpus — los clusters son pequeños (2-34 docs) y el vertical ya cubre
  los docs relevantes. El mecanismo queda disponible y medido; su valor crece
  con el corpus. No se promueve como garantía de producción sin corpus mayor.

## Fase 4 — Escala

- PostgreSQL solo al superar umbrales (etapa 10: topic_clusters > 10k,
  agent_episodes > 50k, o concurrencia multi-agente real).
- Stage 6 (validación end-to-end) y Stage 7 (hardening) del research roadmap.

## Lo que NO entra

- ReAct abierto, loops sin presupuesto (PAT-004: budgets y trazabilidad por loop).
- Graph database (SQLite alcanza de sobra para una persona; PG solo por umbrales).
- Deliberación multi-agente (evidencia: daño neto -3/+3 con damage real).
- Multi-usuario, red, auth: local-first, una persona, una GPU con slots.
- Consolidación automática de memoria sin aprobación humana.

## Lógica de fondo

Cada fase tiene un consumidor real que valida la anterior: las tools de Fase 1
son las que el Tutor usa; la web research es lo que hace al Tutor útil con un
corpus chico; la jerarquía de Fase 3 llega cuando ya hay sesiones reales que
navegar; la consolidación llega cuando hay episodios que consolidar.

## Verification

- [x] Fase 0: sesión sobrevive entre procesos; identidad desde YAML cambia el
  comportamiento sin tocar código
- [x] Fase 0: episodios escritos desde CLI y dashboard comparten store
- [x] Fase 0: episodios enlazados automáticamente a `topic_cluster_id` cuando existe
  (TutorSession.lesson con cluster_store; nullable cuando no hay cluster)
- [x] Fase 1: `tool_call`/`tool_result` schema-validados en cada ejecución de tool
- [x] Fase 1: `ResearchRequest` end-to-end con `web_source` hashes verificables
  y approval gate runtime completo (TutorSession: create → approve/reject
  humano → execute via Fase 1 executor; contract ResearchRequest schema-validado)
- [x] Fase 1: omnipresencia CLI ↔ dashboard (misma sesión, mismo recuerdo)
- [x] Fase 2: contratos `user_topic_record` + `user_evidence` congelados con
  consumidor (TutorStore) en la misma fase
- [x] Fase 2: loop Tutor completo (diagnóstico → roadmap aprobado → lección →
  assessment → mastery update) medido contra `pedagogical_v1`:
  720/720 generaciones, 0 failures, diagnosis 0.6628, next_step 0.75,
  json 0.9028, abstention 0.8125 (run `tutor-fase2-qwen35-9b-3.0`, EXP-004)
- [x] Fase 2: el assessment actualiza `user_topic_records` con evidencia trazable
- [x] Fase 3: multi-hop vs vertical medido (recall/citas/latencia), no asumido:
  experimento E13 sobre corpus E12 real — multi-hop +1.6pp recall promedio
  (0.290 → 0.306) con latencia despreciable (~80ms), cobertura por diversidad
  de documentos; ganancia modesta pero positiva en queries de cobertura estrecha
- [x] Fase 3: consolidación semi-automática con aprobación humana funcional
  (MemoryConsolidator + UserModelInference, propuestas pending, nunca borra)
- [x] Tests: contratos nuevos validados en `tests/` contra `contract_vocabulary`

## Risks / Considerations

- **Over-contracting**: congelar schemas sin consumidor congela abstracciones
  equivocadas. Regla: contrato solo con consumidor de la misma fase; evolución
  por `supersedes`, nunca mutación.
- **Calidad del 9B**: consolidación de memoria e inferencia del user model serán
  burdas; la estructura está diseñada para mejorar con un modelo mejor sin
  reescribir (DEC-001: el provider es reemplazable, los contratos no cambian).
- **VRAM compartida**: múltiples superficies golpeando un modelo de 6 GB usa los
  slots de recurso del orquestador (`--resource gpu`).
- **Privacidad**: episodios y user model contienen conversaciones personales;
  viven localmente, nunca salen. PostgreSQL, si llega, es local.
- **Scope del rol**: sin scope de estado por rol hay sangrado (episodios de
  tutoría en curación de reportes). El rol es policy + scope, no solo prompt.
- **Backward compatibility**: chunks existentes sin `topic_cluster_id` requieren
  re-indexación parcial en Fase 3.

## Relación con otros planes

- `docs/Horizontalidad.md` — fuente de los pilares; este plan reordena sus
  etapas alrededor del núcleo agentivo (no del dashboard) y le agrega la capa
  contract-first y la web research temprana.
- `docs/plans/agent-runtime-migration.md` — el primer slice
  (QueryIR → EvidenceSet → ContextPackage) ya validado por EXP-003 es la base
  del núcleo; este plan lo extiende con sesiones, roles y memoria.
- `docs/plans/orchestrator-improvements.md` — slots de recurso y disciplina de
  procesos reutilizados para la VRAM compartida entre superficies.
- `knowledge/decisions/DEC-002` (a crear en Fase 0) — ownership del user model
  unificado y boundaries del agent core; cierra el gap de RES-002.
