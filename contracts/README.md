# Contracts

Estos contratos son la autoridad del laboratorio. Los adapters externos deben
convertir sus resultados a estas formas sin filtrar sus propios modelos al
resto del sistema.

## Required records

```text
ArtifactRef
ProcessingManifest
ParserResult
CanonicalDocument
DocumentChunk
SourceSpan
EmbeddingRecord
SearchHit
KnowledgeDelta
LearningGoal
Concept
Roadmap
AssessmentResult
ResearchRequest
ReporterReport
ReporterDocumentDecision
TopicLink
```

Todos los records deben conservar identidad, versión y provenance. Las
implementaciones concretas pueden cambiar; los contratos no se cambian para
acomodar una herramienta antes de completar su competencia.

## Tutor Agent contracts

Los cinco contratos del Tutor Agent usan JSON Schema Draft 2020-12:

| Record | Schema | Responsabilidad |
|---|---|---|
| `LearningGoal` | `learning_goal.schema.json` | Objetivo definido o confirmado por el usuario |
| `Concept` | `concept.schema.json` | Concepto enseñable con fuentes y criterios de dominio |
| `Roadmap` | `roadmap.schema.json` | Plan versionado de 3–7 unidades, sujeto a aprobación |
| `AssessmentResult` | `assessment_result.schema.json` | Evaluación estructurada contra rúbrica y evidencia |
| `ResearchRequest` | `research_request.schema.json` | Investigación acotada nacida de un gap explícito |

`tutor_common.schema.json` contiene las definiciones compartidas: identidad,
hash, timestamp, referencias a fuentes, provenance de generación, aprobación
humana y estados de dominio.

Cada schema declara `x-field-origin` para indicar el origen normativo de cada
campo. Cada instancia lleva `field_origins` para registrar el origen efectivo:
`source`, `user`, `generated`, `system` o `mixed`. Los campos generados o mixtos
requieren `generation` con modelo, prompt, timestamp e input hash. Los campos
respaldados por corpus llevan `source_refs` sin sustituir el texto canónico.

Estados de dominio iniciales:

```text
unknown → exposed → understood → applied
                         ├── needs_review
                         └── misconception
```

No son una progresión automática. `needs_review` y `misconception` se asignan
solo con evidencia de evaluación.

Reglas human-in-the-loop:

- un objetivo `confirmed`, `active` o `completed` requiere aprobación humana;
- un roadmap `approved`, `active`, `completed` o `superseded` requiere aprobación;
- una investigación puede prepararse en `draft` o `pending_approval`, pero no
  puede pasar a `approved`, `running` o `completed` sin aprobación humana;
- la evaluación puede actualizar métricas basadas en evidencia, pero no altera
  objetivos ni roadmaps aprobados.

## Reporter contracts

`ReporterReport` mantiene una estructura estable para reportes periódicos, pero
sus categorías son emergentes y agnósticas del dominio. `ReporterDocumentDecision`
registra relevancia, novedad, calidad, impacto y curación sin borrar artifacts.
`TopicLink` conserva la continuidad entre períodos (`new`, `stable`, `growing`,
`declining`, `split`, `merged`, `disappeared`, `ambiguous`).

Los schemas correspondientes son `reporter_report.schema.json`,
`reporter_document_decision.schema.json` y `topic_link.schema.json`. El validador
es `scripts/validate_reporter_contract.py`.
