# Estrategia de ramas — Git Flow

Este documento es la referencia normativa del repositorio. Cualquier rama nueva debe encajar en
uno de los prefijos descritos aquí.

Modelo: **Git Flow** sobre un **monorepo** (backend + frontend + admin + infra), con una única
línea de versiones (SemVer) para todo el sistema.

---

## 1. Ramas permanentes

| Rama | Propósito | Reglas |
|---|---|---|
| `main` | Código en producción. Solo recibe merges desde `release/*` y `hotfix/*`. Cada merge lleva tag `vX.Y.Z`. | Protegida. Prohibido commit directo. |
| `develop` | Rama de integración y base de todas las `feature/*`. Debe estar siempre desplegable en el entorno de desarrollo. | Protegida. Solo merge vía PR con CI en verde. |

## 2. Ramas de soporte

| Prefijo | Nace de | Se fusiona en | Uso |
|---|---|---|---|
| `feature/*` | `develop` | `develop` | Cada unidad funcional del documento de diseño. |
| `release/x.y.z` | `develop` | `main` + `develop` | Congelar alcance, estabilizar, bump de versión, changelog. |
| `hotfix/x.y.z` | `main` | `main` + `develop` | Fallos críticos en producción. |
| `bugfix/*` | `release/*` | `release/*` | Correcciones durante la estabilización de una release. |
| `support/x.y` | `main` | — | Mantenimiento de una versión antigua. No se usa por ahora. |

Prefijos adicionales, con el mismo flujo que `feature/*`:

- `chore/*` — tooling, CI, dependencias, scaffolding.
- `docs/*` — documentación pura.

## 3. Prefijos de módulo

El nombre de la rama debe decir a qué parte del sistema toca. Los prefijos derivan de las
secciones del documento de diseño:

| Prefijo | Alcance | Sección |
|---|---|---|
| `ingestion-` | Extracción de texto, chunking, embeddings | 3.1 |
| `storage-` | Base de datos vectorial y esquema de metadatos | 3.2 |
| `retrieval-` | Búsqueda por similitud, filtros, reranking | 3.3 |
| `generation-` | Prompt, LLM, citas | 3.4 |
| `api-` | Endpoints FastAPI | 4 |
| `ui-` | Chat de usuario final (Streamlit / Next.js) | 5.1 |
| `admin-` | Backoffice: CRUD, analíticas, parámetros | 5.2 |
| `eval-` | Golden set y métricas de calidad del RAG | — |
| `ops-` | Docker, despliegue, observabilidad, seguridad, rendimiento | 2.2 |

Formato final: `feature/<prefijo-modulo>-<descripcion-en-kebab-case>`
Ejemplo: `feature/retrieval-reranker`.

## 4. Roadmap de ramas por release

> Las ramas `feature/*` se crean **bajo demanda**, justo antes de empezar a trabajar en ellas.
> Esta es la lista planificada y su orden sugerido.

### `v0.1.0` — MVP: pipeline RAG local end-to-end

1. `chore/project-scaffold` — estructura del monorepo, `pyproject.toml`, settings con Pydantic, `docker-compose`, pre-commit, CI base.
2. `feature/ingestion-text-extraction` — PyMuPDF / Unstructured; PDF, DOCX, TXT; metadatos (página, autor, fecha).
3. `feature/ingestion-chunking` — `RecursiveCharacterTextSplitter`, 500–1000 tokens, overlap 10–20 %.
4. `feature/ingestion-embeddings` — interfaz `EmbeddingProvider` + `text-embedding-3-small` (1536 dim), batching.
5. `feature/storage-vectordb-chroma` — interfaz `VectorStore` + ChromaDB; esquema doc_id / chunk_id / vector / texto / metadatos.
6. `feature/api-documents-endpoints` — `POST /api/v1/documents/ingest`, `GET /api/v1/documents`, borrado.
7. `feature/retrieval-similarity-search` — cosine similarity, Top-K = 5, filtros por metadatos (`filters.category`).
8. `feature/generation-prompt-assembly` — plantilla de sistema, inyección del Top-K, respuesta «No tengo información suficiente».
9. `feature/generation-llm-provider` — interfaz `LLMProvider`; Claude vía API y/o Llama 3 local vía Ollama.
10. `feature/api-chat-query` — `POST /api/v1/chat/query` con el contrato exacto de request/response, incluido `sources[]`.
11. `feature/ui-streamlit-prototype` — chat mínimo para validar el pipeline completo.

→ `release/0.1.0`

### `v0.2.0` — Precisión, trazabilidad y evaluación

- `feature/retrieval-reranker` — Cross-Encoder ligero sobre el Top-K.
- `feature/generation-citations` — citas exactas: `doc_id`, `page`, `text_snippet`.
- `feature/ingestion-async-jobs` — ingesta en background para lotes grandes.
- `feature/eval-rag-harness` — golden set de preguntas y métricas (recall@k, fidelidad, tasa de «sin información»).
- `feature/api-chat-history` — persistencia de `chat_history_id`.

→ `release/0.2.0`

### `v0.3.0` — Frontend definitivo y backoffice

- `feature/ui-chat-nextjs` — chat React/Next.js.
- `feature/ui-source-panel-pdf-viewer` — panel lateral de fuentes; abre el PDF resaltando el párrafo citado.
- `feature/admin-knowledge-manager` — CRUD de documentos.
- `feature/admin-analytics-dashboard` — preguntas no respondidas / brechas de documentación.
- `feature/admin-runtime-params` — ajuste de chunk size y Top-K sin tocar código.
- `feature/ops-auth-rbac` — separación usuario final / administrador.

→ `release/0.3.0`

### `v1.0.0` — Requisitos no funcionales y producción

- `feature/storage-pgvector-backend` — backend alternativo pgvector / Pinecone para entorno distribuido.
- `feature/ops-latency-benchmark` — verificar el objetivo de < 500 ms.
- `feature/ops-privacy-zdr` — Zero Data Retention, políticas de logs, tratamiento de PII.
- `feature/ops-deployment` — Docker de producción, observabilidad, backups del índice.

→ `release/1.0.0`

## 5. Convenciones

- **Commits**: Conventional Commits con scope de módulo — `feat(retrieval): add cross-encoder reranker`.
- **Versionado**: SemVer. `MINOR` por cada release funcional, `PATCH` para hotfix.
- **Merge**: squash de `feature/*` → `develop`; merge commit (`--no-ff`) de `release/*` y `hotfix/*` → `main`.
- **Vida de una feature**: corta (días, no semanas). Si crece demasiado, se parte en dos ramas con el mismo prefijo de módulo.
- **Tags**: solo en `main`, formato `v0.1.0`.

## 6. Comandos habituales

```bash
# Nueva funcionalidad
git flow feature start ingestion-text-extraction
git flow feature finish ingestion-text-extraction

# Preparar una versión
git flow release start 0.1.0
git flow release finish 0.1.0

# Corrección urgente en producción
git flow hotfix start 0.1.1
git flow hotfix finish 0.1.1
```

Las ramas con prefijo `chore/` o `docs/` se crean a mano, siguiendo el mismo ciclo:

```bash
git checkout develop
git checkout -b chore/project-scaffold
```
