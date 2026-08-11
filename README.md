# Asistente RAG de Nicho

Herramienta de IA especializada basada en **Generación Aumentada por Recuperación (RAG)**.
Ingiere documentos complejos (contratos legales, manuales técnicos) y permite consultarlos en
lenguaje natural con respuestas fundamentadas y **citas exactas** (documento + página).

Especificación completa: [`docs/design/`](docs/design/).

## Arquitectura (resumen)

| Módulo | Responsabilidad | Tecnología prevista |
|---|---|---|
| 3.1 Ingesta | Extracción de texto → chunking → embeddings | PyMuPDF / Unstructured, `RecursiveCharacterTextSplitter`, `text-embedding-3-small` |
| 3.2 Almacenamiento | Índice semántico y metadatos | ChromaDB (local) · pgvector / Pinecone (distribuido) |
| 3.3 Recuperación | Búsqueda por similitud del coseno (Top-K) + reranking | Cross-Encoder ligero |
| 3.4 Generación | Ensamblaje del prompt y llamada al LLM | Claude vía API · Llama 3 local vía Ollama |

**Backend:** Python / FastAPI · **Orquestador:** LangChain
**UI:** Streamlit (prototipo) → React/Next.js (chat definitivo + backoffice)

## Arranque rápido

Requisitos: Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
```

```bash
cp .env.example .env
```

Arranca la API con recarga automática:

```bash
uv run uvicorn app.main:app --reload --app-dir backend
```

- Comprobación de salud: <http://localhost:8000/api/v1/health>
- Documentación interactiva: <http://localhost:8000/docs>

Calidad y tests:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

> Si `uv` falla con `invalid peer certificate: UnknownIssuer` (proxy o antivirus que
> intercepta TLS), añade `--system-certs` o exporta `UV_NATIVE_TLS=true`.

Alternativa con Docker:

```bash
docker compose up --build
```

## Estructura del monorepo

```
backend/app/        Aplicación FastAPI
  core/             Configuración (Pydantic Settings) y logging
  api/v1/           Routers y endpoints
  ingestion/        Módulo 3.1 — extracción, chunking, embeddings
  storage/          Módulo 3.2 — base de datos vectorial
  retrieval/        Módulo 3.3 — búsqueda semántica y reranking
  generation/       Módulo 3.4 — prompt y LLM
  schemas/          Contratos Pydantic de la API
frontend/           Chat de usuario final (Streamlit primero, Next.js después)
admin/              Backoffice: gestor de conocimiento, analíticas, parámetros
infra/              Dockerfile y configuración de despliegue
tests/              Suite de pytest
docs/               Documentación de diseño y de proceso
```

Los módulos 3.1–3.4 son paquetes vacíos por ahora: cada rama del roadmap los rellena.

## API

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/api/v1/documents/ingest` | Sube un archivo, lo trocea, genera embeddings y los indexa |
| `GET` | `/api/v1/documents` | Lista los documentos indexados |
| `POST` | `/api/v1/chat/query` | Consulta principal: devuelve `answer` + `sources[]` |

## Flujo de trabajo

Este repositorio sigue **Git Flow**. Antes de crear una rama, lee
[`docs/branching-strategy.md`](docs/branching-strategy.md).

```bash
git flow feature start ingestion-text-extraction
```

- Ramas permanentes: `main` (producción) y `develop` (integración).
- Las `feature/*` nacen de `develop` y usan prefijo de módulo: `feature/<modulo>-<detalle>`.
- Commits en formato Conventional Commits: `feat(retrieval): add cross-encoder reranker`.
