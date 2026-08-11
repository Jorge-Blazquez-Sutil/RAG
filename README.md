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

## Estructura del monorepo

```
backend/    API FastAPI y módulos RAG (ingesta, almacenamiento, recuperación, generación)
frontend/   Chat de usuario final (Streamlit primero, Next.js después)
admin/      Backoffice: gestor de conocimiento, analíticas, parámetros
infra/      Docker, despliegue, configuración de entornos
docs/       Documentación de diseño y de proceso
```

> Los directorios se van creando conforme avanzan las ramas del roadmap; a día de hoy solo
> existe `docs/`.

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
