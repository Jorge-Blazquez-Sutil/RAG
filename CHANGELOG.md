# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y versionado según [SemVer](https://semver.org/lang/es/).

## [Unreleased]

### Added
- Inicialización del repositorio y estrategia de ramas Git Flow
  (ver [docs/branching-strategy.md](docs/branching-strategy.md)).
- Andamiaje del monorepo: aplicación FastAPI con `/api/v1/health`, configuración con
  Pydantic Settings, logging, paquetes de los módulos 3.1–3.4, gestión de dependencias
  con uv, ruff, pytest, pre-commit, CI de GitHub Actions y Docker Compose.
- Extracción de texto (módulo 3.1): extractores de PDF, DOCX y TXT/MD con texto paginado,
  metadatos (título, autor, fecha) y errores tipados para formato no soportado, archivo
  corrupto, PDF cifrado y documento sin capa de texto.
- Segmentación (módulo 3.1): troceado recursivo por tokens con solapamiento configurable,
  fragmentos que nunca cruzan página, IDs deterministas por contenido y conteo de tokens
  sustituible (`TOKENIZER`) que degrada a estimación cuando tiktoken no está disponible.
- Embeddings (módulo 3.1): interfaz `EmbeddingProvider` con implementación de OpenAI
  (lotes, orden garantizado por `index`, validación de dimensiones) y proveedor
  determinista sin red para tests y arranque local, seleccionables con `EMBEDDING_PROVIDER`.
- Base de datos vectorial (módulo 3.2): interfaz `VectorStore` con implementación en
  ChromaDB — similitud del coseno, upsert idempotente, filtros por metadatos, inventario
  de documentos, borrado por documento y detección de cambios de dimensión.
- API de consulta (sección 4): `POST /chat/query` con el contrato del documento
  (`query`, `chat_history_id`, `filters` → `answer`, `sources[]`), encadenando
  recuperación, prompt y generación; sin contexto responde la frase de rendición sin
  llamar al modelo, y reporta latencias de recuperación y total.
- API de documentos (sección 4): `POST /documents/ingest` (subida multipart con etiquetas,
  límite de tamaño verificado al escribir, nombre saneado contra path traversal),
  `GET /documents` y `DELETE /documents/{doc_id}`, con los errores de ingesta mapeados a
  códigos HTTP; el pipeline de ingesta vive en `app.ingestion.pipeline` para poder
  reutilizarlo fuera de HTTP.
- Generación (módulo 3.4): interfaz `LLMProvider` con implementación de Claude —solo lee
  los bloques de texto, sin parámetros de muestreo, comprueba `stop_reason` antes del
  contenido y activa el fallback de rechazo— más un proveedor `echo` sin modelo ni clave
  para probar el circuito completo; `LLM_PROVIDER` selecciona cuál.
- Ensamblaje del prompt (módulo 3.4): plantilla de sistema con la respuesta de rendición
  literal, citas por documento y página, y contexto tratado como datos frente a inyección
  de instrucciones; `build_prompt()` ajusta el CONTEXTO a un presupuesto de tokens
  (`MAX_CONTEXT_TOKENS`) e incorpora el historial acotado.
- Recuperación (módulo 3.3): `Retriever` que vectoriza la consulta con el proveedor de la
  ingesta, busca el Top-K con filtros de metadatos, aplica umbral de score opcional
  (`RETRIEVAL_MIN_SCORE`) y mide la latencia; `RetrievalResult.as_context()` prepara el
  CONTEXTO etiquetado con documento y página para el prompt del módulo 3.4.
