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
