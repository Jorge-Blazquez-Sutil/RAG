"""Módulo de Generación (sección 3.4 del documento de diseño).

Ensambla el prompt final (plantilla de sistema + CONTEXTO Top-K + PREGUNTA)
y llama al LLM: Claude vía API o Llama 3 local vía Ollama.

Ramas: feature/generation-prompt-assembly, feature/generation-llm-provider,
feature/generation-citations.
"""
