"""Resultado de una recuperación (módulo 3.3)."""

from pydantic import BaseModel, Field

from app.storage.base import SearchResult


class RetrievalResult(BaseModel):
    """Fragmentos recuperados para una consulta, con su traza de ejecución."""

    query: str
    results: list[SearchResult]
    #: Candidatos que devolvió el índice antes de aplicar el umbral de score.
    candidates: int = Field(ge=0)
    #: Latencia total: el requisito no funcional fija el objetivo en 500 ms.
    elapsed_ms: float = Field(ge=0)

    @property
    def is_empty(self) -> bool:
        """True si no hay nada que pasarle al LLM como CONTEXTO."""
        return not self.results

    @property
    def best_score(self) -> float | None:
        return self.results[0].score if self.results else None

    def as_context(self, separator: str = "\n\n---\n\n") -> str:
        """Concatena los fragmentos para inyectarlos en el prompt (módulo 3.4).

        Cada bloque va precedido de su origen para que el modelo pueda citar sin
        tener que adivinar de qué documento salió cada cosa.
        """
        return separator.join(
            f"[{item.chunk.filename}, página {item.chunk.page_number}]\n{item.chunk.text}"
            for item in self.results
        )
