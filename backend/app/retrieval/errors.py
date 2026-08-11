"""Errores del motor de recuperación."""


class RetrievalError(Exception):
    """Error base de la recuperación."""


class EmptyQueryError(RetrievalError):
    """La consulta llegó vacía o solo con espacios."""

    def __init__(self) -> None:
        super().__init__("La consulta no puede estar vacía")
