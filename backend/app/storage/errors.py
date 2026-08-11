"""Errores de la capa de almacenamiento vectorial."""


class VectorStoreError(Exception):
    """Error base de la base de datos vectorial."""


class VectorStoreConfigurationError(VectorStoreError):
    """Configuración inválida o motor no soportado."""


class DimensionMismatchError(VectorStoreError):
    """El vector no tiene la dimensión con la que se creó la colección.

    Ocurre al cambiar de modelo de embeddings sin reindexar: los vectores nuevos
    y los viejos no son comparables, así que hay que detectarlo al escribir y no
    cuando las búsquedas empiecen a devolver resultados sin sentido.
    """

    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        super().__init__(
            f"La colección almacena vectores de {expected} dimensiones y llegó uno "
            f"de {received}. Si has cambiado de modelo de embeddings, hay que "
            "reindexar los documentos."
        )
