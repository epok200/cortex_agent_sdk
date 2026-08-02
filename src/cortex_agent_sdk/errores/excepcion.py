from cortex_agent_sdk.errores.catalogo import CodigoError, Severidad


class AppError(Exception):
    """Error del SDK con código estable y mensaje seguro opcional."""

    def __init__(
        self,
        codigo: CodigoError,
        detalle: str,
        severidad: Severidad = Severidad.ERROR,
        mensaje_seguro: str | None = None,
    ) -> None:
        self.codigo = codigo
        self.detalle = detalle
        self.severidad = severidad
        self.mensaje_seguro = mensaje_seguro
        super().__init__(f"[{codigo.value}] {detalle}")

