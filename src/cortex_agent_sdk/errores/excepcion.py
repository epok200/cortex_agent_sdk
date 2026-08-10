from cortex_agent_sdk.errores.catalogo import CodigoError, Severidad


class AppError(Exception):
    """Error de Cortex con código estable."""

    def __init__(
        self,
        codigo: CodigoError,
        detalle: str,
        severidad: Severidad = Severidad.ERROR,
    ) -> None:
        self.codigo = codigo
        self.detalle = detalle
        self.severidad = severidad
        super().__init__(f"[{codigo.value}] {detalle}")
