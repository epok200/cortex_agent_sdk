from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, SecretStr


class OpenAICompatibleGateway(BaseModel):
    """Conexión autenticada a un endpoint compatible con OpenAI Responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: AnyHttpUrl
    api_key: SecretStr = Field(min_length=32)

    def __init__(
        self,
        *,
        url: str | AnyHttpUrl,
        api_key: str | SecretStr,
    ) -> None:
        super().__init__(url=url, api_key=api_key)

    @property
    def base_url(self) -> str:
        return f"{str(self.url).rstrip('/')}/v1"
