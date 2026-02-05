import os
import tomllib
from openai import api_key
from pydantic import BaseModel
import httpx


class Config(BaseModel):
    model: str = "deepseek-chat"
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 60
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model=os.getenv("LLM_MODEL_ID"),
            provider=os.getenv("LLM_PROVIDER"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", 4096))
            if os.getenv("MAX_TOKENS")
            else None,
        )

    @classmethod
    def from_toml(cls, filename: str = "config.toml", id: str = "deepseek") -> "Config":
        fd = open(filename, "rb")
        config = tomllib.load(fd)
        fd.close()
        sect = config.get(id, {})
        if not sect:
            return cls.from_env()
        return cls(
            model=sect.get("model", ""),
            provider=sect.get("provider", ""),
            api_key=sect.get("api_key", ""),
            base_url=sect.get("base_url", ""),
            debug=sect.get("debug", "false").lower() == "true",
            log_level=sect.get("LOG_LEVEL", "INFO"),
            temperature=float(sect.get("temperature", 0.7)),
            max_tokens=int(sect.get("max_tokens", 4096)),
            max_history=int(sect.get("max_history", 100)),
        )

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()


if __name__ == "__main__":
    config = Config.from_toml()
    print(config.to_dict())
