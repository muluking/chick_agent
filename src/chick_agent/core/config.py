"""
配置模块

定义 Agent 运行所需的配置类，支持从环境变量和 TOML 文件加载配置。
提供统一的配置管理接口，包含模型参数、超时设置、日志配置等。
"""

import os
import tomllib

from pydantic import BaseModel


class Config(BaseModel):
    """
    Agent 配置类

    使用 Pydantic BaseModel 提供数据验证和配置管理功能。
    支持从环境变量或 TOML 文件加载配置。

    Attributes:
        model: LLM 模型标识符，默认为 "deepseek-chat"
        provider: LLM 提供商名称，默认为 "deepseek"
        api_key: API 密钥，用于认证 LLM 服务
        base_url: API 基础 URL，默认为空字符串
        temperature: 生成温度参数，控制随机性 (0.0-2.0)
        max_tokens: 最大生成 token 数，None 表示不限制
        timeout: 请求超时时间（秒），默认 60 秒
        debug: 是否开启调试模式，默认 False
        log_level: 日志级别，默认 "INFO"
        max_history_length: 保留的最大历史消息数量，默认 100
    """

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
        """
        从环境变量加载配置

        从当前系统的环境变量中读取配置项。
        变量名遵循 LLM_* 前缀规范。

        Returns:
            Config: 从环境变量构建的配置实例

        Environment Variables:
            LLM_MODEL_ID: 模型标识符
            LLM_PROVIDER: 提供商名称
            LLM_API_KEY: API 密钥
            LLM_BASE_URL: API 基础 URL
            TEMPERATURE: 生成温度
            MAX_TOKENS: 最大 token 数
            DEBUG: 调试模式 ("true"/"false")
            LOG_LEVEL: 日志级别
        """
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
    def from_toml(
        cls, filename: str = "config.toml", section: str = "deepseek"
    ) -> "Config":
        """
        从 TOML 文件加载配置

        读取指定 TOML 配置文件中的 section，若不存在或为空则回退到环境变量。

        Args:
            filename: TOML 配置文件路径，默认为 "config.toml"
            section: 配置 section 名称，默认为 "deepseek"

        Returns:
            Config: 从 TOML 文件或环境变量构建的配置实例

        Raises:
            FileNotFoundError: 当配置文件不存在时
        """
        try:
            with open(filename, "rb") as fd:
                config = tomllib.load(fd)
        except FileNotFoundError:
            # 配置文件不存在时，回退到环境变量
            return cls.from_env()

        sect = config.get(section, {})
        if not sect:
            return cls.from_env()

        return cls(
            model=sect.get("model", ""),
            provider=sect.get("provider", ""),
            api_key=sect.get("api_key", ""),
            base_url=sect.get("base_url", ""),
            debug=sect.get("debug", "false").lower() == "true",
            log_level=sect.get("log_level", "INFO"),
            temperature=float(sect.get("temperature", 0.7)),
            max_tokens=int(sect.get("max_tokens", 4096)),
            max_history_length=int(sect.get("max_history_length", 100)),
        )

    def to_dict(self) -> dict[str, object]:
        """
        转换为字典格式

        Returns:
            dict[str, object]: 包含所有配置字段的字典
        """
        return self.model_dump()


if __name__ == "__main__":
    config = Config.from_toml()
    print(config.to_dict())
