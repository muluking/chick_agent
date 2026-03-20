"""
Agent 基类模块

定义所有 AI Agent 的抽象基类，提供通用的状态管理、消息历史和 LLM 交互接口。
Agent 是 AI 系统的核心抽象，负责处理用户输入、调用 LLM 并返回响应。
"""

from abc import ABC, abstractmethod

from chick_agent.core.config import Config
from chick_agent.core.exceptions import LLMException
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message


class Agent(ABC):
    """
    Agent 抽象基类

    所有具体 Agent 实现必须继承此类并实现 run 方法。
    提供 LLM 客户端管理、系统提示词配置、消息历史管理等通用功能。

    Attributes:
        name: Agent 实例的名称，用于标识和日志记录
        llm: LLM 客户端实例，负责与语言模型交互
        system_prompt: 系统提示词，用于设定 Agent 的角色和行为
        config: Agent 配置对象，控制模型参数等行为设置
        _history: 消息历史列表，记录所有对话内容
    """

    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
    ):
        """
        初始化 Agent 实例

        Args:
            name: Agent 的名称，应具有唯一性
            llm: 已初始化的 LLM 客户端实例，不能为空
            system_prompt: 可选的系统提示词，定义 Agent 的行为模式
            config: 可选的配置对象，若不提供则使用默认配置

        Raises:
            LLMException: 当 llm 参数为空或未初始化时抛出
        """
        # 验证 LLM 客户端必须存在且已初始化
        if not llm:
            raise LLMException("llm client should be initialized.")

        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        # 如果未提供配置，使用默认配置实例
        self.config = config or Config()
        # 初始化空的消息历史列表
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        处理用户输入并返回响应的抽象方法

        具体 Agent 实现必须实现此方法以定义其核心逻辑。

        Args:
            input_text: 用户的输入文本
            **kwargs: 其他可选参数（如 temperature, max_tokens 等）

        Returns:
            str: Agent 的响应文本

        Note:
            此方法为抽象方法，子类必须实现具体逻辑
        """
        pass

    def add_message(self, message: Message):
        """
        向对话历史中添加一条消息

        用于记录用户消息和 Agent 响应，以便在多轮对话中保持上下文。

        Args:
            message: Message 对象，包含消息角色和内容信息
        """
        self._history.append(message)

    def clear_history(self):
        """
        清空所有对话历史

        调用后 _history 列表将被清空，重新开始新的对话会话。
        通常用于开始新的对话主题或重置对话状态。
        """
        self._history.clear()

    def get_history(self) -> list[Message]:
        """
        获取完整的对话历史

        Returns:
            list[Message]: 包含所有历史消息的列表
        """
        return self._history

    def __str__(self) -> str:
        """
        返回 Agent 的字符串表示

        Returns:
            str: 包含名称和 LLM 提供者的格式化字符串
        """
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        """
        返回 Agent 的官方字符串表示

        用于调试和开发工具中的显示。

        Returns:
            str: 与 __str__ 相同的字符串表示
        """
        return self.__str__()
