from typing import override
from chick_agent.agent.basic_agent import BasicAgent
from chick_agent.core.config import Config
from chick_agent.core.llm import ChickAgentLLM
from chick_agent.core.message import Message
from chick_agent.tools import ToolRegistry
import httpx


class SimpleAgent(BasicAgent):
    def __init__(
        self,
        name: str,
        llm: ChickAgentLLM | None = None,
        system_prompt: str | None = None,
        tool_registry: ToolRegistry | None = None,
        config: Config | None = None,
        client: httpx.Client | None = None,
    ):
        super().__init__(name, llm, system_prompt, tool_registry, config, client)

    @override
    def run(
        self,
        input_text: str,
        stream: bool = False,
        max_tool_iterations: int = 3,
        **kwargs,
    ) -> str:
        messages = self._build_messages(input_text)

        if not self.enable_tool_calling:
            return self._handle_no_tool_calling(messages, input_text, stream, **kwargs)

        full_response = self._execute_tool_loop(
            messages, stream, max_tool_iterations, **kwargs
        )

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        return full_response

    def _build_messages(self, input_text: str) -> list[dict]:
        """构建消息列表，包含系统提示、历史记录和用户输入."""
        messages = []
        enhanced_prompt = self._get_system_tool_prompt()
        messages.append({"role": "system", "content": enhanced_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})
        return messages

    def _handle_no_tool_calling(
        self,
        messages: list[dict],
        input_text: str,
        stream: bool,
        **kwargs,
    ) -> str:
        """处理禁用工具调用的情况."""
        response = self._execute_llm(messages, stream, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        return response

    def _execute_tool_loop(
        self,
        messages: list[dict],
        stream: bool,
        max_tool_iterations: int,
        **kwargs,
    ) -> str:
        """执行工具调用循环直到获得最终响应."""
        current_iteration = 0

        while current_iteration < max_tool_iterations:
            current_iteration += 1
            response = self._execute_llm(messages, stream, **kwargs)
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                return response

            self._process_tool_calls(messages, tool_calls, response)

        # 达到最大迭代次数，执行最后一次LLM调用
        return self._execute_llm(messages, stream, **kwargs)

    def _process_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        response: str,
    ) -> None:
        """处理工具调用，将工具结果添加到消息列表中."""
        tool_results = []
        prev_response = response

        for call in tool_calls:
            result = self._execute_tool_call(call["tool_name"], call["parameters"])
            tool_results.append(result)
            prev_response = prev_response.replace(call["original"], "")

        messages.append({"role": "assistant", "content": prev_response})
        tool_results_text = "\n\n".join(tool_results)
        messages.append({
            "role": "user",
            "content": f"工具执行结果: \n{tool_results_text}\n\n请基于这些结果给出完整的答复",
        })
