import asyncio

from concurrent import futures
from pdb import run

from chick_agent.tools.tool import Tool, ToolParameter
from chick_agent.protocols.mcp import MCPClient


class MCPTool(Tool):
    def __init__(
        self,
        name: str = "mcp",
        description: str | None = None,
        server_command: list[str] | None = None,
        server_args: list[str] | None = None,
        server: object | None = None,
        auto_expand: bool = True,
        env: dict[str, str] | None = None,
    ):
        self.name = name
        self.server_command = server_command
        self.server_args = server_args or []
        self.server = server
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""
        self._client = None
        self._available_tool = []
        self.env = env
        self._discover_tools()
        super().__init__(name=name, description=description)

    def auto_expand_tools(self) -> list[Tool] | None:
        if not self.auto_expand:
            return None
        tools = []
        for mcp_tool in self._available_tool:
            tool = Tool(
                name=f"{self.name}_{mcp_tool['name']}",
                description=mcp_tool.get("description", ""),
                # func=lambda input_text, name=mcp_tool["name"]: self.run(
                #     {
                #         "action": "call_tool",
                #         "tool_name": name,
                #         "arguments": {"input": input_text},
                #     }
                # ),
            )
            tools.append(tool)
        return tools

    def _discover_tools(self):
        try:

            async def discover():
                source = self.server if self.server else self.server_command
                async with MCPClient(
                    source, server_args=self.server_args, env=self.env
                ) as client:
                    tools = await client.list_tools()
                    return tools

            try:
                _ = asyncio.get_running_loop()

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(discover())
                    finally:
                        new_loop.close()

                with futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    self._available_tool = future.result()
            except RuntimeError:
                self._available_tool = asyncio.run(discover())

        except Exception as e:
            self._available_tool = []

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt",
                required=True,
            ),
            ToolParameter(
                name="tool_name",
                type="string",
                description="工具名称（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="arguments",
                type="object",
                description="工具参数（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="uri",
                type="string",
                description="资源 URI（read_resource 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_name",
                type="string",
                description="提示词名称（get_prompt 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_arguments",
                type="object",
                description="提示词参数（get_prompt 操作可选）",
                required=False,
            ),
        ]

    def run(self, parameters: dict[str, object]) -> str:
        pass
