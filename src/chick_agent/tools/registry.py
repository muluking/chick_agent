from chick_agent.tools.tool import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, object]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        if auto_expand:
            if hasattr(tool, "auto_expand") and tool.auto_expand:
                expanded_tools = tool.auto_expand_tools()
                if expanded_tools:
                    for t in expanded_tools:
                        self._tools[t.name] = t
                    print(f"{tool.name} 展开为: {len(expanded_tools)} 个工具")
                    return
