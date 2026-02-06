from chick_agent.agent import SimpleAgent
from chick_agent.core import ChickAgentLLM
from chick_agent.tools import MCPTool
from chick_agent.core import Config

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


def repr():
    import httpx

    agent = SimpleAgent(
        name="🤖",
        system_prompt="你是一名有用的AI助手",
        config=Config.from_toml(id="nl"),
        client=httpx.Client(trust_env=False),
    )

    kb = KeyBindings()

    @kb.add(Keys.ControlJ)
    def _(event):
        event.current_buffer.validate_and_handle()

    session = PromptSession(
        history=FileHistory("/tmp/.chat.history"), key_bindings=kb, multiline=True
    )

    while True:
        try:
            user_input = session.prompt("🙈: ").strip()

            if user_input.lower() in ["exit", "quit", "bye", "q", "x"]:
                print("退出")
                break
            if not user_input:
                continue
            print(f"{agent.name}: ", end="", flush=True)
            agent.run(user_input, stream=True)
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            break


if __name__ == "__main__":
    repr()
