import sys

from chick_agent.agent import SimpleAgent
from chick_agent.core import ChickAgentLLM
from chick_agent.core import Config

import httpx

SYSTEM_PROMPT = """
你现在是一名资深的软件工程师，你熟悉多种编程语言和开发框架，对软件开发的生命周期有深入的理解。你擅长解决技术问题，并具有优秀的逻辑思维能力。

你的主要工作是审查我的代码变更，根据变更内容生成变更信息。我会给你提供我的代码提交diff文件内容，依据这些内容生成提交信息，要求提交信息简明扼要，覆盖修改内容，思考过程使用中文，最终输出的信息采用英文。

生成的提交信息按照如下格式, 第一行为标题，总体描述修改的内容， 然后空一行， 后续按照改动点进行描述，例如：

Add TOML config support and refactor agent initialization

- Add `config.toml` to .gitignore to exclude local configuration files
- Refactor BasicAgent/SimpleAgent to support initialization via Config (instead of direct ChickAgentLLM)
  - Add `config` and `client` parameters to agent constructors
  - Auto-generate ChickAgentLLM instance from Config if llm is not provided
- Extend Config model with LLM-specific fields (model, api_key, base_url, timeout)
"""


def git_diff_commiter():
    agent = SimpleAgent(
        name="🤖",
        system_prompt=SYSTEM_PROMPT,
        config=Config.from_toml(id="doubao"),
        client=httpx.Client(trust_env=False),
    )

    sys.stdin.reconfigure(encoding="utf-8")
    content = sys.stdin.read()
    if not content:
        print("no stdin content")
        return
    print(f"{agent.name}: ", end="", flush=True)
    agent.run(content, stream=True)


if __name__ == "__main__":
    git_diff_commiter()
