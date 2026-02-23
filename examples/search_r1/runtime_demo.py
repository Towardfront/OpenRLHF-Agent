import asyncio
from datetime import datetime
from openrlhf_agent.backends import OpenAIEngine
from openrlhf_agent.agentkit.runtime import AgentRuntime
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.tools import LocalSearchTool


CUSTOM_SYSTEM_PROMPT = """
You are a helpful assistant.
Your Knowledge cutoff: 2023-06
Current date: {date}

## Core Loop
Work independently with tools until the solution is correct:
- Reason, plan, and use tools.
- Validate: verify claims, test edge cases, ensure proper format.

## Final Protocol
Your final response must be:
1) A **Markdown-formatted explanation**.
2) The **very last line** must be exactly:
   `Answer: \\boxed{{final_answer_string}}`

Rules:
- Do not add anything after the final line (no extra text or whitespace-only lines).
- Inside `\\boxed{{...}}`, include **only** the final answer string.
""".strip()


async def main() -> None:
    agent_runtime = AgentRuntime(
        protocol=Qwen3ThinkingProtocol(),
        engine=OpenAIEngine(
            model="qwen3", 
            base_url="http://localhost:8009/v1",
            api_key="empty"
        ),
        environment=FunctionCallEnvironment(
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
            tools=[
                LocalSearchTool(base_url="http://localhost:8000/retrieve"),
            ],
        ),
    )
    messages = [{"role": "user", "content": "Please use the commentary tool to share your thoughts, and use local_search to find what Python is."}]
    async for message in agent_runtime.run_steps(messages):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
