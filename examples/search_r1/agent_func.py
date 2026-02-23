from datetime import datetime
from typing import Any, Dict

import re
import string
from openrlhf_agent.agentkit.rewards.result_rewards.hub.math_utils import extract_answer

from openrlhf_agent.agentkit.rewards import RewardPipeline
from openrlhf_agent.agentkit.session import AgentSession
from openrlhf_agent.agentkit.environments import FunctionCallEnvironment
from openrlhf_agent.agentkit.protocols import Qwen3ThinkingProtocol
from openrlhf_agent.agentkit.rewards.result_rewards import MatchingReward
from openrlhf_agent.agentkit.tools import LocalSearchTool

from openrlhf.utils.agent import MultiTurnAgentExecutor, AgentInstanceBase


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


# Synchronize with search-r1
def normalize_answer(s):
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class EMMatchingReward(MatchingReward):
    def score_response(self, response: str, label) -> float:
        if isinstance(label, str):
            labels = [label]
        elif isinstance(label, list):
            labels = label
        else:
            raise NotImplementedError(f"Unsupported label type: {type(label)!r}")
        
        labels = [normalize_answer(l) for l in labels]

        try:
            pred_answer = normalize_answer(extract_answer(response.strip()))
        except Exception:
            return self.miss_score

        for golden_answer in labels:
            if pred_answer == golden_answer:
                return self.correct_score

        return self.miss_score


class AgentInstance(AgentInstanceBase):
    def __init__(self):
        environment = FunctionCallEnvironment(
            system_prompt=CUSTOM_SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d")),
            tools=[
                LocalSearchTool(base_url="http://localhost:8000/retrieve"),
            ],
        )
        self.session = AgentSession(
            environment=environment,
            protocol=Qwen3ThinkingProtocol(),
            reward_pipeline=RewardPipeline(
                result_reward=EMMatchingReward(
                    correct_score=1.0, miss_score=0.0
                ),
            )
        )

    async def reset(self, states: dict):
        prompt = await self.session.initialize(states.get("observation"))
        return {"observation": prompt}

    async def step(self, states: dict) -> Dict[str, Any]:
        action_text: str = states.get("action_text", "")
        label = states.get("label")

        observation, reward = await self.session.step_from_text(action_text, label=label)

        reward = float(reward) if reward is not None else 0.0
        reward = max(reward, -1.0)

        done = observation.done
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": "" if done else observation.feedback_text,
            "done": done,
            "sampling_params": states.get("sampling_params", None),
            "extra_logs": {
                "dummy_scores": reward,
                "turn_count": observation.step_index,
            },
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
