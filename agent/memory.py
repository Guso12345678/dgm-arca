"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.

Our compression policy (the part the practice leaves to us), all of it optional so the
defaults behave like the original class:

* ``pin_first``: the first message (the user's task) is never summarised. A summary of the
  task is the worst thing to lose: every later decision depends on it.
* the cut never separates a tool observation from the assistant message that asked for it:
  an orphan observation ("0.38") means nothing without the call that produced it.
* ``max_observation_chars``: long observations (retrieved passages, JSON) are clipped
  before they are summarised; the numbers that matter are in the first characters, the rest
  is what makes the middle of the context get lost.
* ``extractive_summary``: a deterministic summary ("called X with ... -> ...") that needs no
  model call. Small models summarise badly and every call costs latency and tokens; the
  facts of an agent trace (which tool, which arguments, which number came back) are
  structured, so extracting them is more reliable than asking a 0.6B model to paraphrase.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

CLIP_MARKER = " […]"


def clip(text: str, limit: int | None) -> str:
    """Cut ``text`` to ``limit`` characters, marking the cut. ``None`` means no limit."""
    if limit is None or len(text) <= limit:
        return text
    return text[:limit].rstrip() + CLIP_MARKER


def extractive_summary(messages: list[dict], max_observation_chars: int = 300) -> str:
    """Deterministic summary of agent turns: what was done and what came back."""
    lines = []
    for message in messages:
        role, content = message["role"], str(message["content"])
        if role == "assistant":
            lines.append(f"- You did: {clip(' '.join(content.split()), 240)}")
        elif role == "tool":
            name = message.get("name", "tool")
            lines.append(
                f"- Result of {name}: {clip(' '.join(content.split()), max_observation_chars)}"
            )
        else:
            lines.append(f"- {role}: {clip(' '.join(content.split()), 240)}")
    return "\n".join(lines)


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    pin_first: bool = False
    max_observation_chars: int | None = None
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages to send to the model: system (+ summary) followed by the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        return [{"role": "system", "content": system}, *self.messages]

    def _cut_index(self, start: int) -> int:
        """Index where the live window starts: ``keep_last`` messages back, aligned so that a
        tool observation stays together with the call that produced it."""
        cut = len(self.messages) - self.keep_last
        while start < cut < len(self.messages) and self.messages[cut]["role"] == "tool":
            cut -= 1
        return cut

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Summarise the oldest turns when over budget. Returns True if it compressed.

        What is summarised: everything between the pinned first message (if ``pin_first``)
        and the last ``keep_last`` messages, with long observations clipped. When it runs:
        only when the token budget is exceeded, so short tasks never pay for it.
        """
        if not self.over_budget():
            return False
        start = 1 if self.pin_first and self.messages else 0
        cut = self._cut_index(start)
        if cut <= start:
            return False
        old = [
            {**m, "content": clip(str(m["content"]), self.max_observation_chars)}
            if m["role"] == "tool"
            else m
            for m in self.messages[start:cut]
        ]
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = self.messages[:start] + self.messages[cut:]
        return True


def describe_arguments(arguments: dict) -> str:
    """Compact one-line rendering of tool arguments, for logs and summaries."""
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True)