"""Memory mechanics and the benchmark checker are given; the loop is yours."""

from agent.evaluate import check_answer
from agent.memory import ConversationMemory


def test_memory_compresses_old_messages_when_over_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=60, keep_last=2)
    for i in range(6):
        memory.add("user" if i % 2 == 0 else "assistant", f"message number {i} " * 2)
    assert memory.over_budget()
    compressed = memory.compress(lambda old: f"{len(old)} old messages summarised")
    assert compressed
    assert len(memory.messages) == 2
    assert "4 old messages" in memory.summary
    assert memory.render()[0]["role"] == "system" and "Summary" in memory.render()[0]["content"]


def test_memory_does_nothing_under_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10_000)
    memory.add("user", "hello")
    assert not memory.compress(lambda old: "unused")


def test_check_answer_kinds():
    assert check_answer({"check": {"type": "contains", "value": "Madrid"}}, "It is in madrid.")
    assert check_answer({"check": {"type": "number", "value": 7}}, "The name has 7 letters")
    assert check_answer({"check": {"type": "number", "value": 21.5, "tolerance": 0.5}}, "≈ 21.2 °C")
    assert not check_answer({"check": {"type": "regex", "value": r"\bF\b"}}, "no unit here")


def _tool_conversation(budget: int, **options) -> ConversationMemory:
    memory = ConversationMemory(
        "sys", count_tokens=len, token_budget=budget, keep_last=3, **options
    )
    memory.add("user", "TASK")
    for i in range(3):
        memory.add("assistant", f"call {i}")
        memory.add("tool", f"result {i} " + "x" * 50, name="t")
    return memory


def test_pin_first_keeps_the_task_out_of_the_summary():
    memory = _tool_conversation(40, pin_first=True)
    seen = []
    assert memory.compress(lambda old: seen.append(old) or "S")
    assert memory.messages[0]["content"] == "TASK"
    assert all(m["content"] != "TASK" for m in seen[0])


def test_the_cut_never_orphans_an_observation():
    memory = _tool_conversation(40, pin_first=True)
    memory.compress(lambda old: "S")
    live = [m["role"] for m in memory.messages[1:]]
    assert live[0] == "assistant"  # the window starts at a call, not at its result


def test_long_observations_are_clipped_before_summarising():
    memory = _tool_conversation(40, pin_first=True, max_observation_chars=10)
    seen = []
    memory.compress(lambda old: seen.append(old) or "S")
    assert all(len(m["content"]) <= 10 + 4 for m in seen[0] if m["role"] == "tool")


def test_extractive_summary_keeps_tools_and_results():
    from agent.memory import extractive_summary

    text = extractive_summary(
        [
            {"role": "assistant", "content": "<tool_call>get_euribor</tool_call>"},
            {"role": "tool", "name": "get_euribor", "content": '{"rate_pct": 2.5}'},
        ]
    )
    assert "get_euribor" in text and "2.5" in text
