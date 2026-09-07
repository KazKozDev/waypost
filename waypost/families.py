"""Which model is this, really?

Two offerings at different hosts can be the same weights. A fallback
ladder built from "groq/llama-3.3-70b" then "openrouter/llama-3.3-70b"
looks diverse by provider and is not: when the failure is the model —
it cannot call tools, it mangles this language, it is simply too small
for the task — every rung fails the same way, and the ladder has spent
four attempts learning one fact.

Provider diversity protects against a host being down. Family diversity
protects against a model being wrong. The plan needs both, and they are
not the same axis.

The mapping is a table of substrings because that is what model ids
actually are. It is checked longest-first so that "qwen2.5-coder" is a
coder before it is a qwen, and specific vendors win over the generic
words they contain.
"""
from __future__ import annotations

# Ordered by specificity: the first match wins, so a more specific family
# must come before the general one it contains.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Code-specialised lines are their own family: a coder and a chat
    # model from the same vendor fail on entirely different things.
    ("coder", ("coder", "codestral", "starcoder", "codellama", "code-")),
    ("deepseek", ("deepseek", "r1-distill")),
    ("qwen", ("qwen", "qwq")),
    ("llama", ("llama", "nemotron")),  # nemotron is a Llama derivative
    ("gemma", ("gemma", "gemini")),
    ("mistral", ("mistral", "mixtral", "magistral", "ministral", "devstral")),
    ("phi", ("phi-", "phi3", "phi4")),
    ("glm", ("glm", "chatglm")),
    ("minimax", ("minimax",)),
    ("kimi", ("kimi", "moonshot")),
    ("yi", ("yi-", "01-ai")),
    ("command", ("command-r", "cohere", "aya")),
    ("claude", ("claude",)),
    ("grok", ("grok", "xai")),
    ("openai", ("gpt-", "gpt3", "gpt4", "gpt5", "o1-", "o3-", "o4-", "oss")),
    ("granite", ("granite",)),
    ("falcon", ("falcon",)),
    ("olmo", ("olmo",)),
    ("smol", ("smollm", "smol")),
    ("internlm", ("internlm", "intern-")),
    ("exaone", ("exaone",)),
    ("solar", ("solar",)),
)


def model_family(model_id: str) -> str:
    """Best guess at the weights behind an id.

    Unknown ids fall back to the vendor prefix ("acme/thing-7b" → "acme")
    and finally to the id itself. Falling back to something unique is
    deliberate: an unrecognised model is treated as its own family, so
    diversity is over-estimated rather than under-estimated. Two models
    wrongly called different costs one redundant attempt; two wrongly
    called the same removes a real fallback.
    """
    low = model_id.lower()
    for family, markers in _FAMILIES:
        if any(m in low for m in markers):
            return family
    if "/" in low:
        return low.split("/", 1)[0]
    return low


# The ensemble has called it this since before the table existed.
get_model_family = model_family
