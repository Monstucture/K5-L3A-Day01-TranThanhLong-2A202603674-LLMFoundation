"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to src/optional_llm_api/solution/solution.py when done.
    4. Run from the repo root: pytest tests/test_solution.py -v

Models (11/2026): gpt-4o / gpt-4o-mini are removed from the OpenAI API on 2027-01-14
(https://developers.openai.com/api/docs/deprecations), so this lab uses gpt-6.1-terra / gpt-6-luna.
The result keys and table headers below still say "gpt4o" / "GPT-4o" / "Mini": they are a test
contract kept for compatibility. Read "gpt4o" as "the large model" and "mini" as "the small model".
"""

import os
import time
from typing import Any, Callable
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — OpenAI pricing page, 11/2026. Update if pricing changes.
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


# Add get client to avoid code duplication
def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found — check the variable name in your .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: OPENAI_MODEL).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",          # GPT-6 models only accept temperature/top_p with effort "none"
            temperature=temperature, top_p=top_p,
            max_completion_tokens=max_tokens,  # GPT-6 models reject max_tokens
        )
    """
    # TODO: import OpenAI, create client, call chat.completions.create,
    #       measure start/end time, return (response_text, latency)
    client = get_client()

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"effort": "none"}},  # OpenRouter's way to disable reasoning
    )
    latency = time.perf_counter() - t0

    text = resp.choices[0].message.content or ""
    return text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using OPENAI_MINI_MODEL and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both OPENAI_MODEL and OPENAI_MINI_MODEL with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    # TODO: call call_openai and call_openai_mini, assemble and return the dict
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    # ~0.75 words per token -> tokens ≈ words / 0.75
    est_tokens = len(gpt4o_response.split()) / 0.75
    gpt4o_cost_estimate = est_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns (= 6 messages): history = history[-6:]
    """
    # TODO: enter while-loop, read user input, stream response, maintain history
    client = get_client() 
    history: list[dict] = []

    print("Streaming chatbot — type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            extra_body={"reasoning": {"effort": "none"}},   # OpenRouter form, not reasoning_effort
        )

        print("Assistant: ", end="", flush=True)
        reply_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        history.append({"role": "assistant", "content": "".join(reply_parts)})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    for attempt in range(max_retries + 1):   # 1 initial call + max_retries retries
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise                         # re-raise the last exception if all retries are exhausted
            time.sleep(base_delay * (2 ** attempt)) 


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, add "prompt" key
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    # TODO: build and return a formatted table string
    raise NotImplementedError("Implement format_comparison_table")
    


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    # print("=== Comparing models ===")
    # result = compare_models(test_prompt)
    # for key, value in result.items():
    #     print(f"{key}: {value}")

    

    # --- Task 1: call_openai -------------------------------------------------
    print("=== Task 1: call_openai ===")
    text, latency = call_openai(test_prompt)
    print(f"Response: {text}")
    print(f"Latency:  {latency:.2f}s\n")

    # --- Task 2: call_openai_mini --------------------------------------------
    print("=== Task 2: call_openai_mini ===")
    text, latency = call_openai_mini(test_prompt)
    print(f"Response: {text}")
    print(f"Latency:  {latency:.2f}s\n")

    # --- Task 3: compare_models ----------------------------------------------
    print("=== Task 3: compare_models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")
    print()

    # --- Task 4: retry_with_backoff (offline — no API calls) -----------------
    print("=== Task 4: retry_with_backoff ===")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError(f"simulated failure #{calls['n']}")
        return "success"

    print(f"Flaky fn result: {retry_with_backoff(flaky)} after {calls['n']} calls")

    def always_fails():
        raise RuntimeError("always fails")

    try:
        retry_with_backoff(always_fails, max_retries=2)
    except RuntimeError as e:
        print(f"Exhausted retries as expected: {e}")

    text, latency = retry_with_backoff(lambda: call_openai_mini("Say hi in 3 words."))
    print(f"Real API call via retry: {text} ({latency:.2f}s)\n")

    # --- Task 5: batch_compare -----------------------------------------------
    print("=== Task 5: batch_compare ===")
    prompts = [
        "What is tokenization?",
        "Name one difference between GPT and BERT.",
    ]
    results = batch_compare(prompts)
    print(f"Got {len(results)} results; keys: {list(results[0].keys())}\n")

    # --- Task 6: format_comparison_table -------------------------------------
    print("=== Task 6: format_comparison_table ===")
    # print(format_comparison_table(results))

    # --- Task 7: streaming_chatbot ----------------------------------
    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
