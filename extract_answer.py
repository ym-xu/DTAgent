from __future__ import annotations

from openai import OpenAI

client = OpenAI()


def extract_answer(
    question: str,
    output: str,
    prompt: str,
    model_name: str = "gpt-4o",
) -> str:
    """
    Use an LLM to extract the final answer string from the agent's analysis output.
    Returns the extracted answer or 'Failed' on error.
    """
    messages = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": f"\n\nQuestion:{question}\nAnalysis:{output}\n",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return "Failed"


__all__ = ["extract_answer"]
