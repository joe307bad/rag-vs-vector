from __future__ import annotations

import json
import random

import anthropic

import config

JUDGE_PROMPT = """\
You are an objective evaluator scoring AI-generated answers against a reference answer.

Question: {question}

Reference Answer: {reference}

{answers}

Score each answer from 0 to 10 on two dimensions:
- correctness: Does the answer accurately reflect the reference answer's facts? (0=completely wrong, 10=perfectly correct)
- relevance: Does the answer directly address the question? (0=totally off-topic, 10=perfectly on-point)

Respond ONLY with valid JSON, one entry per answer number:
{example}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _score(value) -> int:
    """Scores sometimes come back as strings ("9"); anything not a number 0-10 is invalid."""
    score = int(float(value))
    if not 0 <= score <= 10:
        raise ValueError(f"score out of range: {value!r}")
    return score


def _scores_for(by_number: dict, count: int) -> dict[str, dict]:
    return {
        str(i): {"correctness": _score(by_number[str(i)]["correctness"]), "relevance": _score(by_number[str(i)]["relevance"])}
        for i in range(1, count + 1)
    }


def judge_answers(questions: list[dict], results: dict) -> dict:
    """
    Score every approach's answer to each question in one judge call. Answers are anonymized and their
    order is shuffled per question (seeded by question id) to avoid label and position bias.
    Returns {"per_question": {qid: {approach: {correctness, relevance}}}, "aggregate": {approach: {...}}}.
    """
    client = config.make_anthropic_client()
    approaches = list(results.keys())
    per_question: dict[str, dict] = {}

    for q in questions:
        qid = q["id"]
        order = approaches[:]
        random.Random(qid).shuffle(order)
        answers = "\n\n".join(
            f"Answer {i}:\n{results[a].get(qid, {}).get('answer') or '(no answer)'}" for i, a in enumerate(order, 1)
        )
        example = json.dumps({str(i): {"correctness": "<int>", "relevance": "<int>"} for i in range(1, len(order) + 1)})
        prompt = JUDGE_PROMPT.format(question=q["question"], reference=q["reference_answer"], answers=answers, example=example)

        last_error = None
        for _attempt in range(2):
            try:
                response = client.messages.create(
                    model=config.JUDGE_MODEL,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                text_blocks = [b.text for b in response.content if hasattr(b, "text")]
                if not text_blocks:
                    raise ValueError(f"No text block in response; got: {[type(b).__name__ for b in response.content]}")
                by_number = _scores_for(_parse_json(text_blocks[0]), len(order))
                per_question[qid] = {a: by_number[str(i)] for i, a in enumerate(order, 1)}
                print(f"  {qid}: " + "  ".join(f"{a}={per_question[qid][a]}" for a in approaches))
                last_error = None
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, anthropic.APIError) as e:
                last_error = e
        if last_error is not None:
            print(f"  {qid}: judge error after retry — {last_error}")
            per_question[qid] = {a: {"correctness": 0, "relevance": 0} for a in approaches}
            per_question[qid]["error"] = str(last_error)

    aggregate: dict[str, dict] = {}
    for approach in approaches:
        scores = [per_question[q["id"]][approach] for q in questions]
        times = [results[approach].get(q["id"], {}).get("elapsed", 0.0) for q in questions]
        aggregate[approach] = {
            "avg_correctness": round(sum(s["correctness"] for s in scores) / len(scores), 2),
            "avg_relevance": round(sum(s["relevance"] for s in scores) / len(scores), 2),
            "avg_time": round(sum(times) / len(times), 2),
            "n": len(scores),
        }
        costs = [results[approach].get(q["id"], {}).get("cost_usd") for q in questions]
        if any(c is not None for c in costs):
            aggregate[approach]["total_cost_usd"] = round(sum(c or 0 for c in costs), 4)
        usages = [results[approach].get(q["id"], {}).get("usage") or {} for q in questions]
        aggregate[approach]["avg_input_tokens"] = round(sum(u.get("input_tokens", 0) for u in usages) / len(usages))
        aggregate[approach]["avg_output_tokens"] = round(sum(u.get("output_tokens", 0) for u in usages) / len(usages))

    return {"per_question": per_question, "aggregate": aggregate}
