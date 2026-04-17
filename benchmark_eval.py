import re
import subprocess
import sys
import tempfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any


SUPPORTED_EVAL_DATASETS = frozenset({"gsm8k", "humaneval", "mbpp"})
CODE_EVAL_DATASETS = frozenset({"humaneval", "mbpp"})
DEFAULT_EVAL_TIMEOUT_SEC = 5.0

CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def dataset_supports_eval(dataset_name: str) -> bool:
    return dataset_name in SUPPORTED_EVAL_DATASETS


def evaluate_benchmark_run(
    dataset_name: str,
    dataset: Any,
    responses: list[dict[str, Any]],
    response_metadata: list[dict[str, Any]],
    tokenizer: Any,
    timeout_sec: float = DEFAULT_EVAL_TIMEOUT_SEC,
) -> dict[str, Any]:
    result = {
        "enabled": True,
        "supported": dataset_supports_eval(dataset_name),
        "dataset": dataset_name,
        "metric_name": "accuracy" if dataset_name == "gsm8k" else "pass@1",
        "timeout_sec": float(timeout_sec) if dataset_name in CODE_EVAL_DATASETS else None,
        "methods": {},
        "warnings": [],
    }
    if not result["supported"]:
        result["message"] = f"Dataset `{dataset_name}` does not have an implemented eval backend."
        return result

    if not responses:
        result["message"] = "No responses available to evaluate."
        return result

    if len(responses) != len(response_metadata):
        result["warnings"].append(
            f"response count {len(responses)} != metadata count {len(response_metadata)}; evaluation may be incomplete."
        )

    methods = list(responses[0].keys())
    evaluation_examples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped_turns = 0
    for response, metadata in zip(responses, response_metadata):
        dataset_index = int(metadata.get("dataset_index", -1))
        turn_index = int(metadata.get("turn_index", 0))
        if turn_index != 0:
            skipped_turns += 1
            continue
        if dataset_index < 0 or dataset_index >= len(dataset):
            result["warnings"].append(f"Skipping invalid dataset index {dataset_index}.")
            continue
        evaluation_examples.append((dataset[dataset_index], response))

    if skipped_turns > 0:
        result["warnings"].append(
            f"Skipped {skipped_turns} non-zero turn responses; eval mode currently scores only single-turn outputs."
        )

    for method in methods:
        status_counts: Counter[str] = Counter()
        num_correct = 0
        num_prediction_available = 0

        for example, response in evaluation_examples:
            method_response = response.get(method)
            if method_response is None:
                status_counts["missing_response"] += 1
                continue

            generated_ids = method_response.output_ids[0, method_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            sample_result = evaluate_sample(
                dataset_name=dataset_name,
                example=example,
                output_text=output_text,
                timeout_sec=timeout_sec,
            )
            status_counts[str(sample_result["status"])] += 1
            num_correct += int(bool(sample_result["passed"]))
            num_prediction_available += int(bool(sample_result.get("prediction_available", True)))

        num_total = len(evaluation_examples)
        result["methods"][method] = {
            "metric_name": result["metric_name"],
            "score": (num_correct / num_total) if num_total else None,
            "num_correct": int(num_correct),
            "num_total": int(num_total),
            "num_prediction_available": int(num_prediction_available),
            "status_counts": dict(sorted(status_counts.items())),
        }

    result["num_examples"] = len(evaluation_examples)
    return result


def evaluate_sample(
    dataset_name: str,
    example: dict[str, Any],
    output_text: str,
    timeout_sec: float,
) -> dict[str, Any]:
    if dataset_name == "gsm8k":
        return evaluate_gsm8k_sample(example, output_text)
    if dataset_name == "humaneval":
        return evaluate_humaneval_sample(example, output_text, timeout_sec)
    if dataset_name == "mbpp":
        return evaluate_mbpp_sample(example, output_text, timeout_sec)
    return {
        "passed": False,
        "status": "unsupported_dataset",
        "prediction_available": False,
    }


def evaluate_gsm8k_sample(example: dict[str, Any], output_text: str) -> dict[str, Any]:
    target_answer = extract_gsm8k_reference_answer(example)
    predicted_answer = extract_gsm8k_prediction_answer(output_text)
    if target_answer is None:
        return {
            "passed": False,
            "status": "missing_reference",
            "prediction_available": predicted_answer is not None,
        }
    if predicted_answer is None:
        return {
            "passed": False,
            "status": "missing_prediction",
            "prediction_available": False,
        }
    is_correct = predicted_answer == target_answer
    return {
        "passed": is_correct,
        "status": "correct" if is_correct else "wrong",
        "prediction_available": True,
    }


def evaluate_humaneval_sample(
    example: dict[str, Any],
    output_text: str,
    timeout_sec: float,
) -> dict[str, Any]:
    prompt = str(example.get("prompt", ""))
    entry_point = str(example.get("entry_point", "")).strip()
    prompt_prefix = prompt.split(f"def {entry_point}", 1)[0] if entry_point and f"def {entry_point}" in prompt else ""
    candidate_snippets = extract_python_candidates(output_text, entry_point=entry_point)
    if not candidate_snippets:
        candidate_snippets = [output_text]

    source_candidates = []
    for snippet in candidate_snippets:
        cleaned_snippet = snippet.lstrip("\n")
        if entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\b", cleaned_snippet):
            source_candidates.append(trim_python_suffix(prompt_prefix + cleaned_snippet))
        else:
            source_candidates.append(trim_python_suffix(prompt + cleaned_snippet))

    test_source = f"{str(example.get('test', '')).rstrip()}\n\ncheck({entry_point})\n"
    return run_python_candidates(source_candidates, test_source, timeout_sec)


def evaluate_mbpp_sample(
    example: dict[str, Any],
    output_text: str,
    timeout_sec: float,
) -> dict[str, Any]:
    entry_point = infer_mbpp_entry_point(example)
    candidate_snippets = extract_python_candidates(output_text, entry_point=entry_point)
    if not candidate_snippets:
        candidate_snippets = [output_text]

    setup_sections = []
    test_setup_code = str(example.get("test_setup_code", "") or "").strip()
    if test_setup_code:
        setup_sections.append(test_setup_code)
    test_imports = example.get("test_imports", []) or []
    if isinstance(test_imports, list):
        setup_sections.extend(str(import_line).strip() for import_line in test_imports if str(import_line).strip())
    else:
        raw_imports = str(test_imports).strip()
        if raw_imports:
            setup_sections.append(raw_imports)
    setup_prefix = "\n".join(setup_sections).strip()

    source_candidates = []
    for snippet in candidate_snippets:
        cleaned_snippet = trim_python_suffix(snippet)
        if setup_prefix:
            source_candidates.append(f"{setup_prefix}\n\n{cleaned_snippet}")
        else:
            source_candidates.append(cleaned_snippet)

    tests = example.get("test_list", []) or []
    test_source = "\n".join(str(test_line).rstrip() for test_line in tests if str(test_line).strip()) + "\n"
    return run_python_candidates(source_candidates, test_source, timeout_sec)


def infer_mbpp_entry_point(example: dict[str, Any]) -> str | None:
    for key in ("function_name", "entry_point"):
        value = str(example.get(key, "") or "").strip()
        if value:
            return value

    tests = example.get("test_list", []) or []
    for test_line in tests:
        test_line = str(test_line)
        match = re.search(r"assert\s+(?:set\()?(?:sorted\()?(?:list\()?\s*([A-Za-z_]\w*)\s*\(", test_line)
        if match is not None:
            return match.group(1)
    return None


def extract_python_candidates(output_text: str, entry_point: str | None = None) -> list[str]:
    candidates: list[str] = []

    for block in CODE_BLOCK_PATTERN.findall(output_text):
        block = block.strip()
        if block:
            candidates.append(block)

    if entry_point:
        match = re.search(rf"\bdef\s+{re.escape(entry_point)}\b", output_text)
        if match is not None:
            candidates.append(output_text[match.start() :].strip())
    else:
        generic_match = re.search(r"\b(?:def|class)\s+[A-Za-z_]\w*", output_text)
        if generic_match is not None:
            candidates.append(output_text[generic_match.start() :].strip())

    stripped_output = output_text.strip()
    if stripped_output:
        candidates.append(stripped_output)

    unique_candidates: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append(normalized)
    return unique_candidates


def trim_python_suffix(source: str) -> str:
    lines = source.rstrip().splitlines()
    while lines:
        candidate = "\n".join(lines).rstrip() + "\n"
        try:
            compile(candidate, "<candidate>", "exec")
            return candidate
        except SyntaxError:
            lines.pop()
    return source.rstrip() + "\n"


def run_python_candidates(
    source_candidates: list[str],
    test_source: str,
    timeout_sec: float,
) -> dict[str, Any]:
    best_result = {
        "passed": False,
        "status": "empty_output",
        "prediction_available": False,
    }
    best_priority = failure_priority(best_result["status"])

    for source_candidate in source_candidates:
        source_candidate = source_candidate.strip()
        if not source_candidate:
            continue

        run_result = run_python_program(source_candidate, test_source, timeout_sec)
        if run_result["passed"]:
            return run_result

        candidate_priority = failure_priority(run_result["status"])
        if candidate_priority > best_priority:
            best_result = run_result
            best_priority = candidate_priority

    return best_result


def run_python_program(
    source_code: str,
    test_source: str,
    timeout_sec: float,
) -> dict[str, Any]:
    program = source_code.rstrip() + "\n\n" + test_source.rstrip() + "\n"
    try:
        compile(program, "<benchmark_eval>", "exec")
    except SyntaxError:
        return {
            "passed": False,
            "status": "syntax_error",
            "prediction_available": True,
        }

    with tempfile.TemporaryDirectory(prefix="benchmark_eval_") as temp_dir:
        script_path = Path(temp_dir) / "candidate.py"
        script_path.write_text(program, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-I", str(script_path)],
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _, stderr = process.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return {
                "passed": False,
                "status": "timeout",
                "prediction_available": True,
            }

    if process.returncode == 0:
        return {
            "passed": True,
            "status": "passed",
            "prediction_available": True,
        }

    stderr = stderr or ""
    if "AssertionError" in stderr:
        status = "failed"
    elif "SyntaxError" in stderr or "IndentationError" in stderr:
        status = "syntax_error"
    else:
        status = "runtime_error"
    return {
        "passed": False,
        "status": status,
        "prediction_available": True,
    }


def failure_priority(status: str) -> int:
    return {
        "empty_output": 0,
        "missing_prediction": 0,
        "syntax_error": 1,
        "failed": 2,
        "runtime_error": 2,
        "timeout": 2,
        "wrong": 2,
        "correct": 3,
        "passed": 3,
    }.get(status, 0)


def extract_gsm8k_reference_answer(example: dict[str, Any]) -> str | None:
    raw_answer = str(example.get("answer", "") or "")
    boxed_answer = extract_last_boxed_content(raw_answer)
    if boxed_answer is not None:
        return normalize_math_answer(boxed_answer)

    marker_match = re.findall(r"####\s*([^\n]+)", raw_answer)
    if marker_match:
        marked_answer = marker_match[-1]
        fallback_number = extract_last_number(marked_answer)
        if fallback_number is not None:
            return normalize_math_answer(fallback_number)
        return normalize_math_answer(marked_answer)

    fallback_number = extract_last_number(raw_answer)
    if fallback_number is not None:
        return normalize_math_answer(fallback_number)
    return None


def extract_gsm8k_prediction_answer(output_text: str) -> str | None:
    boxed_answer = extract_last_boxed_content(output_text)
    if boxed_answer is not None:
        return normalize_math_answer(boxed_answer)

    marker_match = re.findall(r"####\s*([^\n]+)", output_text)
    if marker_match:
        marked_answer = marker_match[-1]
        fallback_number = extract_last_number(marked_answer)
        if fallback_number is not None:
            return normalize_math_answer(fallback_number)
        return normalize_math_answer(marked_answer)

    phrase_match = re.findall(r"final answer is[:\s]+([^\n]+)", output_text, flags=re.IGNORECASE)
    if phrase_match:
        phrase_answer = phrase_match[-1]
        fallback_number = extract_last_number(phrase_answer)
        if fallback_number is not None:
            return normalize_math_answer(fallback_number)
        return normalize_math_answer(phrase_answer)

    fallback_number = extract_last_number(output_text)
    if fallback_number is not None:
        return normalize_math_answer(fallback_number)
    return None


def extract_last_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    search_end = len(text)
    while True:
        marker_index = text.rfind(marker, 0, search_end)
        if marker_index < 0:
            return None
        cursor = marker_index + len(marker)
        depth = 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[marker_index + len(marker) : cursor]
            cursor += 1
        search_end = marker_index


def extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace("\\,", ""))
    if not matches:
        return None
    return matches[-1]


def normalize_math_answer(answer: str) -> str:
    normalized = answer.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\,", ",")
    normalized = normalized.strip()

    while True:
        updated = normalized
        if updated.startswith("{") and updated.endswith("}"):
            updated = updated[1:-1].strip()
        if updated.startswith("(") and updated.endswith(")"):
            updated = updated[1:-1].strip()
        if updated == normalized:
            break
        normalized = updated

    normalized = normalized.replace(",", "").strip()
    if normalized.endswith("."):
        normalized = normalized.rstrip(".").strip()
    if normalized.startswith("+"):
        normalized = normalized[1:]

    if re.fullmatch(r"-?\d+/\d+", normalized):
        fraction = Fraction(normalized)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"

    if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
        try:
            value = Decimal(normalized)
        except InvalidOperation:
            return normalized
        value_str = format(value.normalize(), "f")
        if "." in value_str:
            value_str = value_str.rstrip("0").rstrip(".")
        if value_str == "-0":
            value_str = "0"
        return value_str

    return normalized
