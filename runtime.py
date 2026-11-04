"""Data, prompts, inference, and scoring helpers for MuSLR."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


ANSWER_RE = re.compile(
    r"(?im)^\s*(?:final\s+)?answer\s*:\s*\{?\s*"
    r"(true|false|unknown|[a-d])\s*\}?\s*(?:\.|$)"
)
FALLBACK_ANSWER_RE = re.compile(
    r"(?i)(?:final\s+answer|answer)\s+(?:is|would\s+be)\s*"
    r"\{?\s*(true|false|unknown|[a-d])\s*\}?"
)
EMBEDDED_IMAGE_RE = re.compile(r"(?m)^Image: (?P<payload>[A-Za-z0-9+/=]+)\s*$")


@dataclass(frozen=True)
class Sample:
    id: str
    source_id: str
    row_index: int
    image: Any
    domain: str
    symbol: str
    depth: int | None
    context: str
    question: str
    choices: list[str]
    answer: str
    reasoning: dict[str, str] = field(default_factory=dict)

    @property
    def task_type(self) -> str:
        return "multiple_choice" if self.choices else "truth_evaluation"


def _parse_json_maybe(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _records_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise ValueError("Dataset JSON must be an object or a list of objects.")

    for key in ("data", "train", "records", "examples"):
        candidate = data.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

    list_values = [value for value in data.values() if isinstance(value, list)]
    if len(list_values) == 1:
        return [item for item in list_values[0] if isinstance(item, dict)]
    return [data]


def _resolve_dataset_file(dataset_path: Path) -> Path:
    if dataset_path.is_dir():
        for name in ("metadata.csv", "metadata.jsonl", "metadata.json", "muslr_context.json", "muslr.json"):
            candidate = dataset_path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No supported MuSLR metadata file found under {dataset_path}")
    return dataset_path


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    metadata_path = _resolve_dataset_file(dataset_path)
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {metadata_path}:{line_number}")
            records.append(value)
        return records
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return _records_from_json(data)


def load_samples(dataset_path: Path) -> list[Sample]:
    """Load GitHub JSON or a JSON export of the Hugging Face dataset."""
    samples: list[Sample] = []
    seen_ids: dict[str, int] = {}
    for index, record in enumerate(_load_records(dataset_path)):
        raw_choices = _parse_json_maybe(record.get("choices", record.get("options", [])), [])
        if raw_choices is None:
            raw_choices = []
        if isinstance(raw_choices, str):
            raw_choices = [line.strip() for line in raw_choices.splitlines() if line.strip()]
        if not isinstance(raw_choices, list):
            raise ValueError(f"Record {index} has unsupported choices: {type(raw_choices).__name__}")

        raw_reasoning = _parse_json_maybe(record.get("reasoning", {}), {})
        if isinstance(raw_reasoning, dict):
            reasoning = {
                str(label): str(step).strip()
                for label, step in raw_reasoning.items()
                if str(step).strip()
            }
        elif isinstance(raw_reasoning, list):
            reasoning = {
                f"R{step_index}": str(step).strip()
                for step_index, step in enumerate(raw_reasoning, start=1)
                if str(step).strip()
            }
        elif isinstance(raw_reasoning, str) and raw_reasoning.strip():
            reasoning = {"R1": raw_reasoning.strip()}
        else:
            reasoning = {}

        image = record.get("image_file", record.get("file_name", record.get("image")))
        if image is None:
            raise ValueError(f"Record {index} has no image_file, file_name, or image field.")

        depth = record.get("depth")
        try:
            normalized_depth = int(depth) if depth is not None else None
        except (TypeError, ValueError):
            normalized_depth = None

        source_id = str(record.get("id", index))
        seen_ids[source_id] = seen_ids.get(source_id, 0) + 1
        sample_id = source_id if seen_ids[source_id] == 1 else f"{source_id}#{seen_ids[source_id]}"
        samples.append(
            Sample(
                id=sample_id,
                source_id=source_id,
                row_index=index,
                image=image,
                domain=str(record.get("domain", "unknown")),
                symbol=str(record.get("symbol", "unknown")).lower(),
                depth=normalized_depth,
                context=str(record.get("full_context", record.get("context", ""))).strip(),
                question=str(record.get("question", "")).strip(),
                choices=[str(choice).strip() for choice in raw_choices],
                answer=str(record.get("answer", record.get("label", ""))).strip(),
                reasoning=reasoning,
            )
        )
    return samples


def _resolve_local_image_path(value: Any, dataset_dir: Path, image_root: Path | None) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path") or value.get("local_path") or value.get("filename")

    if not isinstance(value, str) or not value.strip():
        return None
    image_ref = value.strip()
    if image_ref.startswith(("http://", "https://", "data:")):
        return None

    candidate = Path(image_ref)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if image_root is not None:
            candidates.append(image_root / candidate.name)
            candidates.append(image_root / candidate)
        candidates.append(dataset_dir / candidate)
        candidates.append(dataset_dir / candidate.name)

    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _decode_image_bytes(value: Any, dataset_dir: Path, image_root: Path | None) -> tuple[bytes, str]:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            raw = value["bytes"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            if not isinstance(raw, bytes):
                raise ValueError("Hugging Face image.bytes is not bytes or base64 text.")
            return raw, "image/jpeg"
        value = value.get("path") or value.get("src") or value.get("url")

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Image reference is empty or unsupported.")

    image_ref = value.strip()
    if image_ref.startswith(("http://", "https://")):
        with urllib.request.urlopen(image_ref, timeout=60) as response:
            content_type = response.headers.get_content_type()
            return response.read(), content_type if content_type.startswith("image/") else "image/jpeg"

    candidate = Path(image_ref)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if image_root is not None:
            candidates.append(image_root / candidate.name)
            candidates.append(image_root / candidate)
        candidates.append(dataset_dir / candidate)
        candidates.append(dataset_dir / candidate.name)

    for path in candidates:
        if path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return path.read_bytes(), mime
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate image {image_ref!r}. Checked: {checked}")


def _mime_suffix(mime: str) -> str:
    if mime in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    return ".img"


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    temporary = target.with_suffix(target.suffix + f".{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(target)


def _write_bytes_once(data: bytes, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == len(data):
        return
    temporary = target.with_suffix(target.suffix + f".{uuid.uuid4().hex}.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(data)
    temporary.replace(target)


def sample_image_data_url(sample: Sample, dataset_dir: Path, image_root: Path | None) -> str:
    image_bytes, mime = _decode_image_bytes(sample.image, dataset_dir, image_root)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def sample_image_local_url(
    sample: Sample,
    dataset_dir: Path,
    image_root: Path | None,
    local_media_root: Path,
) -> str:
    local_media_root = local_media_root.resolve()
    local_path = _resolve_local_image_path(sample.image, dataset_dir, image_root)
    if local_path is not None:
        target = local_media_root / "dataset_images" / local_path.name
        _link_or_copy(local_path, target)
        return target.resolve().as_uri()

    image_bytes, mime = _decode_image_bytes(sample.image, dataset_dir, image_root)
    digest = hashlib.sha256(image_bytes).hexdigest()[:20]
    target = local_media_root / "dataset_images" / f"{sample.id}_{digest}{_mime_suffix(mime)}"
    _write_bytes_once(image_bytes, target)
    return target.resolve().as_uri()


def template_path(template_dir: Path, method: str, symbol: str) -> Path:
    if method == "baseline":
        return template_dir / "baseline" / "baseline_CoT.txt"
    if method == "logicam":
        filename = "prompt_fol.txt" if symbol == "fol" else "prompt.txt"
        return template_dir / "logicam" / filename
    raise ValueError(f"Unsupported method: {method}")


def parse_template_content(
    template: str,
    local_media_root: Path | None = None,
    template_name: str = "template",
) -> list[dict[str, Any]]:
    """Replace official prompt JPEG base64 lines with OpenAI image content."""
    content: list[dict[str, Any]] = []
    position = 0
    for match in EMBEDDED_IMAGE_RE.finditer(template):
        payload = match.group("payload")
        try:
            decoded = base64.b64decode(payload, validate=True)
        except ValueError:
            continue
        if not decoded.startswith(b"\xff\xd8\xff"):
            continue
        text = template[position : match.start()] + "Image:"
        if text:
            content.append({"type": "text", "text": text})
        if local_media_root is not None:
            digest = hashlib.sha256(decoded).hexdigest()[:20]
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", template_name)
            target = local_media_root.resolve() / "prompt_images" / f"{safe_name}_{digest}.jpg"
            _write_bytes_once(decoded, target)
            url = target.resolve().as_uri()
        else:
            url = f"data:image/jpeg;base64,{payload}"
        content.append({"type": "image_url", "image_url": {"url": url}})
        position = match.end()
    suffix = template[position:]
    if suffix:
        content.append({"type": "text", "text": suffix})
    return content or [{"type": "text", "text": template}]


def problem_text(sample: Sample) -> str:
    sections = ["\n\n---\n\nNew instance:\nImage:"]
    sections.extend(["\n\nContext:\n", sample.context, "\n\nQuestion:\n", sample.question])
    if sample.choices:
        sections.extend(["\n\nOptions:\n", "\n".join(sample.choices)])
        answer_format = "one option letter (A, B, C, or D)"
    else:
        answer_format = "one label (True, False, or Unknown)"
    sections.extend(
        [
            "\n\nReason carefully from the image and context. End with exactly "
            f"`Answer: {{{answer_format}}}` and do not put any other label after it.",
        ]
    )
    return "".join(sections)


def build_message_content(
    sample: Sample,
    method: str,
    template_dir: Path,
    dataset_dir: Path,
    image_root: Path | None,
    media_mode: str,
    local_media_root: Path | None,
) -> list[dict[str, Any]]:
    path = template_path(template_dir, method, sample.symbol)
    template = path.read_text(encoding="utf-8")
    if media_mode == "local":
        if local_media_root is None:
            raise ValueError("local_media_root is required when media_mode='local'.")
        content = parse_template_content(template, local_media_root=local_media_root, template_name=path.name)
        image_url = sample_image_local_url(sample, dataset_dir, image_root, local_media_root)
    elif media_mode == "data-url":
        content = parse_template_content(template)
        image_url = sample_image_data_url(sample, dataset_dir, image_root)
    else:
        raise ValueError(f"Unsupported media_mode: {media_mode}")
    content.append({"type": "text", "text": problem_text(sample)})
    content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def parse_answer(text: str | None, task_type: str) -> str | None:
    if not text:
        return None
    matches = ANSWER_RE.findall(text)
    candidate = matches[-1] if matches else None
    if candidate is None:
        fallback_matches = FALLBACK_ANSWER_RE.findall(text)
        candidate = fallback_matches[-1] if fallback_matches else None
    if candidate is None:
        return None
    normalized = candidate.upper()
    if task_type == "multiple_choice":
        return normalized if normalized in {"A", "B", "C", "D"} else None
    return normalized if normalized in {"TRUE", "FALSE", "UNKNOWN"} else None


class CompletionClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout: float,
        retries: int,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.local = threading.local()

    def _client(self) -> OpenAI:
        client = getattr(self.local, "client", None)
        if client is None:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
            self.local.client = client
        return client

    def complete(self, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None, str | None]:
        retryable = (APIConnectionError, APITimeoutError, RateLimitError, APIError)
        for attempt in range(self.retries + 1):
            try:
                response = self._client().chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage.model_dump() if response.usage else None
                return choice.message.content or "", usage, choice.finish_reason
            except retryable:
                if attempt == self.retries:
                    raise
                time.sleep(min(2**attempt, 8))
        raise RuntimeError("Unreachable")


def run_one(
    sample: Sample,
    completion_client: CompletionClient,
    method: str,
    template_dir: Path,
    dataset_dir: Path,
    image_root: Path | None,
    media_mode: str,
    local_media_root: Path | None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        content = build_message_content(
            sample,
            method,
            template_dir,
            dataset_dir,
            image_root,
            media_mode,
            local_media_root,
        )
        output, usage, finish_reason = completion_client.complete(content)
        prediction = parse_answer(output, sample.task_type)
        return {
            "id": sample.id,
            "source_id": sample.source_id,
            "row_index": sample.row_index,
            "status": "completed" if prediction else "unparsed",
            "prediction": prediction,
            "output": output,
            "finish_reason": finish_reason,
            "usage": usage,
            "latency_seconds": round(time.monotonic() - started, 3),
            "task_type": sample.task_type,
            "domain": sample.domain,
            "symbol": sample.symbol,
            "depth": sample.depth,
        }
    except FileNotFoundError as error:
        return {
            "id": sample.id,
            "source_id": sample.source_id,
            "row_index": sample.row_index,
            "status": "skipped_missing_image",
            "error": str(error),
        }
    except Exception as error:  # Preserve the checkpoint and let a later --resume retry it.
        return {
            "id": sample.id,
            "source_id": sample.source_id,
            "row_index": sample.row_index,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "latency_seconds": round(time.monotonic() - started, 3),
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if isinstance(value, dict):
            records.append(value)
    return records


def selected_samples(
    samples: Iterable[Sample],
    ids: set[str] | None = None,
    domains: set[str] | None = None,
    symbols: set[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> list[Sample]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    selected: list[Sample] = []
    for index, sample in enumerate(samples):
        if index % num_shards != shard_index:
            continue
        if ids and sample.id not in ids and sample.source_id not in ids:
            continue
        if domains and sample.domain.lower() not in domains:
            continue
        if symbols and sample.symbol not in symbols:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def run_samples(
    samples: list[Sample],
    output_path: Path,
    completion_client: CompletionClient,
    method: str,
    template_dir: Path,
    dataset_dir: Path,
    image_root: Path | None,
    media_mode: str,
    local_media_root: Path | None,
    workers: int,
    resume: bool,
    retry_statuses: set[str] | None = None,
) -> dict[str, int]:
    previous = load_jsonl(output_path) if resume else []
    previous_by_id = {
        str(record["id"]): record for record in previous if record.get("id") is not None
    }
    if retry_statuses:
        pending = [
            sample
            for sample in samples
            if sample.id not in previous_by_id
            or previous_by_id[sample.id].get("status") in retry_statuses | {"failed"}
        ]
    else:
        done_ids = {
            record_id
            for record_id, record in previous_by_id.items()
            if record.get("status") in {"completed", "unparsed", "skipped_missing_image"}
        }
        pending = [sample for sample in samples if sample.id not in done_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retry_path = output_path.with_suffix(output_path.suffix + ".retrying")
    active_output_path = retry_path if retry_statuses else output_path
    mode = "w" if retry_statuses else ("a" if resume and output_path.exists() else "w")

    counters = {"selected": len(samples), "resumed": len(samples) - len(pending), "completed": 0, "unparsed": 0, "failed": 0, "skipped_missing_image": 0}
    if not pending:
        return counters

    def progress_status() -> str:
        return (
            f"ok:{counters['completed']} "
            f"unparsed:{counters['unparsed']} "
            f"failed:{counters['failed']} "
            f"skipped:{counters['skipped_missing_image']}"
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(separator="/"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[status]}"),
    )
    with active_output_path.open(mode, encoding="utf-8") as output_file:
        with progress:
            task_id = progress.add_task(
                f"{method} inference",
                total=len(samples),
                completed=counters["resumed"],
                status=progress_status(),
            )
            def on_outcome(record: dict[str, Any]) -> None:
                output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                output_file.flush()
                status = str(record["status"])
                counters[status] = counters.get(status, 0) + 1
                progress.advance(task_id)
                progress.update(task_id, status=progress_status())

            if getattr(completion_client, "uses_ripo_batch", False):
                completion_client.run_batch(
                    pending,
                    lambda client, sample: run_one(
                        sample,
                        client,
                        method,
                        template_dir,
                        dataset_dir,
                        image_root,
                        media_mode,
                        local_media_root,
                    ),
                    max_workers=workers,
                    task_id=lambda sample: sample.id,
                    on_outcome=on_outcome,
                )
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures: dict[Future[dict[str, Any]], Sample] = {
                        executor.submit(
                            run_one,
                            sample,
                            completion_client,
                            method,
                            template_dir,
                            dataset_dir,
                            image_root,
                            media_mode,
                            local_media_root,
                        ): sample
                        for sample in pending
                    }
                    for future in as_completed(futures):
                        on_outcome(future.result())
    if retry_statuses:
        retry_records = load_jsonl(retry_path)
        for record in retry_records:
            record_id = str(record.get("id", ""))
            if record_id and record.get("status") != "failed":
                previous_by_id[record_id] = record
        ordered_records = sorted(
            previous_by_id.values(),
            key=lambda record: (
                int(record.get("row_index", 2**31 - 1)),
                str(record.get("id", "")),
            ),
        )
        with retry_path.open("w", encoding="utf-8") as output_file:
            for record in ordered_records:
                output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        retry_path.replace(output_path)
    return counters


def sample_index(samples: Iterable[Sample]) -> dict[str, Sample]:
    return {sample.id: sample for sample in samples}


def summarize_results(records: Iterable[dict[str, Any]], samples_by_id: dict[str, Sample]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = str(record.get("id", ""))
        if record_id in samples_by_id:
            unique[record_id] = record

    total = len(unique)
    parsed = 0
    correct = 0
    status_counts: dict[str, int] = {}
    for record_id, record in unique.items():
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        predicted = record.get("prediction")
        if not isinstance(predicted, str):
            continue
        parsed += 1
        expected = samples_by_id[record_id].answer.upper()
        if predicted.upper() == expected:
            correct += 1

    return {
        "records": total,
        "parsed": parsed,
        "correct": correct,
        "coverage_percent": round(100 * parsed / total, 2) if total else 0.0,
        "accuracy_scored_percent": round(100 * correct / parsed, 2) if parsed else 0.0,
        "accuracy_end_to_end_percent": round(100 * correct / total, 2) if total else 0.0,
        "status_counts": status_counts,
    }


def grouped_summary(
    records: Iterable[dict[str, Any]], samples_by_id: dict[str, Sample], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        sample = samples_by_id.get(str(record.get("id", "")))
        if sample is None:
            continue
        value = getattr(sample, field)
        groups.setdefault(str(value), []).append(record)
    return {name: summarize_results(group, samples_by_id) for name, group in sorted(groups.items())}


def metadata_for_samples(samples: list[Sample], args: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_samples": len(samples),
        "sample_ids": [sample.id for sample in samples],
        "configuration": args,
    }
