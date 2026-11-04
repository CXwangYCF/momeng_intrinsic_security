#!/usr/bin/env python3
"""Run JSON-constrained LogiCAM on MuSLR through the RIPO VLM client."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import request
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
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

WORKSPACE = Path(__file__).resolve().parents[2]
load_dotenv(WORKSPACE / ".env", override=False)
RIPO_ROOT = Path(os.environ.get("RIPO_ROOT", WORKSPACE.parent / "RIPO")).expanduser()
for source_root in (WORKSPACE, RIPO_ROOT / "src"):
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from pi_engine.context import msg
from pi_engine.inference import PIVLM
from ripo.agent.client import (
    build_ripo_client_from_config,
    resolve_model_config,
)
from ripo.configs.core.model import ModelConfig

from logicam_json.schema import (
    RETURN_EXAMPLE,
    SCHEMA_VERSION,
    problem_text,
    response_format_for,
)
from muslr_core.runtime import (
    Sample,
    load_jsonl,
    load_samples,
    parse_template_content,
    selected_samples,
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def parse_endpoints(value: str) -> tuple[str, ...]:
    endpoints = tuple(part.strip() for part in value.split(",") if part.strip())
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    return endpoints


def resolve_image(sample: Sample, data_path: Path) -> Path:
    value: Any = sample.image
    if isinstance(value, dict):
        value = value.get("path") or value.get("local_path") or value.get("filename")
    if not isinstance(value, str) or not value.strip():
        raise FileNotFoundError(f"Sample {sample.id} has no local image path")
    image = Path(value.strip())
    dataset_dir = data_path if data_path.is_dir() else data_path.parent
    candidates = (
        [image]
        if image.is_absolute()
        else [
            dataset_dir / image,
            dataset_dir / "images" / image.name,
            dataset_dir / image.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Image for {sample.id} not found; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def template_path(template_dir: Path, symbol: str) -> Path:
    filename = "prompt_fol.txt" if symbol.lower() == "fol" else "prompt.txt"
    return template_dir / "logicam" / filename


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def template_blocks(path: Path, media_root: Path) -> tuple[dict[str, Any], ...]:
    """Load every byte of the official prompt and materialize its example images."""
    return tuple(
        parse_template_content(
            path.read_text(encoding="utf-8"),
            local_media_root=media_root,
            template_name=path.name,
        )
    )


def logicam_messages(
    sample: Sample,
    *,
    data_path: Path,
    template_dir: Path,
    media_root: Path,
) -> tuple[list[Any], Path]:
    prompt_path = template_path(template_dir, sample.symbol).resolve()
    blocks = []
    for block in template_blocks(prompt_path, media_root.resolve()):
        if block.get("type") == "text":
            blocks.append(msg.text(str(block.get("text", ""))))
            continue
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else None
        if not isinstance(url, str) or not url.startswith("file://"):
            raise ValueError("Official LogiCAM example image was not materialized locally")
        blocks.append(msg.image(Path(unquote(urlparse(url).path))))
    blocks.append(msg.text(problem_text(sample)))
    blocks.append(msg.image(resolve_image(sample, data_path)))
    return [msg.user(blocks)], prompt_path


def served_model_roots(
    endpoints: tuple[str, ...], api_key: str
) -> dict[str, list[str]]:
    roots: dict[str, list[str]] = {}
    for endpoint in endpoints:
        headers = (
            {}
            if not api_key or api_key == "EMPTY"
            else {"Authorization": f"Bearer {api_key}"}
        )
        try:
            http_request = request.Request(
                f"{endpoint.rstrip('/')}/models", headers=headers
            )
            with request.urlopen(http_request, timeout=3) as response:
                payload = json.load(response)
            roots[endpoint] = [
                str(item.get("root") or item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict)
            ]
        except Exception as error:  # noqa: BLE001 - metadata must not block inference.
            roots[endpoint] = [f"unavailable: {type(error).__name__}: {error}"]
    return roots


def run_one(
    client: Any,
    sample: Sample,
    *,
    data_path: Path,
    template_dir: Path,
    media_root: Path,
    max_tokens: int,
) -> dict[str, Any]:
    raw_output: str | None = None
    usage: dict[str, Any] | None = None
    latency_seconds: float | None = None
    prompt_path: Path | None = None
    try:
        messages, prompt_path = logicam_messages(
            sample,
            data_path=data_path,
            template_dir=template_dir,
            media_root=media_root,
        )
        metadata = client.complete_with_metadata(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=response_format_for(sample),
            max_try=1,
            wait_time=0.0,
        )
        if not metadata.responses:
            raise RuntimeError("RIPO returned no response")
        raw_output = PIVLM.text(metadata.responses[0])
        latency_seconds = metadata.latency_sec
        usage = dict(metadata.usage) if metadata.usage else None
        if usage is not None:
            usage["endpoint"] = metadata.endpoint
        structured = PIVLM.parse(raw_output, return_example=RETURN_EXAMPLE)
        if not isinstance(structured, dict):
            raise TypeError("Structured output is not a JSON object")
        reasoning_steps = structured["reasoning_steps"]
        if not isinstance(reasoning_steps, list) or not all(
            isinstance(step, str) for step in reasoning_steps
        ):
            raise TypeError("reasoning_steps is not a list of strings")
        prediction = str(structured["final_answer"]).upper()
        return {
            "id": sample.id,
            "source_id": sample.source_id,
            "row_index": sample.row_index,
            "status": "completed",
            "prediction": prediction,
            "expected_answer": sample.answer.upper(),
            "correct": prediction == sample.answer.upper(),
            "structured_output": structured,
            "output": "\n\n".join(reasoning_steps),
            "raw_output": raw_output,
            "schema_version": SCHEMA_VERSION,
            "method": "logicam_json_v2",
            "task_type": sample.task_type,
            "domain": sample.domain,
            "symbol": sample.symbol,
            "depth": sample.depth,
            "usage": usage,
            "latency_seconds": latency_seconds,
            "model": metadata.model,
            "prompt_template": str(prompt_path),
            "prompt_sha256": file_sha256(prompt_path),
        }
    except FileNotFoundError as error:
        status = "skipped_missing_image"
        error_text = f"{type(error).__name__}: {error}"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        status = "invalid_json"
        error_text = f"{type(error).__name__}: {error}"
    except Exception as error:  # noqa: BLE001 - checkpoint arbitrary provider failures.
        status = "failed"
        error_text = f"{type(error).__name__}: {error}"
    return {
        "id": sample.id,
        "source_id": sample.source_id,
        "row_index": sample.row_index,
        "status": status,
        "prediction": None,
        "expected_answer": sample.answer.upper(),
        "error": error_text,
        "output": raw_output,
        "usage": usage,
        "latency_seconds": latency_seconds,
        "schema_version": SCHEMA_VERSION,
        "method": "logicam_json_v2",
        "task_type": sample.task_type,
        "domain": sample.domain,
        "symbol": sample.symbol,
        "depth": sample.depth,
        "prompt_template": str(prompt_path) if prompt_path else None,
        "prompt_sha256": file_sha256(prompt_path) if prompt_path else None,
    }


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    samples: list[Sample],
    endpoints: tuple[str, ...],
    roots: dict[str, list[str]],
    prompt_hashes: dict[str, str],
) -> None:
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["api_key"] = "***"
    configuration["max_try"] = 1
    configuration["method"] = "logicam_json_v2"
    configuration["prompt_contract"] = "released_few_shot_prompt_plus_minimal_json_wrapper"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selected_samples": len(samples),
        "sample_ids": [sample.id for sample in samples],
        "resolved_endpoints": list(endpoints),
        "served_model_roots": roots,
        "official_prompt_sha256": prompt_hashes,
        "configuration": configuration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    default_output_dir = env_path(
        "MUSLR_OUTPUT_DIR",
        WORKSPACE / "outputs",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=env_path("MUSLR_DATA", WORKSPACE / "data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=env_path(
            "MUSLR_LOGICAM_JSON_OUTPUT",
            default_output_dir / "muslr_logicam_json_v2_qwen3vl8b.jsonl",
        ),
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=env_path("MUSLR_PROMPT_DIR", WORKSPACE / "prompts"),
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=env_path("MUSLR_LOCAL_MEDIA_ROOT", default_output_dir / "muslr_media"),
    )
    parser.add_argument(
        "--endpoints",
        default=os.environ.get("MUSLR_LOGICAM_JSON_ENDPOINTS", "auto"),
        help="Comma-separated endpoints or 'auto' for RIPO runtime discovery.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MUSLR_LOGICAM_JSON_MODEL", "anom-pi"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MUSLR_LOGICAM_JSON_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--image-url-mode",
        choices=("data", "file"),
        default=os.environ.get("MUSLR_LOGICAM_JSON_IMAGE_URL_MODE", "data"),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=env_int("MUSLR_LOGICAM_JSON_MAX_TOKENS", 4096),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=env_int("MUSLR_LOGICAM_JSON_WORKERS", 2),
    )
    parser.add_argument(
        "--pool-concurrency",
        type=int,
        default=env_int("MUSLR_LOGICAM_JSON_POOL_CONCURRENCY", 2),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=env_float("MUSLR_LOGICAM_JSON_TIMEOUT", 300.0),
    )
    parser.add_argument("--ids", help="Comma-separated sample IDs.")
    parser.add_argument("--domains", help="Comma-separated domain filter.")
    parser.add_argument("--symbols", help="Comma-separated PL, FOL, or NM filters.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, prompts, and schemas without contacting the model.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        parser.error(f"MuSLR data path does not exist: {args.data}")
    if not args.template_dir.is_dir():
        parser.error(f"Prompt directory does not exist: {args.template_dir}")
    if not RIPO_ROOT.is_dir():
        parser.error(f"RIPO root does not exist: {RIPO_ROOT}")
    if args.max_tokens < 1 or args.workers < 1 or args.pool_concurrency < 1:
        parser.error("max-tokens, workers, and pool-concurrency must be positive")
    try:
        raw_endpoints = parse_endpoints(args.endpoints)
    except ValueError as error:
        parser.error(str(error))

    all_samples = load_samples(args.data)
    ids = (
        {part.strip() for part in args.ids.split(",") if part.strip()}
        if args.ids
        else None
    )
    samples = selected_samples(
        all_samples,
        ids=ids,
        domains=parse_csv(args.domains),
        symbols=parse_csv(args.symbols),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    if not samples:
        parser.error("No samples selected")
    used_prompt_paths = {
        template_path(args.template_dir, sample.symbol).resolve() for sample in samples
    }
    missing_prompts = [path for path in used_prompt_paths if not path.is_file()]
    if missing_prompts:
        parser.error("Missing official prompts: " + ", ".join(map(str, missing_prompts)))
    prompt_hashes = {str(path): file_sha256(path) for path in sorted(used_prompt_paths)}
    for sample in samples:
        response_format_for(sample)
        resolve_image(sample, args.data)

    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if args.dry_run:
        for path in used_prompt_paths:
            template_blocks(path, args.media_root.resolve())
        write_metadata(
            metadata_path,
            args=args,
            samples=samples,
            endpoints=raw_endpoints,
            roots={},
            prompt_hashes=prompt_hashes,
        )
        print(
            f"Validated {len(samples)} of {len(all_samples)} samples: {metadata_path}"
        )
        return

    config = resolve_model_config(
        ModelConfig(
            endpoints=raw_endpoints,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            image_url_mode=args.image_url_mode,
            max_try=1,
            wait_time=0.0,
            pool_concurrency=args.pool_concurrency,
        )
    )
    roots = served_model_roots(config.endpoints, config.api_key)
    write_metadata(
        metadata_path,
        args=args,
        samples=samples,
        endpoints=config.endpoints,
        roots=roots,
        prompt_hashes=prompt_hashes,
    )

    previous = [] if args.no_resume else load_jsonl(args.output)
    previous_ids = {str(record.get("id")) for record in previous if record.get("id")}
    pending = [sample for sample in samples if sample.id not in previous_ids]
    counters: Counter[str] = Counter()
    counters["resumed"] = len(samples) - len(pending)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.no_resume else "a"
    workers = min(args.workers, config.max_workers, max(1, len(pending)))
    print(
        f"Selected {len(samples)}/{len(all_samples)} | resumed {counters['resumed']} | "
        f"pending {len(pending)} | workers {workers} | endpoints {config.endpoints}"
    )
    print(f"Served model roots: {roots}")
    if not pending:
        print("All selected samples already have checkpoints.")
        return

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

    def status_text() -> str:
        return (
            f"ok:{counters['completed']} invalid:{counters['invalid_json']} "
            f"failed:{counters['failed']} "
            f"missing:{counters['skipped_missing_image']}"
        )

    with (
        build_ripo_client_from_config(
            config,
            max_tokens=args.max_tokens,
            temperature=0.0,
        ) as client,
        args.output.open(mode, encoding="utf-8") as output_file,
        progress,
    ):
        progress_task = progress.add_task(
            "LogiCAM-JSON v2",
            total=len(samples),
            completed=counters["resumed"],
            status=status_text(),
        )
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="logicam-json"
        ) as executor:
            futures: dict[Future[dict[str, Any]], Sample] = {
                executor.submit(
                    run_one,
                    client,
                    sample,
                    data_path=args.data,
                    template_dir=args.template_dir,
                    media_root=args.media_root,
                    max_tokens=args.max_tokens,
                ): sample
                for sample in pending
            }
            for future in as_completed(futures):
                record = future.result()
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                counters[str(record["status"])] += 1
                progress.advance(progress_task)
                progress.update(progress_task, status=status_text())

    print(json.dumps(dict(counters), indent=2))


if __name__ == "__main__":
    main()
