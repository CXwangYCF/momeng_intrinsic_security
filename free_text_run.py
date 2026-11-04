#!/usr/bin/env python3
"""Run the official MuSLR baseline or LogiCAM prompt against an OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from muslr_core.ripo_batch import RipoBatchClient
from muslr_core.runtime import CompletionClient, load_samples, metadata_for_samples, run_samples, selected_samples


WORKSPACE = Path(__file__).resolve().parents[2]
load_dotenv(WORKSPACE / ".env")


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def default_template_dir() -> Path:
    return env_path("MUSLR_PROMPT_DIR", WORKSPACE / "prompts")


def default_local_media_root() -> Path:
    output_dir = env_path(
        "MUSLR_OUTPUT_DIR",
        WORKSPACE.parent / "outputs",
    )
    return env_path("MUSLR_LOCAL_MEDIA_ROOT", output_dir / "muslr_media")


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def parse_endpoints(value: str) -> tuple[str, ...]:
    endpoints = tuple(part.strip() for part in value.split(",") if part.strip())
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    return endpoints


def default_image_root(data_path: Path, explicit_image_root: Path | None) -> Path | None:
    if explicit_image_root is not None:
        return explicit_image_root
    if data_path.is_dir() and (data_path / "images").is_dir():
        return data_path / "images"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=env_path("MUSLR_DATA", WORKSPACE / "data"),
        help="MuSLR metadata file or dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["MUSLR_OUTPUT"]).expanduser()
        if "MUSLR_OUTPUT" in os.environ
        else None,
        help="Checkpointed JSONL output path (default: MUSLR_OUTPUT_DIR/muslr_<method>.jsonl).",
    )
    parser.add_argument(
        "--method",
        choices=("baseline", "logicam"),
        default=os.environ.get("MUSLR_METHOD", "baseline"),
    )
    parser.add_argument("--image-root", type=Path, help="Directory containing GitHub image files.")
    parser.add_argument("--template-dir", type=Path, default=default_template_dir())
    parser.add_argument(
        "--media-mode",
        choices=("local", "data-url"),
        default="local",
        help="Use local file:// image paths by default; data-url is available as a fallback.",
    )
    parser.add_argument(
        "--local-media-root",
        type=Path,
        default=default_local_media_root(),
        help="Directory used to materialize images embedded in official prompts.",
    )
    parser.add_argument("--batch-backend", choices=("ripo", "openai"), default="ripo")
    parser.add_argument(
        "--endpoints",
        default=os.environ.get("RIPO_ENDPOINTS", "http://127.0.0.1:22000/v1"),
        help="Comma-separated RIPO endpoints; each endpoint maps to one serving process.",
    )
    parser.add_argument(
        "--ripo-root",
        type=Path,
        default=env_path("RIPO_ROOT", WORKSPACE.parent / "RIPO"),
    )
    parser.add_argument(
        "--ripo-image-url-mode",
        choices=("data", "file"),
        default=os.environ.get("RIPO_IMAGE_URL_MODE", "data"),
        help="How RIPO sends local images to vLLM; data works without --allowed-local-media-path access.",
    )
    parser.add_argument(
        "--endpoint-concurrency",
        type=int,
        default=env_int("RIPO_ENDPOINT_CONCURRENCY", 1),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:22000/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(
            "RIPO_API_KEY",
            os.environ.get("OPENAI_API_KEY", "EMPTY"),
        ),
    )
    parser.add_argument("--model", default=os.environ.get("RIPO_MODEL", "anom-pi"))
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Generation limit (default: method-specific value from .env).",
    )
    parser.add_argument("--timeout", type=float, default=env_float("RIPO_TIMEOUT", 120.0))
    parser.add_argument("--retries", type=int, default=env_int("RIPO_RETRIES", 2))
    parser.add_argument(
        "--workers",
        type=int,
        default=env_int("MUSLR_WORKERS", 1),
        help="Outer batch workers; normally endpoints x endpoint concurrency.",
    )
    parser.add_argument("--ids", help="Comma-separated IDs to run.")
    parser.add_argument("--domains", help="Comma-separated domain filter, case-insensitive.")
    parser.add_argument("--symbols", help="Comma-separated logic symbols, such as pl,nm,fol.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Limit selected samples after filters and sharding.")
    parser.add_argument("--no-resume", action="store_true", help="Do not retain prior JSONL records.")
    parser.add_argument(
        "--retry-statuses",
        help="Comma-separated prior statuses to rerun in place, for example: unparsed,failed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and write metadata only.")
    args = parser.parse_args()

    if args.max_tokens is None:
        fallback_max_tokens = env_int("MUSLR_MAX_TOKENS", 1024)
        method_variable = f"MUSLR_{args.method.upper()}_MAX_TOKENS"
        args.max_tokens = env_int(method_variable, fallback_max_tokens)
    if args.output is None:
        output_dir = env_path("MUSLR_OUTPUT_DIR", WORKSPACE / "outputs")
        args.output = output_dir / f"muslr_{args.method}.jsonl"

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.endpoint_concurrency < 1:
        parser.error("--endpoint-concurrency must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if not args.data.exists():
        parser.error(f"MuSLR data path does not exist: {args.data}")
    retry_statuses = parse_csv(args.retry_statuses)
    allowed_retry_statuses = {"unparsed", "failed", "skipped_missing_image"}
    if retry_statuses and not retry_statuses <= allowed_retry_statuses:
        parser.error(
            "--retry-statuses only supports: " + ",".join(sorted(allowed_retry_statuses))
        )
    if retry_statuses and args.no_resume:
        parser.error("--retry-statuses cannot be combined with --no-resume")
    if not args.template_dir.is_dir():
        parser.error(f"Prompt directory does not exist: {args.template_dir}")
    try:
        endpoints = parse_endpoints(args.endpoints)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_backend == "ripo" and args.media_mode != "local":
        parser.error("RIPO batch backend requires --media-mode local so images stay on local paths.")
    if args.batch_backend == "ripo" and not args.ripo_root.is_dir():
        parser.error(f"RIPO root does not exist: {args.ripo_root}")

    image_root = default_image_root(args.data, args.image_root)
    samples = load_samples(args.data)
    ids = {part.strip() for part in args.ids.split(",") if part.strip()} if args.ids else None
    selected = selected_samples(
        samples,
        ids=ids,
        domains=parse_csv(args.domains),
        symbols=parse_csv(args.symbols),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    if not selected:
        parser.error("No samples selected.")

    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args_dict["api_key"] = "***"
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata_for_samples(selected, args_dict), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected {len(selected)} of {len(samples)} samples. Metadata: {metadata_path}")
    if args.dry_run:
        return

    if args.batch_backend == "ripo":
        client = RipoBatchClient(
            endpoints=endpoints,
            ripo_root=args.ripo_root,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            endpoint_concurrency=args.endpoint_concurrency,
            image_url_mode=args.ripo_image_url_mode,
        )
    else:
        client = CompletionClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    try:
        counters = run_samples(
            samples=selected,
            output_path=args.output,
            completion_client=client,
            method=args.method,
            template_dir=args.template_dir,
            dataset_dir=args.data if args.data.is_dir() else args.data.parent,
            image_root=image_root,
            media_mode=args.media_mode,
            local_media_root=args.local_media_root if args.media_mode == "local" else None,
            workers=args.workers,
            resume=not args.no_resume,
            retry_statuses=retry_statuses,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    print(json.dumps(counters, indent=2))


if __name__ == "__main__":
    main()
