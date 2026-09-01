"""Command line entrypoint: bootstrap, plan, apply, status, outputs, destroy.

Usage:
    python -m infra.aws.cli bootstrap --config infra/aws/config/beta.json
    python -m infra.aws.cli plan      --config infra/aws/config/beta.json
    python -m infra.aws.cli apply     --config infra/aws/config/beta.json --image-tag <sha> --yes
    python -m infra.aws.cli status    --config infra/aws/config/beta.json --deep
    python -m infra.aws.cli outputs   --config infra/aws/config/beta.json --out infra-outputs.json
    python -m infra.aws.cli destroy   --config infra/aws/config/beta.json --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import orchestrator
from .config import EnvironmentConfig, load_environment_config
from .session import SessionRequest, build_session
from .status import run_status_checks


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="infra.aws.cli")
    parser.add_argument("command", choices=["bootstrap", "plan", "apply", "status", "outputs", "destroy"])
    parser.add_argument("--config", required=True, help="Path to the environment JSON config")
    parser.add_argument("--account-id", help="Override account_id from the config file")
    parser.add_argument("--region", help="Override region from the config file")
    parser.add_argument(
        "--environment", help="Override environment; this provisioner is beta-only"
    )
    parser.add_argument("--profile", help="AWS CLI profile to use")
    parser.add_argument("--assume-role-arn", help="IAM role to assume instead of using --profile directly")
    parser.add_argument("--image-tag", help="Commit-SHA image tag; required for apply")
    parser.add_argument("--deep", action="store_true", help="Run the full status check suite")
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation prompt")
    parser.add_argument("--out", help="For 'outputs': also write the result JSON to this local file path")
    return parser.parse_args(argv)


def _confirm(command: str, config: EnvironmentConfig, *, skip_prompt: bool) -> None:
    if skip_prompt:
        return
    answer = input(
        f"About to run {command!r} against account={config.account_id} "
        f"region={config.region} environment={config.environment}. Type 'yes' to continue: "
    )
    if answer.strip().lower() != "yes":
        print("Aborted.")
        sys.exit(1)


def _load_config(args: argparse.Namespace) -> EnvironmentConfig:
    overrides: dict[str, Any] = {}
    if args.account_id:
        overrides["account_id"] = args.account_id
    if args.region:
        overrides["region"] = args.region
    if args.environment:
        overrides["environment"] = args.environment
    return load_environment_config(args.config, overrides=overrides)


def _build_context(config: EnvironmentConfig, args: argparse.Namespace) -> orchestrator.OrchestratorContext:
    session = build_session(
        SessionRequest(
            region=config.region,
            expected_account_id=config.account_id,
            profile_name=args.profile,
            assume_role_arn=args.assume_role_arn,
            session_name=f"{config.resource_prefix}-provisioner",
        )
    )
    return orchestrator.OrchestratorContext(
        config=config,
        session=session,
        image_tag=args.image_tag or "untagged",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    config = _load_config(args)

    if args.command == "bootstrap":
        print("bootstrap: no S3 state backend; apply prints the inventory manifest")
        return 0

    ctx = _build_context(config, args)

    if args.command == "plan":
        print(json.dumps(orchestrator.plan(ctx), indent=2))
        return 0

    if args.command == "status":
        report = run_status_checks(ctx, deep=args.deep)
        print(json.dumps(report, indent=2))
        return 0 if report["overall"] == "PASS" else 1

    if args.command == "outputs":
        print(
            "outputs: inventory is not stored in S3; use the manifest printed by apply",
            file=sys.stderr,
        )
        return 2

    if args.command == "apply":
        if not args.image_tag:
            print("apply requires --image-tag <git-commit-sha>", file=sys.stderr)
            return 2
        _confirm("apply", config, skip_prompt=args.yes)
        manifest = orchestrator.apply(ctx)
        print(json.dumps(manifest.to_dict(), indent=2))
        print("apply: complete")
        return 0

    if args.command == "destroy":
        _confirm("destroy", config, skip_prompt=args.yes)
        orchestrator.destroy(ctx)
        print("destroy: no persisted inventory; nothing deleted")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
