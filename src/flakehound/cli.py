"""flakehound CLI.

Commands:
    flakehound scan [PATHS...]   Static flaky-pattern scan (no tests executed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flakehound import __version__
from flakehound.config import load_config
from flakehound.scanner import exit_code, scan_paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flakehound", description=__doc__)
    p.add_argument("--version", action="version", version=f"flakehound {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="statically detect flaky-prone patterns in test code")
    scan.add_argument("paths", nargs="*", default=["."], help="files or directories (default: .)")
    scan.add_argument("--format", choices=["text", "json"], default="text")
    scan.add_argument("--fail-on", choices=["high", "medium", "advisory", "never"], default=None,
                      help="minimum confidence tier that causes exit code 1 (default: config or 'high')")
    scan.add_argument("--no-ml-rules", action="store_true", help="disable the ML rule pack (M*)")
    scan.add_argument("--disable", action="append", default=[], metavar="RULE_ID")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "scan":
        return _cmd_scan(args)
    return 2  # pragma: no cover


def _cmd_scan(args: argparse.Namespace) -> int:
    config = load_config()
    if args.disable:
        config.disabled = config.disabled | frozenset(args.disable)
    if args.no_ml_rules:
        config.include_ml_rules = False
    fail_on = args.fail_on or config.fail_on

    result = scan_paths([Path(p) for p in args.paths], config)

    out = sys.stdout
    if args.format == "json":
        payload = {
            "files_scanned": result.files_scanned,
            "parse_errors": result.parse_errors,
            "findings": [f.__dict__ | {"confidence": f.confidence.value} for f in result.findings],
        }
        out.write(json.dumps(payload, indent=2, default=str) + "\n")
    else:
        for f in result.findings:
            out.write(f.format_text() + "\n")
        for err in result.parse_errors:
            out.write(f"parse error: {err}\n")
        out.write(
            f"\nflakehound: {len(result.findings)} finding(s) in "
            f"{result.files_scanned} test file(s)\n"
        )
    return exit_code(result, fail_on)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
