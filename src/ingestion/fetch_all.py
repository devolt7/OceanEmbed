"""Run all four ingestion scripts in sequence.

Convenience entry point for the notebook / CI flow:
    python -m src.ingestion.fetch_all [--config config.bootstrap.yaml]

Individual scripts remain independently runnable:
    python -m src.ingestion.fetch_argo
    python -m src.ingestion.fetch_oisst
    python -m src.ingestion.fetch_copernicus
    python -m src.ingestion.fetch_smap
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.ingestion import fetch_argo, fetch_copernicus, fetch_oisst, fetch_smap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OceanEmbed ingestion driver")
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument(
        "--only",
        default=[],
        nargs="*",
        choices=["argo", "oisst", "copernicus", "smap"],
        help="run only these sources (default: all)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="write demo-sample CMEMS/SMAP grids instead of downloading (offline mode)",
    )
    args = parser.parse_args(argv)

    cli = ["--config", args.config] if args.config else []
    rc = 0
    for name, mod in (
        ("argo", fetch_argo),
        ("oisst", fetch_oisst),
        ("copernicus", fetch_copernicus),
        ("smap", fetch_smap),
    ):
        if args.only and name not in args.only:
            continue
        logging.getLogger(name).info("=" * 70)
        step = list(cli)
        if args.sample and name in ("copernicus", "smap"):
            step.append("--sample")
        rc |= mod.main(step)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))