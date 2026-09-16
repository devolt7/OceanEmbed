"""Fetch real CMEMS sea-surface height anomaly (SSH) for region/window.

Uses the ``copernicusmarine`` Python toolbox against the Copernicus Marine Data Store.
Requires a **free Copernicus Marine account** supplied via ``.env``:

    COPERNICUSMARINE_USERNAME=...
    COPERNICUSMARINE_PASSWORD=...

If those credentials are absent the script **skips gracefully** (writes a manifest with
``status: skipped``) and the rest of the pipeline simply omits the ``ssh`` feature. This
is the documented degraded-mode behaviour — only real data ever enters the pipeline.

``--sample`` writes a deterministic **demo-sample** SSH grid (manifest ``status: sampled``)
so the full 6-feature pipeline can run offline before credentials are configured. The
sample is clearly labelled synthetic and must be replaced with a live download for any
real-world use.

Run:  ``python -m src.ingestion.fetch_copernicus [--config config.bootstrap.yaml] [--sample]``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion.sample_data import write_sample_surface
from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.io import populated, repo_relative, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_copernicus")

OUT_NAME = "cmems_ssh"


def has_credentials() -> bool:
    username = os.environ.get("COPERNICUSMARINE_USERNAME")
    password = os.environ.get("COPERNICUSMARINE_PASSWORD")
    return populated(username) and populated(password)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--sample", action="store_true",
                        help="write a deterministic demo-sample grid instead of downloading")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = resolve_path(cfg, "raw_dir")

    if args.sample:
        path = write_sample_surface(cfg, "ssh")
        log.info("SSH (Copernicus) demo sample written -> %s", path)
        return 0

    tr = cfg["time_range"]
    out_path = raw_dir / f"{OUT_NAME}_{tr['start'][:10]}_{tr['end'][:10]}.nc"

    if not has_credentials():
        write_manifest(
            cfg,
            "copernicus",
            {
                "status": "skipped",
                "reason": "COPERNICUSMARINE_USERNAME / COPERNICUSMARINE_PASSWORD not set in .env",
            },
        )
        log.warning(
            "SSH (Copernicus) skipped: set COPERNICUSMARINE_USERNAME / "
            "COPERNICUSMARINE_PASSWORD in .env to enable. Proceeding without ssh feature."
        )
        return 0

    if out_path.exists():
        log.info("CMEMS file already exists, skipping download: %s", out_path)
    else:
        try:
            import copernicusmarine as cm
        except ImportError as exc:
            log.error("copernicusmarine package not installed (%s); cannot fetch SSH.", exc)
            return 1

        about = cfg["ingestion"]["copernicus"]
        region = cfg["region"]
        log.info("calling Copernicus Marine subset for %s", about["dataset_id"])
        cm.subset(
            dataset_id=about["dataset_id"],
            variables=[about["variable"]],
            minimum_longitude=region["west"],
            maximum_longitude=region["east"],
            minimum_latitude=region["south"],
            maximum_latitude=region["north"],
            start_datetime=f"{tr['start'][:10]}T00:00:00",
            end_datetime=f"{tr['end'][:10]}T23:59:59",
            output_filename=out_path.name,
            output_directory=str(raw_dir),
            force_download=True,
        )

    write_manifest(
        cfg,
        "copernicus",
        {
            "status": "downloaded",
            "source": "Copernicus Marine (CMEMS) L4 SSH / SLA",
            "file": repo_relative(out_path),
            "dataset_id": cfg["ingestion"]["copernicus"]["dataset_id"],
            "variable": cfg["ingestion"]["copernicus"]["variable"],
        },
    )
    log.info("SSH (Copernicus) downloaded -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))