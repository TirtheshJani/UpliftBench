"""Download the Criteo Uplift v2 CSV.gz with mirror fallback and SHA-256 logging."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Annotated

import requests
import typer
from tqdm import tqdm

from upliftbench.config import CRITEO_CSV, CRITEO_URL_MIRROR, CRITEO_URL_PRIMARY, RAW_DIR

app = typer.Typer(add_completion=False, no_args_is_help=False)

CHUNK = 1 << 20


def _stream_download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code != 200:
                typer.echo(f"  {url} -> HTTP {r.status_code}")
                return False
            total = int(r.headers.get("Content-Length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with (
                open(tmp, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))
            tmp.rename(dest)
        return True
    except (requests.RequestException, OSError) as exc:
        typer.echo(f"  {url} -> {exc}")
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@app.command()
def main(
    dest: Annotated[Path, typer.Option(help="Output path for the CSV.gz.")] = CRITEO_CSV,
    dry_run: Annotated[bool, typer.Option(help="Print URLs without downloading.")] = False,
) -> None:
    """Download Criteo Uplift v2 to `dest`. Tries primary URL, falls back to HF mirror."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        typer.echo(f"Already present: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        typer.echo(f"SHA-256: {_sha256(dest)}")
        return

    if dry_run:
        typer.echo(f"Would try primary: {CRITEO_URL_PRIMARY}")
        typer.echo(f"Would fallback to: {CRITEO_URL_MIRROR}")
        typer.echo(f"Would save to:     {dest}")
        return

    typer.echo(f"Primary: {CRITEO_URL_PRIMARY}")
    if _stream_download(CRITEO_URL_PRIMARY, dest):
        typer.echo(f"Downloaded from primary. SHA-256: {_sha256(dest)}")
        return

    typer.echo(f"Mirror:  {CRITEO_URL_MIRROR}")
    if _stream_download(CRITEO_URL_MIRROR, dest):
        typer.echo(f"Downloaded from mirror. SHA-256: {_sha256(dest)}")
        return

    typer.echo("Both URLs failed.", err=True)
    sys.exit(1)


if __name__ == "__main__":
    app()
