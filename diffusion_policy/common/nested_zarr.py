"""Read-only helpers for ZIP-backed Zarr datasets with an optional prefix."""

from __future__ import annotations

import hashlib
import fcntl
import logging
import os
import pathlib
import shutil
import tempfile
import zipfile
from dataclasses import dataclass

import zarr


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NestedZarrInfo:
    path: pathlib.Path
    prefix: str


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def detect_zarr_prefix(path: str | pathlib.Path) -> NestedZarrInfo:
    """Find a root ``.zgroup`` or one unambiguous top-level Zarr group.

    This examines ZIP directory metadata only and never extracts or mutates the
    archive. A multi-root archive is rejected rather than guessed.
    """

    resolved = pathlib.Path(path).expanduser().resolve()
    with zipfile.ZipFile(resolved, mode="r") as archive:
        names = set(archive.namelist())

    if ".zgroup" in names:
        prefix = ""
    else:
        candidates = sorted(
            name[: -len("/.zgroup")].rstrip("/")
            for name in names
            if name.endswith("/.zgroup") and "/" not in name[: -len("/.zgroup")]
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one top-level Zarr prefix in {resolved}, found "
                f"{candidates or 'none'}"
            )
        prefix = candidates[0]

    logger.info("Detected read-only Zarr prefix %r in %s", prefix, resolved)
    return NestedZarrInfo(path=resolved, prefix=prefix)


def open_nested_zip_group(
    path: str | pathlib.Path,
    *,
    prefix: str | None = None,
    extraction_cache_dir: str | pathlib.Path | None = None,
):
    """Return ``(store, Group, detected_prefix)`` opened read-only.

    The caller owns the returned store and must close it. Each DataLoader
    worker must call this function independently; ``ZipStore`` is intentionally
    not cached here or shared across processes.
    """

    info = detect_zarr_prefix(path) if prefix is None else NestedZarrInfo(
        path=pathlib.Path(path).expanduser().resolve(), prefix=str(prefix).strip("/")
    )
    store = zarr.ZipStore(str(info.path), mode="r")
    try:
        group = zarr.open_group(
            store=store,
            path=info.prefix,
            mode="r",
        )
    except (TypeError, NotImplementedError) as exc:
        store.close()
        logger.warning(
            "This Zarr version cannot open a nested ZipStore group (%s); "
            "using the SHA-256 read-only extraction cache",
            exc,
        )
        return _open_extraction_cache(
            info,
            extraction_cache_dir=extraction_cache_dir,
        )
    except Exception:
        # Codec/metadata/corruption errors must remain visible. Extraction is
        # only a compatibility fallback for an unsupported nested-group API.
        store.close()
        raise
    return store, group, info.prefix


def _open_extraction_cache(
    info: NestedZarrInfo,
    *,
    extraction_cache_dir: str | pathlib.Path | None,
):
    cache_root = pathlib.Path(
        extraction_cache_dir
        if extraction_cache_dir is not None
        else tempfile.gettempdir()
    ).expanduser().resolve() / "indy_umi_nested_zarr_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(info.path)
    cache_entry = cache_root / digest
    complete_marker = cache_entry / ".complete"
    lock_path = cache_root / f"{digest}.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if not complete_marker.is_file():
            build_dir = pathlib.Path(
                tempfile.mkdtemp(prefix=f"{digest}.tmp-", dir=cache_root)
            )
            try:
                prefix_with_slash = f"{info.prefix}/" if info.prefix else ""
                with zipfile.ZipFile(info.path, mode="r") as archive:
                    for member in archive.infolist():
                        if member.is_dir() or not member.filename.startswith(
                            prefix_with_slash
                        ):
                            continue
                        relative = pathlib.PurePosixPath(member.filename)
                        if relative.is_absolute() or ".." in relative.parts:
                            raise ValueError(
                                f"unsafe ZIP member path {member.filename!r}"
                            )
                        destination = build_dir.joinpath(*relative.parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member, "r") as source, destination.open(
                            "wb"
                        ) as target:
                            shutil.copyfileobj(source, target)
                (build_dir / ".complete").write_text(
                    f"source_sha256={digest}\nsource={info.path}\n"
                )
                if cache_entry.exists():
                    shutil.rmtree(cache_entry)
                os.replace(build_dir, cache_entry)
            finally:
                if build_dir.exists():
                    shutil.rmtree(build_dir)

    directory_store = zarr.DirectoryStore(str(cache_entry))
    try:
        group = zarr.open_group(
            store=directory_store,
            path=info.prefix,
            mode="r",
        )
    except Exception:
        directory_store.close()
        raise
    return directory_store, group, info.prefix
