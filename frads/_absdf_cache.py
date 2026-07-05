"""
Hash-based disk cache for aBSDF Cds-matrices.

Pre-computed SunMatrix arrays are large (sensor_count * sun_positions * 3 floats)
and expensive to regenerate (~30s rcontrib per pane*state). We persist them in
``simulation/cache/cds/<sha256>.npz`` keyed by the ``key_parts`` the caller
passes in. This module hashes *exactly* those parts (plus an internal cache
version) and does no invalidation of its own -- correctness depends entirely on
the caller including every result-relevant input in ``key_parts``. The current
caller (``FivePhaseMethod._generate_absdf_sun_matrices``, ``absdf-v2`` key)
covers: role, sender name, pane index, state key, sun basis, the sun_culling
flag, a WEA hash, the sender content bytes (sensor coords/directions or view
spec) and the compiled per-pane octree bytes (scene + window geometry +
blacked-octree modifier + bound BSDF XML path) plus the raw BSDF XML bytes.
Concurrent access is protected by a per-key filelock so that two parallel sims
do not both rebuild the same matrix.

Usage::

    from frads._absdf_cache import cache_get
    arr = cache_get(
        key_parts=[xml_bytes, window_geom_bytes, b"r6", b"r1",
                   sensor_positions_npy_bytes],
        compute_fn=lambda: expensive_sun_matrix_generate(...),
        cache_dir=Path("simulation/cache/cds"),
    )

The compute_fn is called only on cache miss; its return value MUST be a
``numpy.ndarray`` or sequence of ``numpy.ndarray`` objects.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from filelock import FileLock

logger = logging.getLogger("frads._absdf_cache")

_FRADS_CACHE_VERSION = "v1"


def _hash_key(key_parts: Iterable[bytes]) -> str:
    """SHA256 over all key parts (length-prefixed) plus the cache version.

    Each part is hashed as its 8-byte big-endian length followed by its bytes.
    The length prefix makes the encoding injective: two part lists cannot map
    to the same digest by shifting a separator across a part boundary (which a
    bare ``\\x00`` separator allowed when a part itself contained ``\\x00``).
    """
    h = hashlib.sha256()
    h.update(_FRADS_CACHE_VERSION.encode())
    for part in key_parts:
        if not isinstance(part, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"_absdf_cache key parts must be bytes-like, got {type(part).__name__}"
            )
        b = bytes(part)
        h.update(len(b).to_bytes(8, "big"))  # length prefix -> injective
        h.update(b)
    return h.hexdigest()


# Marker array added to a ``.npz`` when the cached result is a
# ``dict[str, np.ndarray]``, so a single-key dict (e.g. ``{"arr": x}``) is not
# misread as a bare single-array entry and vice versa.
_DICT_MARKER = "__absdf_is_dict__"


def _load_cached(npz_path: Path) -> np.ndarray | dict[str, np.ndarray]:
    """Read a cache entry, preserving the stored type (ndarray vs dict).

    A dict entry carries the ``_DICT_MARKER`` key (stripped on return); a bare
    single array is stored under the sole key ``"arr"``. The ``NpzFile`` handle
    is closed via the context manager after the arrays are materialised.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        keys = list(data.files)
        if _DICT_MARKER in keys:
            return {k: data[k] for k in keys if k != _DICT_MARKER}
        if keys == ["arr"]:
            return data["arr"]
        return {k: data[k] for k in keys}


def cache_get(
    key_parts: Iterable[bytes],
    compute_fn: Callable[[], np.ndarray | dict[str, np.ndarray]],
    cache_dir: Path,
) -> np.ndarray | dict[str, np.ndarray]:
    """Return cached array, recomputing once if missing.

    Args:
        key_parts: Iterable of bytes-like inputs that uniquely identify the
            result. Order matters. Include EVERY input the compute_fn depends on.
        compute_fn: Zero-argument callable returning either an ``np.ndarray``
            or a ``dict[str, np.ndarray]`` (saved as multiple arrays in the
            ``.npz`` archive under the dict keys).
        cache_dir: Directory for ``<hash>.npz`` files. Created if missing.

    Returns:
        The cached or freshly computed array(s).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _hash_key(key_parts)
    npz_path = cache_dir / f"{key}.npz"
    lock_path = cache_dir / f"{key}.lock"

    if npz_path.exists():
        return _load_cached(npz_path)

    with FileLock(str(lock_path)):
        if npz_path.exists():  # filled by another process while we waited
            return _load_cached(npz_path)

        logger.info("cache miss %s — recomputing", key[:12])
        result = compute_fn()

        # Atomic write: temp-file in same dir, then rename.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=cache_dir, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            if isinstance(result, np.ndarray):
                np.savez(tmp_path, arr=result)
            elif isinstance(result, dict):
                # Tag with a marker so a single-key dict (e.g. {"arr": x}) is
                # not read back as a bare ndarray.
                np.savez(tmp_path, **result, **{_DICT_MARKER: np.array(True)})
            else:
                raise TypeError(
                    "compute_fn must return np.ndarray or dict[str, np.ndarray], "
                    f"got {type(result).__name__}"
                )
            os.replace(tmp_path, npz_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return result
