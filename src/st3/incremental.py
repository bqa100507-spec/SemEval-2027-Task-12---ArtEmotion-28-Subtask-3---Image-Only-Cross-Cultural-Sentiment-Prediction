"""Checksum-validated, per-image feature storage with atomic batch commits."""

from contextlib import closing
import hashlib
import json
import sqlite3

import numpy as np


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def valid_vectors(vectors, count, width):
    if (vectors.shape != (count, width) or vectors.dtype != np.float32
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0., atol=2e-6)):
        raise ValueError("Invalid frozen feature shape, dtype or norm")


def cached_batch(connection, digests, width):
    found = {}
    for digest in digests:
        row = connection.execute("SELECT vector, checksum, batch_size FROM features WHERE image = ?", (digest,)).fetchone()
        if row is not None:
            payload, checksum, batch_size = row
            if digest_bytes(payload) != checksum:
                raise ValueError(f"Incremental feature checksum mismatch for image {digest}")
            vector = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
            valid_vectors(vector.reshape(1, -1), 1, width)
            found[digest] = (vector, {"vector_sha256": checksum, "encoded_batch_size": batch_size})
    return found


def collect_features(images, directory, identity, batch_size, load_encoder, *, progress=None):
    """Commit each completed batch. A killed process leaves earlier batches reusable."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    profile = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = digest_bytes(profile.encode("utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identity['encoder']}-{key}.sqlite3"
    digests = list(images)
    width = identity["width"]
    vectors, details = {}, {}
    encode = None
    encoded = 0
    with closing(sqlite3.connect(path, timeout=300)) as connection:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS profile (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT OR IGNORE INTO profile VALUES (1, ?)", (profile,))
            if connection.execute("SELECT value FROM profile WHERE id = 1").fetchone() != (profile,):
                raise ValueError("Incremental feature database profile differs from the requested encoder")
            connection.execute("""CREATE TABLE IF NOT EXISTS features (
                image TEXT PRIMARY KEY, vector BLOB NOT NULL, checksum TEXT NOT NULL, batch_size INTEGER NOT NULL)""")
        for start in range(0, len(digests), batch_size):
            keys = digests[start:start + batch_size]
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                hits = cached_batch(connection, keys, width)
                missing = [digest for digest in keys if digest not in hits]
                if missing:
                    if encode is None:
                        encode = load_encoder()
                    batch = encode([(digest, images[digest]) for digest in missing])
                    valid_vectors(batch, len(missing), width)
                    payloads = [vector.astype("<f4", copy=False).tobytes() for vector in batch]
                    connection.executemany("INSERT INTO features VALUES (?, ?, ?, ?)",
                                           [(d, value, digest_bytes(value), len(missing))
                                            for d, value in zip(missing, payloads)])
                fresh = cached_batch(connection, missing, width)
            for origin, entries in (("incremental", hits), ("encoded", fresh)):
                for digest, (vector, info) in entries.items():
                    vectors[digest] = vector
                    details[digest] = {"origin": origin, **info}
            if missing:
                encoded += len(missing)
                if progress is not None:
                    progress(encoded)
    return vectors, {"database": str(path), "identity": identity, "batch_size": batch_size,
                     "encoded": encoded, "incremental_hits": len(digests) - encoded, "images": details}
