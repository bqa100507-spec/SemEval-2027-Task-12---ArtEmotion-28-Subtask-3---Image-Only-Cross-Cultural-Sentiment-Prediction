#!/usr/bin/env python3
"""Generate Hugging Face snapshot pins by code (pre-upload checklist item 2): never type a hash by hand.

Resolves a revision to its commit SHA and lists every requested file from the tree API at that commit.
LFS files get ``weights_sha256`` (the LFS SHA-256); regular files get ``metadata_git_blobs`` (git blob
SHA-1). The license comes from the model card data at the same revision. Optionally recomputes each
metadata blob from a fresh download as an independent check.
"""

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request

API = "https://huggingface.co/api/models"


def fetch_json(url, opener=urllib.request.urlopen):
    with opener(url, timeout=60) as response:
        return json.load(response)


def git_blob(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False).hexdigest()


def pins(model, revision, weights, metadata, opener=urllib.request.urlopen, verify_download=False):
    info = fetch_json(f"{API}/{model}/revision/{revision}", opener)
    commit = info["sha"]
    tree = fetch_json(f"{API}/{model}/tree/{commit}?recursive=true", opener)
    files = {entry["path"]: entry for entry in tree if entry.get("type") == "file"}
    result = {"model": model, "revision": commit, "license": (info.get("cardData") or {}).get("license"),
              "weights_sha256": {}, "metadata_git_blobs": {}, "sizes": {}}
    for pattern, target, lfs in ((weights, "weights_sha256", True), (metadata, "metadata_git_blobs", False)):
        for glob in pattern:
            matched = sorted(name for name in files if fnmatch.fnmatch(name, glob))
            if not matched:
                raise ValueError(f"No file matches {glob} in {model}@{commit}")
            for name in matched:
                entry = files[name]
                if lfs:
                    if not entry.get("lfs"):
                        raise ValueError(f"{name} is not an LFS file; pin it as metadata")
                    result[target][name] = entry["lfs"]["oid"]
                else:
                    result[target][name] = entry["oid"]
                result["sizes"][name] = entry["size"]
    if verify_download:
        for name, expected in result["metadata_git_blobs"].items():
            with opener(f"https://huggingface.co/{model}/resolve/{commit}/{name}", timeout=60) as response:
                if git_blob(response.read()) != expected:
                    raise ValueError(f"Downloaded {name} does not match its tree oid")
        result["metadata_verified_by_download"] = True
    for name, value in result["weights_sha256"].items():
        if len(value) != 64:
            raise ValueError(f"Malformed SHA-256 for {name}")
    for name, value in result["metadata_git_blobs"].items():
        if len(value) != 40:
            raise ValueError(f"Malformed git blob for {name}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--weights", nargs="+", required=True, help="Glob patterns of LFS weight files")
    parser.add_argument("--metadata", nargs="+", required=True, help="Glob patterns of small metadata files")
    parser.add_argument("--verify-download", action="store_true")
    args = parser.parse_args(argv)
    json.dump(pins(args.model, args.revision, args.weights, args.metadata, verify_download=args.verify_download),
              sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
