"""Probe hf-mirror for a fast hopper_medium-v2.hdf5 copy.

Prints candidate file paths from the takumi/d4rl dataset repo listing and
HEAD status codes for likely download URLs.  Read-only probing helper.
"""

import json
import sys
import urllib.request

API = "https://hf-mirror.com/api/datasets/takumi/d4rl"


def head(url: str) -> str:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            size = response.headers.get("Content-Length", "?")
            return f"{response.status} len={size}"
    except Exception as exc:  # noqa: BLE001 - probe helper
        return f"ERR {exc}"


def main() -> int:
    try:
        with urllib.request.urlopen(API, timeout=40) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print("API error:", exc)
        return 1
    siblings = [entry["rfilename"] for entry in payload.get("siblings", [])]
    print("total files:", len(siblings))
    hopper = [name for name in siblings if "hopper" in name.lower()]
    print("hopper files:")
    for name in hopper[:40]:
        print("  ", name)
    for candidate in (
        "hopper/medium_v2.hdf5",
        "hopper-medium-v2.hdf5",
        "hopper_medium-v2.hdf5",
    ):
        if candidate in siblings or True:
            url = f"https://hf-mirror.com/datasets/takumi/d4rl/resolve/main/{candidate}"
            print(f"HEAD {candidate}: {head(url)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
