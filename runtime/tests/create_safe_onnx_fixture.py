#!/usr/bin/env python3
"""Write ActionGo's source-owned constant-output ONNX test fixture."""

import argparse
import base64
from pathlib import Path


MODEL = "CAgSCEFjdGlvbkdvOv0BCrEBEgdhY3Rpb25zIghDb25zdGFudCqbAQoFdmFsdWUqjgEIAQgdEAEidAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQhBjb25zdGFudF9hY3Rpb25zoAEEEhVhY3Rpb25nb19zYWZlX2ZpeHR1cmVaFQoDb2JzEg4KDAgBEggKAggBCgIIYmIZCgdhY3Rpb25zEg4KDAgBEggKAggBCgIIHUIECgAQDQ=="


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(MODEL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
