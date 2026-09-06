"""Evaluate Reporter claim/citation behavior on a controlled fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ipa.reporter_claims import validate_claims


def evaluate(path: str | Path) -> dict:
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for case in fixture["cases"]:
        claims = validate_claims(case["answer"], case["evidence"])
        actual = claims[0]["support_level"] if claims else "unsupported"
        results.append({"id": case["id"], "expected": case["expected"], "actual": actual, "passed": actual == case["expected"]})
    passed = sum(result["passed"] for result in results)
    return {"benchmark_id": fixture["benchmark_id"], "cases": results, "passed": passed, "total": len(results), "accuracy": passed / len(results) if results else 1.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="tests/fixtures/reporter_v1.json")
    args = parser.parse_args()
    result = evaluate(args.fixture)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
