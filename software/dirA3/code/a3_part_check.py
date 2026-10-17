"""Validate one audit part file {"verdicts": [{k,id,verdict,confidence,note}, ...]} and print the number of valid verdicts."""
import json
import sys

d = json.load(open(sys.argv[1]))
ok = [v for v in d.get("verdicts", []) if isinstance(v, dict) and v.get("id") and v.get("verdict") in ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS")]
print(len(ok))
