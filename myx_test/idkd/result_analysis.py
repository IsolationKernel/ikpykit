from pathlib import Path
import json
from collections import defaultdict

result_file = Path(__file__).resolve().parent / "result.result"

output = defaultdict(int)
with result_file.open("r") as f:
    for line in f:
        result = json.loads(line)
        if result["dataset"] not in output:
            output[result["dataset"]] = result["score"]
        else:
            output[result["dataset"]] = max(output.get(result["dataset"]), result["score"])

with (result_file.parent / "README.md").open("a+") as f:
    for dataset, score in output.items():
        f.write(f"## {dataset}\n")
        f.write(f"{score}\n")
        f.write("\n")
