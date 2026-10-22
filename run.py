import argparse
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="HPatches reliability benchmark on CPU")
    parser.add_argument("mode", choices=("demo","reproduce","evaluate"))
    parser.add_argument("--cache", type=Path, default=ROOT/".cache")
    parser.add_argument("--out", type=Path, default=ROOT/"work")
    parser.add_argument("--predictions", type=Path, default=ROOT/"results/predictions.jsonl")
    args = parser.parse_args()
    cfg = json.loads((ROOT/"configs/benchmark.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        from src.evaluate import evaluate
        with args.predictions.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != 116*5*2:
            raise ValueError(f"Expected 1160 method-pair records, found {len(rows)}")
        evaluate(rows,cfg,args.out)
    else:
        from src.assets import fetch_inputs
        from src.experiment import Matchers, benchmark, demo
        source, data = fetch_inputs(args.cache.resolve(), cfg, need_data=args.mode=="reproduce")
        matchers = Matchers(source,cfg["features"])
        if args.mode == "demo":
            demo(matchers,cfg,args.out)
        else:
            from src.evaluate import evaluate
            from src.assets import digest
            start = time.monotonic()
            rows = benchmark(data,matchers,cfg)
            with (args.out/"predictions.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
            evaluate(rows,cfg,args.out)
            manifest = {
                "run_id":"hpatches-cpu-v1",
                "completed_utc":datetime.now(timezone.utc).isoformat(),
                "inference_and_evaluation_seconds":round(time.monotonic()-start,2),
                "device":"CPU, 2 PyTorch threads, 1 OpenCV thread",
                "python":platform.python_version(),
                "system":platform.system(),
                "dependencies":{p:version(p) for p in ("torch","numpy","opencv-python-headless","Pillow","tqdm")},
                "config_sha256":digest(ROOT/"configs/benchmark.json"),
                "model_commit":cfg["xfeat_commit"],"archive_sha256":cfg["data_sha256"],
                "predictions_sha256":digest(args.out/"predictions.jsonl"),
                "summary_sha256":digest(args.out/"summary.csv"),
                "sequences":{s:sorted({r["sequence"] for r in rows if r["split"]==s}) for s in ("dev","test")},
            }
            (args.out/"run.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
