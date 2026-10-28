# P03 handoff

## Latest verified state · 2026-11-03

Public release. [Personal portfolio](https://zzxavier.github.io/projects/) and [public README](https://github.com/zzXavier/reliability-aware-planar-localization) were verified without authentication.

Verified experiment revision: `4fa0c7e2079420a1a38c09fc2289b1fd280a8eed`. Independent CPU environment and clone: 116 sequences, 1,160 method predictions and 18 summary rows reproduced. Config, JSONL and summary matched the manifest byte hashes.

Next local command: `python run.py demo`.

## Earlier implementation notes


- Project: P03 / reliability-aware-planar-localization.
- Working upstream: original XFeat pretrained checkpoint, pinned in UPSTREAM.md. Upstream minimal example completed on CPU including the 100-iteration stress loop. On this Windows CUDA build we forced CPU availability for the standalone author example; project commands set CUDA_VISIBLE_DEVICES before importing torch.
- `hpatches-cpu-v1` finished on a Core i7-13650HX in a fresh Python 3.11.9 environment, official torch 2.4.1+cpu. Full inference/evaluation took 70.88 seconds after verifying inputs. `results/run.json` records the 39 development and 77 test sequences.
- Three unit tests passed (geometry, calibration/failed-estimate denominators, interrupted HTTP transfer). The generated-poster demo and all 1,160 method/pair observations completed successfully. Rebuilding `results/summary.csv` from those observations produced SHA-256 `085d016bc340922fca8760af9e6329c0755041e4176830c4bc890008468e1fe5`. The prediction file SHA-256 is `c37efd976cc6c84f0eb088522449b571f82598bfdc2f4d84d179f97216e22c01`.
- Release check: clone the private repository into a new directory, run `python run.py reproduce --out work/full` and `python run.py evaluate --predictions work/full/predictions.jsonl --out work/rebuilt`, then verify both result tables against the published observations.
