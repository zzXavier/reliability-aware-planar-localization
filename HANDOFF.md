# P03 handoff

- Project: P03 / reliability-aware-planar-localization.
- Working upstream: original XFeat pretrained checkpoint, pinned in UPSTREAM.md. Upstream minimal example completed on CPU including the 100-iteration stress loop. On this Windows CUDA build we forced CPU availability for the standalone author example; project commands set CUDA_VISIBLE_DEVICES before importing torch.
- Last verified: synthetic known-homography, insufficient-match and calibration-denominator tests; ORB and XFeat generated-poster demo.
- Formal run: CPU evaluation on official HPatches ZIP in progress. Retain private status until formal outputs and a second clean checkout are verified.
- Next: run `python run.py reproduce --out work/full`, then compare `python run.py evaluate --predictions work/full/predictions.jsonl --out work/rebuilt` to its generated tables.
