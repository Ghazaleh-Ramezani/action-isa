# Separating Structural from Parametric Decisions in Sub-Billion Tool Use

Code and per-task records for the NeurIPS 2026 SLM-Agents workshop paper.

## What the paper points at

| file | regenerates |
|---|---|
| `analysis.py` | every statistic in the paper from the released per-task records |
| `block_inference.py` | the block-aware analysis (Appendix on block-aware inference) |
| `bfcl_recipe_harness.py` | the public-benchmark subset and bundles from the public files |
| `bfcl_experiment.py` | the gold, random and model arms of the transfer experiment |
| `verify_prereg.py` | checks the released prediction record against its digest |

## Prediction records

Both files cited in the paper are included with the digests it reports:

    predictions.json   sha256 b0939b91...   (ladder seeds + second family)
    prereg_r3.json     sha256 7655830a...   (robustness, sweep, factorial, ladder)

`verify_prereg.py prereg_r3.json` checks the values. Note that the harness
regenerates this file with a fresh timestamp when it is absent, so a
from-scratch run produces a different digest with identical predicted values;
only the file released here carries the digest in the paper.

## Reproducing

    pip install numpy
    python3 analysis.py            # ends with: All checks reproduce the paper.
    python3 block_inference.py results/paper_seed0_per_task.csv \
            --block-col block --arm-a L0 --arm-b L1
    python3 verify_prereg.py prereg_r3.json

None of these needs a GPU. `analysis.py` reads the per-task records in
`results/` (`r0_results.json`, `r2_results.json`, `r3_results_E.json`,
`partA.json`--`partD.json`) and checks every statistic against the value
printed in the paper. Re-running the model arms needs a GPU; the ladder and
second-family records are under `results/extra/`.
