# Quickstart

RAFT-UP is currently under active development.

This page provides a demonstration of the full RAFT-UP pipeline using
a real experimental notebook.

---

## Minimal demo

- Notebook: [examples/demo_minimal.ipynb](https://github.com/L1feiyu/raftup_repo/blob/main/examples/demo_minimal.ipynb)
- Script: `python scripts/run_demo.py`

This demo generates a toy dataset and writes outputs to `./outputs/`.

---

## Environment requirement

⚠️ **Important**

The demo notebook was run under a custom research environment:

- Python environment: `fsgw_test_paste2`
- Includes dependencies such as:
  - POT / optimal transport libraries
  - PASTE / PASTE2
  - Scanpy
  - PyTorch
  - Other research dependencies

At this stage, we recommend running the notebook inside the same environment
used in the experiments.

A minimal, installable dependency list and a packaged version of RAFT-UP
will be provided in a future release.
