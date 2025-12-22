# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RAFT-UP Minimal Demo
#
# This notebook is a minimal smoke test for the docs + jupytext + script workflow.
#


import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# repo root is two levels up from examples/
REPO_ROOT = Path.cwd()
if (REPO_ROOT / 'examples').exists():
    # running from repo root
    pass
else:
    # running from examples/
    REPO_ROOT = Path(__file__).resolve().parents[1]

out_dir = REPO_ROOT / 'outputs'
out_dir.mkdir(exist_ok=True)

# toy data
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 100)
y = np.sin(2*np.pi*x) + 0.1*rng.standard_normal(size=x.shape[0])

df = pd.DataFrame({'x': x, 'y': y})
csv_path = out_dir / 'demo_minimal.csv'
df.to_csv(csv_path, index=False)
print('Saved:', csv_path)

plt.figure()
plt.plot(x, y)
fig_path = out_dir / 'demo_minimal.png'
plt.savefig(fig_path, dpi=150)
print('Saved:', fig_path)

