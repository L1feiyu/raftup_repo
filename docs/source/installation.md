# Installation

## Main RAFT-UP environment

Clone the repository and install RAFT-UP in editable mode:

```bash
git clone https://github.com/L1feiyu/raftup_repo.git
cd raftup_repo
conda create -n raftup python=3.10
conda activate raftup
pip install -e .
```
To verify that RAFT-UP is imported from the local repository, run:
```bash
python -c "import raftup; print(raftup.__file__)"
```

## Optional Mayavi environment

For Mayavi-based 3D visualization of RAFT-UP alignment, we recommend using a separate environment to avoid dependency conflicts with the main RAFT-UP environment.

Create the Mayavi environment for RAFT-UP visualization with:

```bash
conda create -n mayavi_env -c conda-forge python=3.11 mayavi pyqt scanpy ipykernel openpyxl "numpy<2.4"
conda activate mayavi_env
```


