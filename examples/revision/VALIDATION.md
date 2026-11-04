# Validation of 0.2.0.dev0

- Wheel built and installed into an isolated target directory without modifying the existing environment.
- Imports resolved to that installed wheel, not the source checkout.
- Four functional checks passed: known rigid transform, spatial MNN distance filtering and purity, exact support-pair audit including shared endpoints, and rejection of mismatched cost-cell ordering.
- Actual manual FsGW solver notebook passed on a four-cell synthetic pair.
- All 13 portable notebooks compiled as Python; real-data notebooks require the documented prepared inputs.
- Five existing automatic/manual/experimental source files match their pre-revision snapshot byte-for-byte; hashes are recorded in algorithm_source_sha256.json.
- Existing full datasets, model training and competing-method runs were not rerun for packaging.

Environment qualification: local tests used Python 3.10 on macOS arm64 with an existing PyTorch 2.1.0 installation. PyTorch imports and solver execution pass, but `pip check` reports that this historical Torch distribution is unsupported on the current platform. A clean Conda installation and a Linux/CUDA run are not yet certified; environment.yml is a pinned environment specification, not proof of a new successful installation. See environments/observed-verification-environment.json for the observed package versions.

Re-run checks with `python scripts/validate_revision.py` after installing the revision package.
