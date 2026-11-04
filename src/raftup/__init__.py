from .version import __version__

__all__ = [
    "__version__",
    "_downsample",
    "_load_data",
    "_metrics_two_gpr",
    "_fsgw_utils",
    "_recoverfull_new_new_knn",
    "_gene_cost_somde",
]

# Lazy public entry points keep lightweight imports available for figure-only use.
def run_automated_alignment(*args, **kwargs):
    """Align with calibrated DLPFC/MERFISH selection or explicit experimental test mode."""
    from ._cutoff_selection import run_automated_alignment as run
    return run(*args, **kwargs)


def run_manual_alignment(*args, **kwargs):
    """Align with explicit cutoffs; see raftup._manual_alignment for parameter details."""
    from ._manual_alignment import run_manual_alignment as run
    return run(*args, **kwargs)


__all__ += ["run_automated_alignment", "run_manual_alignment"]
