# Experiments

This page documents representative experiments reproduced using the
packaged **RAFT-UP** implementation.

All experiments were rerun after refactoring the codebase into a
reproducible Python package (`raftup`).

Each experiment corresponds to a concrete notebook that can
be executed end-to-end.

---

## Full slice alignment (Adjacent DLPFC slices) 

![Adjacent slices](_static/results/151508_151509_full_global.png)

**Setting**
- Dataset: DLPFC
- Distance between slices: 10 μm
- Layer-wise accuracy: 
- Geometric preservation rate: 


## Full slice alignment (Far-apart DLPFC slices)

![far-apart slices](_static/results/151509_151510_full_global.png)

**Setting**
- Dataset: DLPFC
- Distance between slices: 300 μm
- Layer-wise accuracy: 
- Geometric preservation rate: 

---

## Full slice alignment (Adjacent MERFISH slices)

![Adjacent slices](_static/results/-0.04_-0.09_full_global.png)

**Setting**
- Dataset: MERFISH
- Distance between slices: 50 μm
- Layer-wise accuracy: 
- Geometric preservation rate: 

## Full slice alignment (Far-apart MERFISH slices)

![Far-apart slices](_static/results/-0.04_-0.19_full_global.png)

**Setting**
- Dataset: MERFISH
- Distance between slices: 150 μm
- Layer-wise accuracy: 
- Geometric preservation rate: 

---

## Overlapping-window alignment (regular window)

![Overlapping regular window alignment](_static/results/raftup_small_window_1.png)

**Setting**
- Dataset: DLPFC
- Window type: overlapping regular window
- Layer-wise accuracy: 
- Geometric preservation rate:

---

## Overlapping-window alignment (irregular window)

![Overlapping irregular window alignment](_static/results/raftup_small_window_2.png)

**Setting**
- Dataset: DLPFC
- Window type: overlapping irregular window
- Layer-wise accuracy: 
- Geometric preservation rate:

---

## Spatiotemporal trajectories across developmental stages

![Spatiotemporal trajectory inference](_static/results/Figure_5_spatiotemporal.png)

**Setting**
- Dataset: Stereo-seq mouse midbrain from E12.5 to E14.5 and from E14.5 to E16.5. Cells are colored
by the expert annotations from original study: RGC, radial glia cell; GlioB, glioblast; NeuB, neu-
roblast

## Spatially preserving analysis of cell-cell communication across slices

![Cell-cell communication](_static/results/Figure_6_ccc.png)

**Setting**
- Dataset: COMMOT is applied independently to each slice to infer cell-cell communication (CCC). CCC inferred on slice A is transferred to slice B using the RAFT-UP spot-to-spot mapping, enabling direct comparison to CCC inferred on
slice B, and vice versa
