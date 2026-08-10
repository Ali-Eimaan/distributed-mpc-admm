# Media

Every file here is generated, never hand-edited. Regenerate with:

```bash
python -m distributed_mpc_admm.plotting
```

which calls `plotting.make_readme_media(...)` and rewrites all three files.

| File | Produced by | Shows |
| --- | --- | --- |
| `4_agent_formation.gif` | `notebooks/03_formation_control.ipynb` | Four agents converging from scattered initial conditions to a square, communication edges overlaid. |
| `topology_switch.gif` | `notebooks/04_switching_topology.ipynb` | A cycle graph splitting into two disconnected pairs and merging again; the two components visibly drift apart while split. |
| `convergence_curves.png` | `notebooks/05_convergence_analysis.ipynb` | Primal and dual residuals versus ADMM iteration across topologies, with tolerance thresholds. |

<!-- TODO(deepseek §12.9): generate all three. Constraints that matter:
     - GitHub will not autoplay a GIF over ~5 MB in a README. Keep each under that.
     - Use `plotting.apply_style("readme")` so the figures are legible in dark mode; a
       white-background PNG in a dark README reads as a rendering bug.
     - The switching GIF must make the *event* legible: annotate the split and merge
       instants on the frame, not only in the caption. -->
