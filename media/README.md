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

Constraints honoured by the generator:

- GitHub will not autoplay a GIF over ~5 MB in a README, so each GIF is capped at ~7
  inches with `dpi<=110` and frame subsampling (both committed GIFs are under 0.5 MB).
- Figures use `plotting.apply_style("readme")` — transparent background and larger fonts —
  so they stay legible on GitHub's dark mode.
- The switching GIF annotates the `SPLIT`/`MERGE` instants directly on the frame
  (`plotting.animate_formation(..., events=plotting._switch_events(log))`), not only in
  the caption.
