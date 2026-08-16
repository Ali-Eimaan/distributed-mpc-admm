# Media

Every file here is generated, never hand-edited. Each asset is written by the notebook
named in the table below; re-executing that notebook rewrites it in place:

```bash
cd python && jupyter nbconvert --to notebook --execute --inplace notebooks/03_formation_control.ipynb
```

`plotting.make_readme_media(log_4agent, log_switching, out_dir)` produces the original three
from two already-computed `SimulationLog`s, which is what a caller holding the logs should
use; it is a library function and takes no command line.

| File | Produced by | Shows |
| --- | --- | --- |
| `4_agent_formation.gif` | `notebooks/03_formation_control.ipynb` | Four agents converging from scattered initial conditions to a square, communication edges overlaid. |
| `8_agent_grid.gif` | `notebooks/03_formation_control.ipynb` | Eight agents converging to a 2×4 lattice on the 4-neighbour graph. At four agents the formation looks like a square whatever the topology is; at eight the graph structure is visible in the motion. |
| `topology_switch.gif` | `notebooks/04_switching_topology.ipynb` | A cycle splitting into two disconnected pairs at `t = 0.6 s` and merging again at `t = 3.0 s`, both instants stamped on the frame. While split, each pair can satisfy only its own edge, so the square collapses — formation error reaches ~0.96 m at the merge instant — and the merge pulls the components back into one formation. |
| `8_agent_topology_switch.gif` | `notebooks/04_switching_topology.ipynb` | The 2×4 lattice losing its two middle horizontal edges at `t = 2.0 s`, leaving two 4-cycles (`{0,1,4,5}` and `{2,3,6,7}`), and regaining them at `t = 4.5 s`. This is where the components genuinely **drift apart** — their centroids separate from 0.97 m to 2.92 m while disconnected — because nothing constrains one 2×2 block relative to the other. |
| `convergence_curves.png` | `notebooks/05_convergence_analysis.ipynb` | Primal and dual residuals versus ADMM iteration, and the `rho` sweep with its empirical optimum. |

Constraints honoured by the generator:

- GitHub will not autoplay a GIF over ~5 MB in a README, so each GIF is capped at ~7 inches
  with `dpi<=110` and frame subsampling. Committed sizes: 444 KB, 508 KB, 743 KB and 1.0 MB
  (the 8-agent switching run is the longest at 71 frames).
- The view box is squared off **before** `set_aspect("equal", adjustable="box")`. Using
  `adjustable="datalim"` lets matplotlib rewrite the limits to satisfy the aspect ratio, and
  on a wide formation such as the 2×4 grid it cropped the leftmost column out of frame — the
  GIF showed six of eight agents.
- Static figures use `plotting.apply_style("readme")` — larger fonts and an **opaque white**
  canvas. Transparency is the wrong choice here: GitHub's page background shows through, so
  on the dark theme the black axes and labels render dark-on-dark. An opaque figure is
  self-contained and legible under both themes.
- **Animations are forced opaque** by `plotting.save_animation`, overriding that style. A GIF
  carries one bit of alpha, so transparent frames get composited onto their predecessors and
  the whole run accumulates into a single unreadable smear.
- The switching GIF annotates the `SPLIT` / `MERGE` instants directly on the frame and holds
  each label for `event_hold` frames; a one-frame label is 50 ms at 20 fps, present in the
  file and invisible to a viewer.

## Why the split fires during the transient

A disconnected pair only has to satisfy its own edge offset, and with `q_position = 0`
nothing anchors the pair's absolute position — so the split has a free mode available to it.
Splitting *after* the formation has settled excites nothing: both components sit exactly
where they are, the square stays visually intact for the whole split phase, and the event is
invisible. The schedule therefore splits at step 6, while the agents are still converging.
