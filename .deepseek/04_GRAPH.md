# §4 · Communication graphs, switching, and the lossy channel

**Governs:** `python/distributed_mpc_admm/communication_graph.py`
**Milestone:** M1 — the first thing you implement, and nothing else can be tested before it.
**Done when:** the graph tests in [§10.2](10_TESTS.md) pass and `algebraic_connectivity()` matches
hand-computed values for the 4-node path and cycle.

The ADMM iteration needs three things from this module: who agent `i` talks to, the spectral
quantities that appear in the convergence-rate bound, and a channel that can drop and delay
messages so the synchronous-update assumption can be broken on purpose and measured.

---

## §4.1 `CommunicationGraph`

Undirected, simple, no self-loops in `adjacency`. Internal state:

```python
self._n: int
self._edges: frozenset[tuple[int, int]]      # canonical (min, max)
self._weights: dict[tuple[int, int], float]  # canonical keys
self._adj_cache: np.ndarray | None
self._lap_cache: dict[bool, np.ndarray]
```

Validation, all raising `ValueError`: `n_agents < 1`; any id outside `[0, n_agents)`; `i == j`.
Duplicate and reversed-duplicate edges collapse silently — that is normal input, not an error.

**`add_edge` and `remove_edge` MUST invalidate both caches.** A graph whose `laplacian()` describes
a topology it no longer has produces spectral numbers that are quietly wrong everywhere downstream,
and the switching experiments in [§12.4](12_ANALYSIS.md) mutate graphs constantly.

`__eq__` compares `(n_agents, edges, weights)`. Leave `__hash__` undefined — the object is mutable,
and a hashable mutable graph used as a dict key in the switching schedule is a bug waiting for a
`remove_edge` call.

### The ordering is load-bearing

`neighbors(i)` returns a **sorted tuple**. `closed_neighborhood(i)` is
`tuple(sorted(neighbors(i) + (i,)))` — the self index sits **in sorted position, not at the front**.

This ordering defines the block layout of the per-agent decision vector in both Python
([05_LOCAL_QP.md §5.3](05_LOCAL_QP.md)) and C++
([09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md)). Two implementations that sort differently produce a
parity failure that looks like a numerical problem and is not.

`contributors(j)` returns `closed_neighborhood(j)` for an undirected graph. Keep it a **separate
method**: the z-update averages over *this* set, and the two coincide only in the undirected case.
Collapsing them now makes a future directed variant a rewrite rather than an override.

## §4.2 Factories

| Factory | Edges | Note |
| --- | --- | --- |
| `complete(n)` | all pairs | fastest consensus, worst bandwidth |
| `cycle(n)` | `i -- (i+1) mod n` | **`n == 2` is a single edge, not a doubled one** — handle it |
| `path(n)` | `i -- i+1` | smallest `λ₂` of any connected graph on `n` nodes; the hard case |
| `star(n, center)` | `center -- everyone` | the natural leader-follower pattern |
| `random_connected(n, p, rng)` | Erdős–Rényi | **resample whole graphs until connected** |
| `from_adjacency(A)` | from a symmetric matrix | validate symmetry and a zero diagonal |

`random_connected` MUST resample, not patch. Adding edges to a disconnected sample biases the degree
distribution toward the nodes that happened to be isolated, and the robustness study in
[§12.8](12_ANALYSIS.md) is a statement about that distribution. Raise `RuntimeError` after
`max_tries`.

## §4.3 Spectral quantities

```
L      = D − A                          # normalized=False
L_norm = I − D^{-1/2} A D^{-1/2}        # normalized=True; isolated node → row of zeros
```

Use `scipy.linalg.eigh` (symmetric), not `numpy.linalg.eig`. The latter returns complex eigenvalues
for a real symmetric matrix in general and you will spend an hour on a `ComplexWarning`.

`algebraic_connectivity()` returns the **second-smallest** eigenvalue, **clipped at 0 from below**.
The smallest is analytically zero and floating point routinely delivers `-1e-16`, which then
propagates as a `nan` through a `sqrt` three functions later.

`is_connected()` MUST be `algebraic_connectivity() > 1e-10`, not a separate BFS. One definition of
connectivity, one code path — otherwise the two disagree at exactly the marginal graphs the
robustness study cares about.

`spectral_gap_ratio()` is `λ₂/λ_max`. It is what the linear-rate bound in
[13_DOCS.md §13.2](13_DOCS.md) depends on, and it is what [§12.5](12_ANALYSIS.md) plots iterations
against.

## §4.4 `TimeVaryingGraph`

Normalise both construction modes into one internal callable in `__init__`:

```python
if callable(schedule):
    self._fn = schedule
else:
    graphs = tuple(schedule)          # validate identical n_agents across all of them
    if mode == "hold":
        self._fn = lambda k: graphs[min(k, len(graphs) - 1)]
    elif mode == "cycle":
        self._fn = lambda k: graphs[k % len(graphs)]
    else:
        raise ValueError(...)
```

Cache `at(k)` in a dict. A callable schedule may be expensive, and `at` is called once per control
step and again for every plotting routine that draws the topology timeline.

`switching(graphs, dwell_time, mode)` maps step `k` to graph index `k // dwell_time`, then applies
`mode`. `dwell_time` is exposed as a first-class parameter because it is the quantity that appears
as the average-dwell-time condition in switched-systems arguments, and [§12.4](12_ANALYSIS.md)
sweeps it.

`union_over(k0, k1)` returns a new graph with the union of the edge sets over the half-open window.
`is_jointly_connected(k0, k1)` is `union_over(k0, k1).is_connected()`.

Joint connectivity over a bounded window is the standard weakening of "connected at every instant"
that switching-topology consensus results need. [§12.8](12_ANALYSIS.md) constructs schedules that
are jointly connected but never instantaneously connected, and reports whether convergence
survives. Do not assume it does.

`switch_steps(k0, k1)` returns the `k` where `at(k) != at(k-1)`, using `CommunicationGraph.__eq__`.

## §4.5 `LossyChannel`

Bernoulli packet loss plus bounded integer delay. State:

```python
self._mailbox: dict[tuple[int, int], Message]   # (receiver, subject) -> freshest arrived
self._inflight: list[tuple[int, Message]]       # (arrival_iteration, message)
self._stats: ChannelStats
```

`send(message, iteration)`:

1. `stats.sent += 1`, `stats.bytes_sent += payload.nbytes`
2. `if self._rng.random() < loss_prob:` → `stats.dropped += 1`, return `False`
3. `d = self._rng.integers(0, max_delay + 1)`
4. push `(iteration + d, message)` onto `_inflight`, return `True`

`advance(iteration)` moves everything with `arrival <= iteration` into `_mailbox`, keeping the entry
with the **largest `admm_iteration`** when several arrive for the same `(receiver, subject)`.

> Arrival order and production order are not the same thing under a random delay. Taking the last
> *arrival* silently reorders time and produces an agent that acts on an older iterate than the one
> it already had — a subtle, seed-dependent regression that no test will catch unless you write
> this one correctly the first time.

`receive(receiver, subject, iteration)` returns the mailbox entry or `None`, and records
`iteration - message.admm_iteration` in `stats.staleness_histogram`.

`set_graph(graph)` swaps the topology mid-run and **does not clear the mailboxes**. An agent keeps
the last thing it heard from a node that has since disconnected. That is realistic, and the
split/merge experiment in [§12.4](12_ANALYSIS.md) depends on it.

Every stochastic path takes an explicit `rng: np.random.Generator | int | None`. **Never call
`np.random.*` module-level functions** ([§16.4](16_CONVENTIONS.md)) — a robustness result that
cannot be reproduced from a seed is not a result.

## §4.6 `communication_load`

Per ADMM iteration, agent `i` sends `|N_i|` local copies and `|N_i|` consensus broadcasts, each of
`horizon * dim` floats:

```
packets_per_iteration = 2 * Σ_i |N_i| = 4 * |E|
bytes = packets_per_iteration * admm_iterations * horizon * dim * float_bytes
```

Return `{"packets", "bytes", "bytes_per_agent", "packets_per_iteration"}`.

[§12.7](12_ANALYSIS.md) cross-checks this closed form against `LossyChannel.stats` from a real run
at `loss_prob = 0`. If they disagree, **the model is wrong, not the measurement** — fix the formula
here rather than adjusting the plot.

## §4.7 Tests owned by this section

In `python/tests/test_formation_consensus.py` ([§10.2](10_TESTS.md)):

- factories produce the expected edge sets, including `cycle(2)`
- `closed_neighborhood` is sorted and contains self in sorted position
- `laplacian` rows sum to zero; `λ₂ = 0` iff disconnected
- `λ₂` matches the analytic values: `path(4) = 2 − √2 ≈ 0.5858`, `cycle(4) = 2`, `complete(4) = 4`
- `add_edge` / `remove_edge` change `λ₂` (the cache-invalidation test)
- `LossyChannel` at `loss_prob = 0`, `max_delay = 0` delivers everything, and `stats` agrees with
  `communication_load`
- `LossyChannel` at `loss_prob = 1` delivers nothing and `receive` returns `None` rather than
  raising
