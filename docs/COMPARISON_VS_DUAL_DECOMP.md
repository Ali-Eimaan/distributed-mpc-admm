# ADMM vs dual decomposition

<!-- TODO(deepseek §13.3): this document must contain measured numbers from this repo, not
     only textbook claims. Implement the dual-decomposition baseline described in section 3
     and populate the table in section 5 with real data. A comparison document with
     hand-waved numbers is worse than none. -->

## 1. The question

<!-- TODO(deepseek §13.3): both methods split the same coupled problem across the same graph and both need
     one round of neighbor communication per iteration. State precisely what differs
     (the augmented term), and why that difference matters more in practice than the
     iteration-complexity bounds suggest. -->

## 2. Dual decomposition

<!-- TODO(deepseek §13.3): the Lagrangian without the quadratic term, the subgradient dual ascent step,
     the step-size condition, and the two structural weaknesses:
       - requires strictly convex f_i, or the inner problem is unbounded / multi-valued
       - step size must be tuned to a Lipschitz constant nobody knows a priori
     Show the update equations side by side with ADMM's so the single added term is
     visually obvious. -->

## 3. Baseline implementation

<!-- TODO(deepseek §13.3): implement dual decomposition in the notebooks against the *same* problem
     instances used in 05_convergence_analysis.ipynb, reusing PerAgentSolver with
     rho = 0 and a dual-ascent outer loop. Same seeds, same tolerances, same hardware.
     Anything less makes the comparison unfalsifiable. -->

## 4. What to measure

<!-- TODO(deepseek §13.3):
     - iterations to 1e-4 across topologies and N
     - sensitivity to the tuning parameter (rho vs step size) -- plot both U-curves on
       one axis; the width of the good region is the real result
     - behaviour when f_i loses strict convexity (set q_velocity = r_input = 0)
     - wall time per iteration (dual decomposition's iterations are cheaper -- account
       for that honestly rather than comparing iteration counts alone)
     - behaviour under packet loss
-->

## 5. Results

<!-- TODO(deepseek §13.3): fill from the notebook. Suggested shape:

| Problem | ADMM iters | Dual decomp iters | ADMM wall (ms) | Dual wall (ms) |
| --- | --- | --- | --- | --- |
| 4 agents, cycle | | | | |
| 4 agents, complete | | | | |
| 8 agents, cycle | | | | |
| 8 agents, path | | | | |
| 4 agents, r_input = 0 | | | | |
-->

## 6. When dual decomposition still wins

<!-- TODO(deepseek §13.3): be fair. It has cheaper iterations, no rho to tune when the Lipschitz constant
     is genuinely known, a smaller memory footprint, and it parallelises across constraints
     rather than across agents. Name the regimes where those matter. A comparison that
     concludes "the method I implemented is better in every case" reads as advocacy and
     will be discounted by exactly the readers this repo is aimed at. -->

## 7. Verdict for this repo

<!-- TODO(deepseek §13.3): state the choice and the reason, in three sentences, referring to the measured
     table rather than to the literature. -->

## References

<!-- TODO(deepseek §13.3): Boyd et al. 2011 sections 2 and 7; Everett 1963 / Danzig-Wolfe for the origin
     of dual decomposition; a distributed-MPC survey for the application context. -->
