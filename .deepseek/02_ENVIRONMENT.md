# §2 · Environment

Target platform: **Ubuntu 26.04 LTS + ROS 2 Lyrical Luth** (distro id `lyrical`), matching the
author's `mpc-cbf` and `cbf-safety-filter` toolchain so that one environment serves all three
repositories.

Pinned, and pinned in one place each:

| Component | Version | Pinned in |
| --- | --- | --- |
| Ubuntu | 26.04 LTS | container image `ros:lyrical-ros-base` |
| ROS 2 | Lyrical Luth (`lyrical`) | `ROS_DISTRO` env in `test.yml` |
| Python | ≥ 3.12, CI on 3.12 and 3.13 | `python/pyproject.toml` (`requires-python`), `test.yml` |
| NumPy | ≥ 1.24 | `pyproject.toml` |
| SciPy | ≥ 1.10 | `pyproject.toml` |
| CVXPY | ≥ 1.4 | `pyproject.toml` — **DPP behaviour varies across minor versions, see §2.1 V8** |
| OSQP (Python) | ≥ 0.6.3 | `pyproject.toml` |
| **OSQP (C++)** | **UNVERIFIED — pin the version you build against** | `package.xml` (`osqp_vendor`), `CMakeLists.txt` |
| C++ | 20 (sources stay C++17-compatible, §16.7) | `CMAKE_CXX_STANDARD` in `CMakeLists.txt` |
| CMake | ≥ 3.22 (range `3.22...3.31`) | `cmake_minimum_required` |
| Eigen | 3.4 (system) | `package.xml` (`eigen`) |
| libzmq / cppzmq | optional | `CMakeLists.txt`, guarded by `CPP_ADMM_WITH_ZMQ` |
| gtest | via `ament_cmake_gtest` | `package.xml` |

---

## §2.1 Version risk register — read this before writing any CI

Lyrical Luth is a **young distro**. Several rows above are forward-looking assumptions, not
verified facts. **Resolve each row below first.** Each fails in a way that wastes hours if you
discover it midway through M8.

Rows V1–V7 carry over verified from the sibling repositories; V8–V11 are new and specific to this
one.

| # | Assumption | How it fails | Verify by |
| --- | --- | --- | --- |
| V1 | GitHub provides an `ubuntu-26.04` runner label | workflow never starts: "no runner matching labels" | **already mitigated** — the workflow uses `ubuntu-latest` plus a `ros:lyrical-*` container, which is what determines the build environment. Do not "fix" this by pinning the runner unless you have confirmed the label exists. |
| V2 | The `ros:lyrical-ros-base` image exists on Docker Hub | `docker pull` fails in every job | pull it locally once |
| V3 | REP 2000 specifies **C++20** for Lyrical | builds locally, then fails or warns on the official toolchain | read REP 2000. If it says C++17, change `CMAKE_CXX_STANDARD` to `17` — the sources use no C++20 feature, so it is a one-line change. Keep it that way (§16.7). |
| V4 | Ubuntu 26.04 ships CPython ≥ 3.13 | `requires-python` mismatch; linters target the wrong version | `python3 --version` in the container |
| V5 | `ament_lint_auto` / `ament_cmake_gtest` are published as debs for `lyrical` | `colcon test` fails at configure | `apt-cache search ros-lyrical-ament` inside the container |
| V6 | `ament_target_dependencies` is deprecated from Kilted onward | deprecation warnings, removal in a future distro | **verified in the sibling repos** — use `target_link_libraries` with `${pkg_TARGETS}` ([§3.4](03_BUILD_SYSTEM.md)); the skeleton already does |
| V7 | `ament_copyright` does not understand bare SPDX identifiers | copyright lint fails with `license=<unknown>` | **verified in the sibling repos** — see §2.6 |
| V8 | The CVXPY DPP formulation in [05_LOCAL_QP.md §5.4](05_LOCAL_QP.md) is accepted as DPP-compliant by your CVXPY version | no error, no crash — just a full re-canonicalisation on every solve, which makes a 4-agent closed-loop run roughly 100× slower and looks like "ADMM is slow" | assert `problem.is_dcp(dpp=True)` in `CvxpyAgentSolver.__init__` and **raise if False**. Do not let this degrade silently. |
| V9 | `osqp_vendor` resolves through rosdep on `lyrical` | `colcon build` fails at `find_package(osqp)` | `rosdep resolve osqp_vendor --rosdistro lyrical` in the container. If it does not, build OSQP from source in the workflow ([§14.3](14_CI.md)) and pin the tag here. |
| V10 | The OSQP **1.x** C API (`OSQPSolver`, `osqp_setup`, `osqp_update_data_vec`, `osqp_update_data_mat`) is what you have | compile errors, or worse, an update call that silently does nothing and leaves the QP solving last iteration's problem | read the installed `osqp.h`. The 0.6 API is different (`OSQPWorkspace`, `osqp_update_lin_cost`, `osqp_update_bounds`). Put the verified names in a comment in `per_agent_qp.cpp`. |
| V11 | OSQP accepts `P` supplied as **upper-triangular** CSC | silently wrong Hessian; the solve succeeds and returns the optimum of a different problem | read the installed docs and assert it in `SetupSucceedsAndHessianIsPsd` ([§10.4](10_TESTS.md)) |

Where a row turns out false, **fix the pin and update the table row to say what is actually true.**
Do not leave a stale assumption here — a version register nobody trusts is worse than no register.

V8, V10 and V11 deserve special care. V8 costs you a wrong performance conclusion; V10 and V11 cost
you a correct-looking wrong answer, which is this repository's characteristic failure.

## §2.2 OSQP

The Python side gets OSQP through CVXPY and needs nothing special. The C++ side needs the library
and headers.

Preferred: `osqp_vendor` through rosdep (risk V9). If that does not resolve, build from source with
a pinned tag:

```bash
git clone --recursive https://github.com/osqp/osqp.git && cd osqp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DOSQP_BUILD_SHARED_LIB=ON
cmake --build build --target install -j"$(nproc)"
```

**Record the tag you actually built** here and in the workflow. `main` is not a pin, and "whatever
was current in August" is not reproducible.

**Do not vendor a copy of OSQP into this repository.** It is a dependency, and a vendored solver
that drifts from the packaged one produces a parity failure that looks like a bug in the
controller.

Two consequences that shape the code:

1. **The QP sparsity pattern is fixed for the lifetime of a `PerAgentQp`.** Every per-iteration
   update is a value-only update ([09_CPP_KERNEL.md §9.3](09_CPP_KERNEL.md)). This is the entire
   reason that class exists.
2. **Only `per_agent_qp.cpp` includes OSQP headers.** A backend change touches one file.

## §2.3 Local setup

```bash
mkdir -p ~/ws/src && cd ~/ws/src && git clone <this-repo> distributed-mpc-admm
```

The repository lives on a Windows machine. The **Python path builds and tests natively on Windows**
and you should use that for M1–M6 — it is the fastest loop available and needs no container.

The **C++ / ROS 2 path (M7–M9) MUST be built inside WSL2 (Ubuntu 26.04) or the
`ros:lyrical-ros-base` container**, exactly as CI does. Do not attempt a native Windows ROS 2
build; nothing here has been validated for it.

If `ros:lyrical-ros-base` is not yet published when you start, use `ros:rolling-ros-base` with
`ROS_DISTRO=rolling` for **local work only**. Do not commit that to the workflow.

### PEP 668

Ubuntu 26.04 marks the system interpreter externally-managed, so `pip install` into it is refused
outright. Create the venv with `--system-site-packages` so the ROS 2 Python modules stay importable:

```bash
python3 -m venv --system-site-packages ~/.venvs/dmpc && . ~/.venvs/dmpc/bin/activate && pip install -e ~/ws/src/distributed-mpc-admm/python[dev]
```

**Do not use `--break-system-packages`.** It works right up until it corrupts a rosdep-installed
package, and then the failure presents as a ROS bug and costs a day to trace.

The CI python job is the one exception — it uses `actions/setup-python`, so PEP 668 never enters
the picture there.

## §2.4 Build and test commands

Python — the loop you will spend M1–M6 in:

```bash
cd python && pip install -e ".[dev]" && pytest -m "not slow"
```

```bash
cd python && ruff check . && black --check . && mypy distributed_mpc_admm
```

Notebooks:

```bash
cd python && pytest --nbmake --nbmake-timeout=900 notebooks/
```

C++ / ROS 2:

```bash
source /opt/ros/lyrical/setup.bash
```

```bash
colcon build --packages-select cpp_admm --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

```bash
colcon test --packages-select cpp_admm --event-handlers console_direct+ && colcon test-result --verbose
```

**Always benchmark in Release.** A Debug OSQP solve time is meaningless and quoting one in the
README would be worse than quoting none. A8 is a Release-only criterion.

## §2.5 ROS 2 API notes for Lyrical Luth

Changes since the Jazzy-era idiom that this codebase touches. Carried over verified from the
sibling repositories. If you find another, add it here rather than fixing it silently in one file.

| Area | Old idiom | What this package does |
| --- | --- | --- |
| CMake dependency linkage | `ament_target_dependencies(tgt dep…)` | deprecated from Kilted onward; use `target_link_libraries` with `${pkg_TARGETS}` — see [03_BUILD_SYSTEM.md §3.4](03_BUILD_SYSTEM.md) |
| CMake minimum | `3.16` | `3.22...3.31`; CMake 4.x warns on older compatibility levels |
| Python packaging | `license = { file = … }` | PEP 639 SPDX string plus `license-files`, requires `setuptools>=77` |

> **Note on the distro choice.** [INFO.md](INFO.md) was written against ROS 2 Jazzy. Lyrical is
> used here because the author's sibling repositories have already been migrated and validated on
> it, and running three repositories on two distros doubles the environment work for no benefit.
> Nothing in this package depends on a Lyrical-only feature — falling back to Jazzy means changing
> `ROS_DISTRO`, the container tag and the CMake minimum, and nothing else. If you make that change,
> record it here.

## §2.6 Licence

**BSD-3-Clause** (`LICENSE` at the repository root, `Copyright (c) 2026, Ali-Eimaan`) — the
conventional choice for ROS 2 packages, what downstream ROS users expect, and the same licence as
the author's `mpc-cbf` and `cbf-safety-filter`, so the repositories can share code without a
compatibility question.

> The repository was created under MIT and relicensed to BSD-3-Clause on the author's instruction
> while still unpublished and single-author, so no third-party contribution needed re-consenting.
> If that ever stops being true, a further licence change needs every contributor's agreement.

Every source file starts with:

```
// Copyright (c) 2026, Ali-Eimaan. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
```

(`#` comments in Python, CMake and YAML; `%` in LaTeX and BibTeX; after any shebang line.)
**New files MUST carry this header.** Jupyter notebooks are exempt — the header would appear as a
rendered cell; record their licence in `media/README.md` and the root README instead.

Note the third BSD clause: the copyright holder's name may not be used to endorse derived products.
That is a real obligation on downstream users and a reason to keep the copyright line accurate
rather than generic.

> **Why an SPDX line may not be enough.** `ament_copyright` matches file content against full
> licence-text templates rather than parsing SPDX identifiers:
> `SourceDescriptor.identify_license()` compares against the templates in
> `ament_copyright/licenses.py`, so a header carrying only a copyright line and an SPDX id is
> reported as `license=<unknown>` and fails the copyright lint. This was **verified on Lyrical in
> the sibling repositories** (risk V7).
>
> Two routes, and the skeleton currently takes the second:
>
> 1. Append the short `bsd_3clause` `file_headers` template text below the SPDX line. **Copy it out
>    of the installed `ament_copyright/licenses.py`; do not retype it from memory** (rule 4,
>    [00_RULES.md](00_RULES.md)) — the match is textual, so an approximation fails silently.
> 2. Disable the copyright linter in `CMakeLists.txt` with a comment saying why —
>    `set(ament_cmake_copyright_FOUND TRUE)`, which is what the skeleton does. A deliberate choice
>    for a single-author research repository, not an oversight.
>
> Run `ament_copyright cpp_admm` before changing which route is in force. If a future distro ships
> an `ament_copyright` that understands SPDX identifiers, prefer route 1 without the template text
> and re-enable the linter.

`package.xml` declares `<license>BSD-3-Clause</license>` and `pyproject.toml` declares
`license = "BSD-3-Clause"`; both MUST agree with `LICENSE`.
