// Copyright (c) 2026, Ali-Eimaan. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// gtest suite for the C++ kernel.
//
// The load-bearing test in this file is PythonParity: the C++ kernel and the Python
// reference must produce the same iterates on the same problem. Everything else guards a
// specific way the OSQP-level optimisations can silently corrupt the math.
//
// Fixtures load their expected values from `test/data/*.json`, exported by
// `python/notebooks/02_4_agent_consensus.ipynb`. Regenerating that data is a deliberate
// act -- if a change makes the parity test fail, fix the C++ or justify the export, but
// never regenerate the fixture to make a red test go green.

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cpp_admm/admm_kernel.hpp"
#include "cpp_admm/per_agent_qp.hpp"

// ----------------------------------------------------------------- global new hook
//
// The NoHeapAllocationDuringIterate test overrides the global allocator to prove the
// header's real-time claim. The hook is disabled unless g_count_allocs is true so the
// rest of the test binary (gtest, Eigen warm-up, nlohmann) runs through it untouched.

namespace
{
std::atomic<long> g_alloc_count{0};
std::atomic<bool> g_count_allocs{false};
}  // namespace

// Replacing the global allocator is exactly what this test is for, but GCC's
// interprocedural analysis pairs the *builtin* operator new (which it still sees inlined
// inside header-only code such as nlohmann/json) with the replacement operator delete
// below and reports -Wmismatched-new-delete. Every replacement here is malloc/free
// throughout, so the pairing is consistent; the diagnostic is a false positive of the
// inlining, not a real mismatch. Scoped to these definitions only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void * operator new(std::size_t sz)
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void * p = std::malloc(sz)) {
    return p;
  }
  throw std::bad_alloc();
}

void * operator new[](std::size_t sz)
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void * p = std::malloc(sz)) {
    return p;
  }
  throw std::bad_alloc();
}

void * operator new(std::size_t sz, const std::nothrow_t &) noexcept
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  return std::malloc(sz);
}

void * operator new[](std::size_t sz, const std::nothrow_t &) noexcept
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  return std::malloc(sz);
}

void * operator new(std::size_t sz, std::align_val_t al)
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  const std::size_t a = static_cast<std::size_t>(al);
  const std::size_t rounded = (sz + a - 1) / a * a;
  if (void * p = std::aligned_alloc(a, rounded)) {
    return p;
  }
  throw std::bad_alloc();
}

void * operator new[](std::size_t sz, std::align_val_t al)
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  const std::size_t a = static_cast<std::size_t>(al);
  const std::size_t rounded = (sz + a - 1) / a * a;
  if (void * p = std::aligned_alloc(a, rounded)) {
    return p;
  }
  throw std::bad_alloc();
}

void * operator new(std::size_t sz, std::align_val_t al, const std::nothrow_t &) noexcept
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  const std::size_t a = static_cast<std::size_t>(al);
  const std::size_t rounded = (sz + a - 1) / a * a;
  return std::aligned_alloc(a, rounded);
}

void * operator new[](std::size_t sz, std::align_val_t al, const std::nothrow_t &) noexcept
{
  if (g_count_allocs.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
  const std::size_t a = static_cast<std::size_t>(al);
  const std::size_t rounded = (sz + a - 1) / a * a;
  return std::aligned_alloc(a, rounded);
}

void operator delete(void * p) noexcept {std::free(p);}
void operator delete[](void * p) noexcept {std::free(p);}
void operator delete(void * p, std::size_t) noexcept {std::free(p);}
void operator delete[](void * p, std::size_t) noexcept {std::free(p);}
void operator delete(void * p, const std::nothrow_t &) noexcept {std::free(p);}
void operator delete[](void * p, const std::nothrow_t &) noexcept {std::free(p);}
void operator delete(void * p, std::align_val_t) noexcept {std::free(p);}
void operator delete[](void * p, std::align_val_t) noexcept {std::free(p);}
void operator delete(void * p, std::size_t, std::align_val_t) noexcept {std::free(p);}
void operator delete[](void * p, std::size_t, std::align_val_t) noexcept {std::free(p);}
void operator delete(void * p, std::align_val_t, const std::nothrow_t &) noexcept {std::free(p);}
void operator delete[](void * p, std::align_val_t, const std::nothrow_t &) noexcept {std::free(p);}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace cpp_admm
{
namespace
{

// --------------------------------------------------------------------------- helpers

/// 4 agents on a cycle, rendezvous formation, horizon 10, dt 0.1.
AgentConfig makeConfig(int agent_id)
{
  AgentConfig c;
  c.agent_id = agent_id;
  c.n_agents = 4;
  c.horizon = 10;
  c.dim = 2;
  c.dt = 0.1;

  switch (agent_id) {
    case 0: c.neighbors = {1, 3}; break;
    case 1: c.neighbors = {0, 2}; break;
    case 2: c.neighbors = {1, 3}; break;
    case 3: c.neighbors = {0, 2}; break;
    default: break;
  }
  for (int j : c.neighbors) {
    c.offsets[j] = Eigen::VectorXd::Zero(c.dim);
  }
  return c;
}

/// A deterministic, feasible initial state for the four-agent tests: positions on the
/// corners of a square, zero velocity. Size 2*dim.
Eigen::VectorXd makeInitialState(int agent_id)
{
  Eigen::VectorXd x0(4);
  const double px[4] = {1.0, -1.0, 1.5, -1.5};
  const double py[4] = {1.0, 1.0, -1.0, -1.0};
  x0 << px[agent_id], py[agent_id], 0.0, 0.0;
  return x0;
}

/// Owning handles for four wired kernels plus their connected loopback transports.
struct FourKernelRig
{
  std::vector<std::unique_ptr<InProcessTransport>> transports;
  std::vector<std::unique_ptr<AdmmKernel>> kernels;

  /// Run one iterate() on every agent in id order, then report whether all converged.
  bool stepAll()
  {
    bool all_converged = true;
    for (auto & k : kernels) {
      all_converged = k->iterate() && all_converged;
    }
    return all_converged;
  }
};

/// Build four kernels wired through InProcessTransport.
FourKernelRig buildFourKernels(
  double loss_prob, int max_delay, uint64_t seed,
  ADMMOptions options = ADMMOptions{})
{
  FourKernelRig rig;
  for (int i = 0; i < 4; ++i) {
    rig.transports.push_back(
      std::make_unique<InProcessTransport>(i, loss_prob, max_delay,
          seed + static_cast<uint64_t>(i)));
  }
  std::vector<InProcessTransport *> raw;
  raw.reserve(rig.transports.size());
  for (auto & t : rig.transports) {
    raw.push_back(t.get());
  }
  InProcessTransport::connect(raw);

  for (int i = 0; i < 4; ++i) {
    rig.kernels.push_back(std::make_unique<AdmmKernel>(
      makeConfig(i), options, rig.transports[static_cast<std::size_t>(i)].get()));
    rig.kernels.back()->configure();
    rig.kernels.back()->setInitialState(makeInitialState(i));
  }
  return rig;
}

// Parity tolerances. The C++ and Python QPs are the *same* optimisation problem but are
// canonicalised differently by OSQP (cvxpy adds epigraph slacks for `abs() <= b` and
// orders variables differently), so the comparison is on extracted U/y values rather than
// bit-for-bit. 1e-6 is comfortably below the OSQP eps=1e-8 stopping tolerance.
constexpr double kIterateTol = 1e-6;
constexpr double kOptimumTol = 1e-6;

/// Zero-valued consensus/dual maps covering the full closed neighborhood of `c`.
std::unordered_map<int, Eigen::VectorXd> zeroConsensus(const AgentConfig & c)
{
  std::unordered_map<int, Eigen::VectorXd> m;
  for (int j : c.closed_neighborhood()) {
    m[j] = Eigen::VectorXd::Zero(c.horizon * c.dim);
  }
  return m;
}

/// JSON array -> Eigen vector.
Eigen::VectorXd jsonVec(const nlohmann::json & j)
{
  Eigen::VectorXd v(static_cast<Eigen::Index>(j.size()));
  for (std::size_t k = 0; k < j.size(); ++k) {
    v(static_cast<Eigen::Index>(k)) = j[k].get<double>();
  }
  return v;
}

/// Sparse P (upper-triangular CSC) -> dense triplets, for structural comparison.
std::vector<Eigen::Triplet<double>> sparseTriplets(const Eigen::SparseMatrix<double> & S)
{
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(S.nonZeros()));
  for (int k = 0; k < S.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
      trips.emplace_back(it.row(), it.col(), it.value());
    }
  }
  return trips;
}

// ------------------------------------------------------------------------- dynamics

TEST(DoubleIntegratorTest, MatricesMatchClosedForm)
{
  const double dt = 0.1;
  const int dim = 2;
  DoubleIntegrator model(dt, dim);

  // A = [[I, dt*I], [0, I]]; B = [[0.5*dt^2*I], [dt*I]].
  Eigen::MatrixXd A_exp = Eigen::MatrixXd::Zero(4, 4);
  Eigen::MatrixXd B_exp = Eigen::MatrixXd::Zero(4, 2);
  for (int d = 0; d < dim; ++d) {
    A_exp(d, d) = 1.0;
    A_exp(d, dim + d) = dt;
    A_exp(dim + d, dim + d) = 1.0;
    B_exp(d, d) = 0.5 * dt * dt;
    B_exp(dim + d, d) = dt;
  }

  EXPECT_TRUE(model.A().isApprox(A_exp, 1e-15));
  EXPECT_TRUE(model.B().isApprox(B_exp, 1e-15));
}

TEST(DoubleIntegratorTest, PredictionMatchesRollout)
{
  const double dt = 0.1;
  const int dim = 2;
  const int horizon = 10;
  DoubleIntegrator model(dt, dim);

  const int n = 2 * dim;
  const int m = dim;
  Eigen::MatrixXd phi, gamma;
  model.predictionMatrices(horizon, phi, gamma);

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Eigen::VectorXd U = Eigen::VectorXd::Random(horizon * m);

  const Eigen::VectorXd pred = phi * x0 + gamma * U;

  Eigen::VectorXd x = x0;
  Eigen::VectorXd rollout(horizon * n);
  for (int t = 0; t < horizon; ++t) {
    x = model.A() * x + model.B() * U.segment(t * m, m);
    rollout.segment(t * n, n) = x;
  }

  EXPECT_TRUE(pred.isApprox(rollout, 1e-12));
}

TEST(DoubleIntegratorTest, PositionRowsAreSubsetOfFullPrediction)
{
  const double dt = 0.1;
  const int dim = 2;
  const int horizon = 10;
  DoubleIntegrator model(dt, dim);

  const int n = 2 * dim;
  Eigen::MatrixXd phi, gamma;
  model.predictionMatrices(horizon, phi, gamma);
  Eigen::MatrixXd phi_p, gamma_p;
  model.positionPredictionMatrices(horizon, phi_p, gamma_p);
  Eigen::MatrixXd phi_v, gamma_v;
  model.velocityPredictionMatrices(horizon, phi_v, gamma_v);

  for (int t = 0; t < horizon; ++t) {
    EXPECT_TRUE(phi_p.block(t * dim, 0, dim, n).isApprox(phi.block(t * n, 0, dim, n), 1e-14));
    EXPECT_TRUE(phi_v.block(t * dim, 0, dim, n).isApprox(
      phi.block(t * n + dim, 0, dim, n), 1e-14));
    EXPECT_TRUE(
      gamma_p.block(t * dim, 0, dim, horizon * dim)
      .isApprox(gamma.block(t * n, 0, dim, horizon * dim), 1e-14));
    EXPECT_TRUE(
      gamma_v.block(t * dim, 0, dim, horizon * dim)
      .isApprox(gamma.block(t * n + dim, 0, dim, horizon * dim), 1e-14));
  }
}

// ------------------------------------------------------------------------------- QP

TEST(PerAgentQpTest, SetupSucceedsAndHessianIsPsd)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();
  ASSERT_TRUE(qp.isReady());

  const Eigen::SparseMatrix<double> & P = qp.hessian();
  for (int k = 0; k < P.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(P, k); it; ++it) {
      EXPECT_LE(it.row(), it.col()) << "P must be upper-triangular CSC";
    }
  }

  const Eigen::MatrixXd Ps = Eigen::MatrixXd(P).selfadjointView<Eigen::Upper>();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ps);
  EXPECT_GE(es.eigenvalues().minCoeff(), -1e-12) << "P must be positive semi-definite";
}

TEST(PerAgentQpTest, BlockLayoutMatchesHeaderContract)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();

  const int M = 3;  // closed neighborhood {0, 1, 3}
  EXPECT_EQ(qp.inputOffset(), 0);
  EXPECT_EQ(qp.numVariables(), (1 + M) * 10 * 2);
  EXPECT_EQ(qp.numConstraints(), 3 * 10 * 2);

  EXPECT_EQ(qp.copyOffset(0), 1 * 20);
  EXPECT_EQ(qp.copyOffset(1), 2 * 20);
  EXPECT_EQ(qp.copyOffset(3), 3 * 20);
}

TEST(PerAgentQpTest, DynamicsConstraintIsSatisfied)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();

  Eigen::VectorXd x0(4);
  x0 << 0.5, -0.25, 0.1, 0.2;
  qp.updateInitialState(x0);
  qp.updateConsensus(zeroConsensus(makeConfig(0)), zeroConsensus(makeConfig(0)), 1.0);
  const QpSolution & sol = qp.solve();
  ASSERT_TRUE(sol.ok());

  DoubleIntegrator model(0.1, 2);
  Eigen::MatrixXd phi_p, gamma_p;
  model.positionPredictionMatrices(10, phi_p, gamma_p);

  const Eigen::VectorXd U = qp.extractInputs(sol.theta);
  const Eigen::VectorXd y_self = qp.extractCopy(sol.theta, 0);
  const Eigen::VectorXd expected = phi_p * x0 + gamma_p * U;

  EXPECT_TRUE(y_self.isApprox(expected, 1e-9));
}

TEST(PerAgentQpTest, InputLimitsRespected)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();

  // Target 5 m away: the input cost is tiny and the velocity cap is 2 m/s, so the
  // controller pins the input at u_max for the first several steps.
  Eigen::VectorXd x0(4);
  x0 << 0.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd ref(20);
  ref.setConstant(5.0);
  qp.updateInitialState(x0);
  qp.updateReference(ref);
  qp.updateConsensus(zeroConsensus(makeConfig(0)), zeroConsensus(makeConfig(0)), 1.0);
  const QpSolution & sol = qp.solve();
  ASSERT_TRUE(sol.ok());

  const double u_max = makeConfig(0).u_max;
  const Eigen::VectorXd U = qp.extractInputs(sol.theta);
  EXPECT_LE(U.maxCoeff(), u_max + 1e-6);
  EXPECT_GE(U.minCoeff(), -u_max - 1e-6);
  // Confirm saturation actually occurred, otherwise the box check above is vacuous.
  EXPECT_GT(U.cwiseAbs().maxCoeff(), u_max - 1e-3);
}

TEST(PerAgentQpTest, UpdateConsensusChangesOnlyLinearTerm)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();
  qp.updateInitialState(Eigen::VectorXd::Zero(4));
  qp.updateConsensus(zeroConsensus(makeConfig(0)), zeroConsensus(makeConfig(0)), 1.0);
  const QpSolution s0 = qp.solve();
  ASSERT_TRUE(s0.ok());

  const auto before = sparseTriplets(qp.hessian());

  auto z1 = zeroConsensus(makeConfig(0));
  auto lam1 = zeroConsensus(makeConfig(0));
  for (int j : makeConfig(0).closed_neighborhood()) {
    z1[j] = Eigen::VectorXd::Ones(20);
    lam1[j] = Eigen::VectorXd::Constant(20, 0.5);
  }
  qp.updateConsensus(z1, lam1, 1.0);
  const QpSolution s1 = qp.solve();
  ASSERT_TRUE(s1.ok());

  // P is bit-identical after a consensus update.
  const auto after = sparseTriplets(qp.hessian());
  ASSERT_EQ(after.size(), before.size());
  for (std::size_t k = 0; k < before.size(); ++k) {
    EXPECT_EQ(after[k].row(), before[k].row());
    EXPECT_EQ(after[k].col(), before[k].col());
    EXPECT_DOUBLE_EQ(after[k].value(), before[k].value());
  }

  // ... but the primal solution moved.
  EXPECT_FALSE(s0.theta.isApprox(s1.theta, 1e-9));
}

TEST(PerAgentQpTest, UpdateRhoTouchesOnlyDiagonalEntries)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();
  qp.updateRho(1.0);  // no-op; P is already built with rho == 1.0

  const auto before = sparseTriplets(qp.hessian());
  qp.updateRho(2.0);
  const auto after = sparseTriplets(qp.hessian());

  ASSERT_EQ(after.size(), before.size());
  const int B = 20;
  const int expected_changed = 3 * B;  // rho*I over the three y blocks only

  int changed = 0;
  for (std::size_t k = 0; k < before.size(); ++k) {
    ASSERT_EQ(after[k].row(), before[k].row());
    ASSERT_EQ(after[k].col(), before[k].col());
    if (after[k].value() != before[k].value()) {
      EXPECT_EQ(after[k].row(), after[k].col()) << "only the diagonal may change";
      EXPECT_GE(after[k].row(), B) << "the U block diagonal must be untouched";
      EXPECT_DOUBLE_EQ(after[k].value() - before[k].value(), 1.0);
      ++changed;
    }
  }
  EXPECT_EQ(changed, expected_changed);
}

TEST(PerAgentQpTest, WarmStartReducesInnerIterations)
{
  PerAgentQp qp(makeConfig(0), QpSettings{});
  qp.setup();

  Eigen::VectorXd x0(4);
  x0 << 1.0, -0.5, 0.3, -0.2;
  qp.updateInitialState(x0);
  qp.updateConsensus(zeroConsensus(makeConfig(0)), zeroConsensus(makeConfig(0)), 1.0);
  const QpSolution cold = qp.solve();
  ASSERT_TRUE(cold.ok());

  Eigen::VectorXd x1 = x0;
  x1(0) += 1e-3;  // tiny perturbation -> previous solution is a near-optimal warm start
  qp.updateInitialState(x1);
  const QpSolution warm = qp.solve();
  ASSERT_TRUE(warm.ok());

  EXPECT_LT(warm.iterations, cold.iterations);
}

// ---------------------------------------------------------------------------- kernel

TEST(AdmmKernelTest, ConfigureRejectsInvalidConfig)
{
  const ADMMOptions opts;

  {
    AgentConfig c = makeConfig(0);
    c.neighbors.push_back(0);  // self in neighbors
    AdmmKernel k(c, opts, nullptr);
    EXPECT_THROW(k.configure(), std::invalid_argument);
  }
  {
    AgentConfig c = makeConfig(0);
    c.neighbors = {1, 1};  // duplicate
    AdmmKernel k(c, opts, nullptr);
    EXPECT_THROW(k.configure(), std::invalid_argument);
  }
  {
    AgentConfig c = makeConfig(0);
    c.q_position = -1.0;  // negative weight
    AdmmKernel k(c, opts, nullptr);
    EXPECT_THROW(k.configure(), std::invalid_argument);
  }
  {
    AgentConfig c = makeConfig(0);
    c.offsets[2] = Eigen::VectorXd::Zero(2);  // offset for a non-neighbor
    AdmmKernel k(c, opts, nullptr);
    EXPECT_THROW(k.configure(), std::invalid_argument);
  }
}

TEST(AdmmKernelTest, SingleAgentConvergesImmediately)
{
  AgentConfig c;
  c.agent_id = 0;
  c.n_agents = 1;
  c.neighbors = {};
  c.horizon = 10;
  c.dim = 2;
  c.dt = 0.1;

  ADMMOptions opts;
  opts.alpha = 1.0;
  AdmmKernel k(c, opts, nullptr);
  k.configure();
  k.setInitialState(makeInitialState(0));
  k.setReference(Eigen::VectorXd::Zero(20));

  // With no neighbors the self-consensus contraction factor is rho/(1+rho) ~ 0.5, so a
  // lone agent settles to solver noise far below any realistic iteration cap.
  const ADMMStats s = k.solve();
  EXPECT_TRUE(s.converged);
  EXPECT_LT(s.iterations, 50);
  EXPECT_NEAR(s.primal_residual, 0.0, 1e-6);
}

TEST(AdmmKernelTest, FourAgentsReachConsensus)
{
  ADMMOptions opts;
  opts.alpha = 1.0;
  opts.adaptive_rho = false;
  opts.eps_abs = 1e-6;
  opts.eps_rel = 1e-6;
  opts.max_iterations = 2000;
  auto rig = buildFourKernels(0.0, 0, 0, opts);

  bool all_converged = false;
  for (int s = 0; s < 2000; ++s) {
    bool this_step = true;
    for (auto & k : rig.kernels) {
      this_step = k->iterate() && this_step;
    }
    if (this_step) {
      all_converged = true;
      break;
    }
  }
  EXPECT_TRUE(all_converged);

  for (int subject = 0; subject < 4; ++subject) {
    const auto nbhd = makeConfig(subject).closed_neighborhood();
    for (std::size_t a = 0; a < nbhd.size(); ++a) {
      for (std::size_t b = a + 1; b < nbhd.size(); ++b) {
        const int ia = nbhd[a];
        const int ib = nbhd[b];
        EXPECT_TRUE(
          rig.kernels[static_cast<std::size_t>(ia)]->localCopy(subject)
          .isApprox(rig.kernels[static_cast<std::size_t>(ib)]->localCopy(subject), 1e-4))
          << "agents " << ia << " and " << ib << " disagree on subject " << subject;
      }
    }
  }
}

TEST(AdmmKernelTest, NoHeapAllocationDuringIterate)
{
  AgentConfig c;
  c.agent_id = 0;
  c.n_agents = 1;
  c.neighbors = {};
  c.horizon = 10;
  c.dim = 2;
  c.dt = 0.1;

  ADMMOptions opts;
  AdmmKernel k(c, opts, nullptr);
  k.configure();
  k.setInitialState(makeInitialState(0));
  k.setReference(Eigen::VectorXd::Zero(20));

  // The first OSQP solve may perform one-time lazy workspace allocations; the real-time
  // claim is about the steady state, so warm up before arming the counter.
  k.iterate();

  g_alloc_count.store(0, std::memory_order_relaxed);
  g_count_allocs.store(true, std::memory_order_relaxed);
  for (int i = 0; i < 10; ++i) {
    k.iterate();
  }
  g_count_allocs.store(false, std::memory_order_relaxed);

  EXPECT_EQ(g_alloc_count.load(std::memory_order_relaxed), 0);
}

TEST(AdmmKernelTest, ShiftWarmStartPreservesTail)
{
  AgentConfig c;
  c.agent_id = 0;
  c.n_agents = 1;
  c.neighbors = {};
  c.horizon = 10;
  c.dim = 2;
  c.dt = 0.1;

  ADMMOptions opts;
  AdmmKernel k(c, opts, nullptr);
  k.configure();
  Eigen::VectorXd x0(4);
  x0 << 0.3, -0.2, 0.4, -0.1;
  k.setInitialState(x0);
  Eigen::VectorXd ref(20);
  for (int t = 0; t < 10; ++t) {
    ref.segment(2 * t, 2) << std::sin(static_cast<double>(t)), std::cos(static_cast<double>(t));
  }
  k.setReference(ref);
  k.iterate();

  const Eigen::VectorXd old_y = k.localCopy(0);
  const Eigen::VectorXd old_lam = k.dual(0);
  const Eigen::VectorXd old_z = k.consensusTrajectory();

  k.shiftWarmStart();

  const Eigen::VectorXd new_y = k.localCopy(0);
  const Eigen::VectorXd new_lam = k.dual(0);
  const Eigen::VectorXd new_z = k.consensusTrajectory();

  const int d = 2;
  for (int t = 0; t < 9; ++t) {
    EXPECT_TRUE(new_y.segment(t * d, d).isApprox(old_y.segment((t + 1) * d, d), 1e-12));
    EXPECT_TRUE(new_lam.segment(t * d, d).isApprox(old_lam.segment((t + 1) * d, d), 1e-12));
    EXPECT_TRUE(new_z.segment(t * d, d).isApprox(old_z.segment((t + 1) * d, d), 1e-12));
  }
  // The last entry is a duplicate of the previous one.
  EXPECT_TRUE(new_y.segment(9 * d, d).isApprox(old_y.segment(9 * d, d), 1e-12));
  EXPECT_TRUE(new_lam.segment(9 * d, d).isApprox(old_lam.segment(9 * d, d), 1e-12));
  EXPECT_TRUE(new_z.segment(9 * d, d).isApprox(old_z.segment(9 * d, d), 1e-12));
}

TEST(AdmmKernelTest, SetNeighborsPreservesSurvivingBlocks)
{
  AgentConfig c = makeConfig(0);  // neighbors {1, 3}
  ADMMOptions opts;
  AdmmKernel k(c, opts, nullptr);
  k.configure();
  k.setInitialState(makeInitialState(0));
  k.iterate();

  const Eigen::VectorXd y1_before = k.localCopy(1);
  const Eigen::VectorXd lam1_before = k.dual(1);

  k.setNeighbors({1});  // drop neighbor 3

  ASSERT_EQ(k.config().neighbors.size(), 1u);
  EXPECT_EQ(k.config().neighbors[0], 1);
  EXPECT_TRUE(k.localCopy(1).isApprox(y1_before, 1e-15));
  EXPECT_TRUE(k.dual(1).isApprox(lam1_before, 1e-15));
  EXPECT_THROW(static_cast<void>(k.localCopy(3)), std::out_of_range);

  // The rebuilt kernel still runs end to end.
  k.setInitialState(makeInitialState(0));
  EXPECT_NO_THROW(k.iterate());
}

TEST(AdmmKernelTest, StaleDataTriggersEarlyExit)
{
  // Two agents on a path; run agent 0 alone so its neighbor (agent 1) goes silent. Agent 0
  // must bail out on staleness rather than spin the full iteration budget on frozen data.
  auto makePathConfig = [](int id) {
      AgentConfig c;
      c.agent_id = id;
      c.n_agents = 2;
      c.horizon = 10;
      c.dim = 2;
      c.dt = 0.1;
      c.neighbors = id == 0 ? std::vector<int>{1} : std::vector<int>{0};
      for (int j : c.neighbors) {
        c.offsets[j] = Eigen::VectorXd::Zero(2);
      }
      return c;
    };

  ADMMOptions opts;
  opts.max_iterations = 200;
  opts.max_staleness = 3;
  opts.poll_timeout = std::chrono::microseconds(0);

  InProcessTransport t0(0, 0.0, 0, 0);
  InProcessTransport t1(1, 0.0, 0, 1);
  InProcessTransport::connect({&t0, &t1});

  AdmmKernel k0(makePathConfig(0), opts, &t0);
  AdmmKernel k1(makePathConfig(1), opts, &t1);
  k0.configure();
  k1.configure();
  k0.setInitialState(makeInitialState(0));
  k1.setInitialState(makeInitialState(1));

  const ADMMStats & s = k0.solve();

  EXPECT_FALSE(s.converged);
  EXPECT_GT(s.max_staleness_seen, opts.max_staleness);
  EXPECT_LT(s.iterations, opts.max_iterations);
}

TEST(AdmmKernelTest, PacketLossDoesNotProduceNaN)
{
  for (uint64_t seed = 0; seed < 20; ++seed) {
    auto rig = buildFourKernels(0.3, 0, seed);
    for (int s = 0; s < 15; ++s) {
      for (auto & k : rig.kernels) {
        k->iterate();
      }
    }
    for (auto & k : rig.kernels) {
      EXPECT_TRUE(k->inputs().allFinite());
      EXPECT_TRUE(k->consensusTrajectory().allFinite());
      for (int j : k->config().closed_neighborhood()) {
        EXPECT_TRUE(k->localCopy(j).allFinite());
        EXPECT_TRUE(k->dual(j).allFinite());
      }
    }
  }
}

// ----------------------------------------------------------------------- parity

TEST(PythonParityTest, MatchesReferenceIterates)
{
  const std::string path = std::string(CPP_ADMM_TEST_DATA_DIR) + "/four_agent_reference.json";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.good()) << "missing fixture: " << path;
  nlohmann::json fx;
  ifs >> fx;

  const int horizon = fx.at("horizon").get<int>();
  const int dim = fx.at("dim").get<int>();
  const int n_agents = fx.at("n_agents").get<int>();
  const double rho = fx.at("rho").get<double>();
  (void)horizon;
  (void)dim;

  std::vector<std::unique_ptr<PerAgentQp>> qps;
  for (int i = 0; i < n_agents; ++i) {
    QpSettings s;
    s.eps_abs = 1e-8;
    s.eps_rel = 1e-8;
    s.max_iter = 100000;
    qps.push_back(std::make_unique<PerAgentQp>(makeConfig(i), s));
    qps.back()->setup();
    qps.back()->updateRho(rho);
  }

  const auto & iters = fx.at("iterates");
  ASSERT_EQ(iters.size(), 20u);

  for (const auto & it : iters) {
    const auto & agents = it.at("agents");
    for (int i = 0; i < n_agents; ++i) {
      const auto & rec = agents[static_cast<std::size_t>(i)];
      const Eigen::VectorXd x0 = jsonVec(fx.at("x0")[static_cast<std::size_t>(i)]);
      const Eigen::VectorXd ref = jsonVec(fx.at("reference")[static_cast<std::size_t>(i)]);

      std::unordered_map<int, Eigen::VectorXd> z, lam;
      for (int j : makeConfig(i).closed_neighborhood()) {
        const std::string key = std::to_string(j);
        z[j] = jsonVec(rec.at("z_in").at(key));
        lam[j] = jsonVec(rec.at("lam_in").at(key));
      }

      PerAgentQp & qp = *qps[static_cast<std::size_t>(i)];
      qp.updateInitialState(x0);
      qp.updateReference(ref);
      qp.updateConsensus(z, lam, rho);
      const QpSolution & sol = qp.solve();
      ASSERT_TRUE(sol.ok());

      const Eigen::VectorXd U_exp = jsonVec(rec.at("U"));
      EXPECT_TRUE(qp.extractInputs(sol.theta).isApprox(U_exp, kIterateTol))
        << "agent " << i << " iteration " << it.at("iteration") << " U mismatch";

      for (int j : makeConfig(i).closed_neighborhood()) {
        const std::string key = std::to_string(j);
        const Eigen::VectorXd y_exp = jsonVec(rec.at("y").at(key));
        EXPECT_TRUE(qp.extractCopy(sol.theta, j).isApprox(y_exp, kIterateTol))
          << "agent " << i << " iteration " << it.at("iteration") << " y^" << j << " mismatch";
      }
    }
  }
}

TEST(PythonParityTest, MatchesReferenceOptimum)
{
  const std::string path = std::string(CPP_ADMM_TEST_DATA_DIR) + "/four_agent_reference.json";
  std::ifstream ifs(path);
  ASSERT_TRUE(ifs.good()) << "missing fixture: " << path;
  nlohmann::json fx;
  ifs >> fx;

  const int n_agents = fx.at("n_agents").get<int>();
  const double rho = fx.at("rho").get<double>();

  // Full kernels, inner solver tightened to match the fixture's export settings.
  ADMMOptions opts;
  opts.rho = rho;
  opts.alpha = 1.0;
  opts.adaptive_rho = false;
  opts.eps_abs = 1e-8;
  opts.eps_rel = 1e-8;
  opts.max_iterations = 2000;
  opts.qp_settings.eps_abs = 1e-8;
  opts.qp_settings.eps_rel = 1e-8;
  opts.qp_settings.max_iter = 100000;

  auto rig = buildFourKernels(0.0, 0, 0, opts);

  // Feed each kernel the fixture's exact initial state.
  for (int i = 0; i < n_agents; ++i) {
    rig.kernels[static_cast<std::size_t>(i)]->setInitialState(
      jsonVec(fx.at("x0")[static_cast<std::size_t>(i)]));
    rig.kernels[static_cast<std::size_t>(i)]->setReference(
      jsonVec(fx.at("reference")[static_cast<std::size_t>(i)]));
  }

  for (int s = 0; s < opts.max_iterations; ++s) {
    bool any = false;
    for (auto & k : rig.kernels) {
      const bool converged = k->iterate();
      any = any || converged;
    }
    if (any) {
      // Continue a little so every agent reaches the same neighbourhood; the loop below
      // still bounds the total work.
    }
    // Stop once all four agree they are done (async delivery means one may lag by one).
    bool all = true;
    for (auto & k : rig.kernels) {
      all = all && k->stats().converged;
    }
    if (all) {
      break;
    }
  }

  for (int i = 0; i < n_agents; ++i) {
    const Eigen::VectorXd expected = jsonVec(fx.at("final_inputs")[static_cast<std::size_t>(i)]);
    const Eigen::VectorXd actual = rig.kernels[static_cast<std::size_t>(i)]->inputs();
    EXPECT_TRUE(actual.isApprox(expected, kOptimumTol))
      << "agent " << i << " final input sequence mismatch";
  }
}

}  // namespace
}  // namespace cpp_admm

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
