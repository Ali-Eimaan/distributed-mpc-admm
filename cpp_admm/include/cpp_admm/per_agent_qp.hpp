// OSQP wrapper for the per-agent local QP.
//
// Decision vector layout (fixed once at setup, relied on everywhere)
// ------------------------------------------------------------------
//   theta = [ U                     ]  horizon*dim      inputs, time-major
//           [ y^{c_0}               ]  horizon*dim      local copies, one block per
//           [ y^{c_1}               ]                   entry of closed_neighborhood(),
//           [ ...                   ]                   in ascending agent-id order
//           [ y^{c_{M-1}}           ]
//
// with M = |closed neighborhood|. Total size n = (1 + M) * horizon * dim.
// Within a block, ordering is time-major: index (t*dim + d).
//
// Constraint rows, in order:
//   1. own-dynamics equality   y^{self} - Gamma_p U = Phi_p x0        (horizon*dim rows)
//   2. input box               -u_max <= U <= u_max                   (horizon*dim rows)
//   3. velocity box            -v_max <= Cv(Phi x0 + Gamma U) <= v_max (horizon*dim rows)
//
// Only the *values* of the bounds and of `q` change between solves; P and A keep a fixed
// sparsity pattern for the lifetime of the object. That is the entire reason this class
// exists -- see the note on updates below.
//
// Why the consensus penalty does not add constraints
// --------------------------------------------------
// The term (rho/2)||y^j - z^j + lam^j||^2 is quadratic in y with Hessian rho*I and linear
// term -rho*(z^j - lam^j). So a change in rho touches the diagonal of P, and a change in
// (z, lam) touches q only. Both are value-only updates on an unchanged pattern.

#ifndef CPP_ADMM__PER_AGENT_QP_HPP_
#define CPP_ADMM__PER_AGENT_QP_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cpp_admm
{

struct AgentConfig;  // defined in admm_kernel.hpp

/// Settings passed through to OSQP. Defaults are tuned for a warm-started MPC inner loop:
/// loose-ish tolerances, polishing off, adaptive rho off (the *outer* ADMM already adapts
/// its own rho and letting OSQP adapt too makes the iteration counts uninterpretable).
struct QpSettings
{
  double eps_abs{1e-5};
  double eps_rel{1e-5};
  int max_iter{4000};
  bool warm_start{true};
  bool polish{false};
  bool adaptive_rho{false};
  bool verbose{false};
  double time_limit_s{0.0};  ///< 0 disables. Set it on hardware; an unbounded inner solve
                             ///< is how a distributed controller misses its deadline.
};

enum class QpStatus : uint8_t
{
  kSolved = 0,
  kSolvedInaccurate,
  kMaxIterReached,
  kTimeLimitReached,
  kPrimalInfeasible,
  kDualInfeasible,
  kError,
};

/// Human-readable status, for log messages.
const char * toString(QpStatus status);

struct QpSolution
{
  Eigen::VectorXd theta;    ///< Full primal solution in the layout documented above.
  double objective{0.0};    ///< Includes the consensus penalty.
  int iterations{0};
  double solve_time_ms{0.0};
  QpStatus status{QpStatus::kError};

  [[nodiscard]] bool ok() const noexcept;
};

/// Discrete double integrator and its condensed prediction matrices.
/// Mirrors `DoubleIntegrator` in the Python reference; `test_admm_kernel.cpp` checks the
/// two agree elementwise.
class DoubleIntegrator
{
public:
  DoubleIntegrator(double dt, int dim);

  [[nodiscard]] int nStates() const noexcept;
  [[nodiscard]] int nInputs() const noexcept;

  [[nodiscard]] const Eigen::MatrixXd & A() const noexcept;
  [[nodiscard]] const Eigen::MatrixXd & B() const noexcept;

  /// X = Phi*x0 + Gamma*U over t = 1..horizon. Shapes (H*n, n) and (H*n, H*m).
  void predictionMatrices(int horizon, Eigen::MatrixXd & phi, Eigen::MatrixXd & gamma) const;

  /// Position rows only. Shapes (H*dim, n) and (H*dim, H*dim).
  void positionPredictionMatrices(
    int horizon, Eigen::MatrixXd & phi_p, Eigen::MatrixXd & gamma_p) const;

  /// Velocity rows only, needed for the velocity box constraint.
  void velocityPredictionMatrices(
    int horizon, Eigen::MatrixXd & phi_v, Eigen::MatrixXd & gamma_v) const;

private:
  double dt_;
  int dim_;
  Eigen::MatrixXd a_;
  Eigen::MatrixXd b_;
};

/// The per-agent QP, backed by an OSQP workspace that is set up exactly once.
class PerAgentQp
{
public:
  PerAgentQp(const AgentConfig & config, QpSettings settings);
  ~PerAgentQp();

  PerAgentQp(const PerAgentQp &) = delete;
  PerAgentQp & operator=(const PerAgentQp &) = delete;

  /// Build P, q, A, l, u and call osqp_setup. Allocates. Throws std::runtime_error if
  /// OSQP reports a setup failure (most often a non-PSD P from a negative weight).
  void setup();

  /// True once `setup()` has succeeded.
  [[nodiscard]] bool isReady() const noexcept;

  // --- value-only updates; none of these change the sparsity pattern ---------------

  /// Update the dynamics equality right-hand side and the velocity-box rows.
  /// Calls osqp_update_data_vec with the bound vectors only.
  void updateInitialState(const Eigen::VectorXd & x0);

  /// Update the tracking part of `q`. Empty reference means no tracking term.
  void updateReference(const Eigen::VectorXd & reference);

  /// Update the formation part of `q` after a morph event.
  void updateOffsets(const std::unordered_map<int, Eigen::VectorXd> & offsets);

  /// Update the consensus part of `q` for the current ADMM iteration.
  /// `z` and `lam` are keyed by neighbor id and each of size horizon*dim.
  void updateConsensus(
    const std::unordered_map<int, Eigen::VectorXd> & z,
    const std::unordered_map<int, Eigen::VectorXd> & lam,
    double rho);

  /// Update the rho-dependent diagonal of P. Uses osqp_update_data_mat with a fixed index
  /// list computed at setup, so it stays a value-only update. Only call it when rho
  /// actually changed -- a P update triggers an OSQP refactorisation and is roughly as
  /// expensive as a solve.
  void updateRho(double rho);

  /// Solve with the current data. Warm starts from the previous solution automatically.
  const QpSolution & solve();

  /// Explicit warm start, used after a shift or a neighborhood change.
  void warmStart(const Eigen::VectorXd & theta, const Eigen::VectorXd & dual);

  /// Drop warm-start state (e.g. after a large jump in x0).
  void resetWarmStart();

  // --- layout helpers; every caller must go through these rather than recompute -----

  [[nodiscard]] int numVariables() const noexcept;
  [[nodiscard]] int numConstraints() const noexcept;

  /// Offset of the U block (always 0) and of the y block for agent `j`.
  [[nodiscard]] int inputOffset() const noexcept;
  [[nodiscard]] int copyOffset(int agent_j) const;

  /// Extract sub-blocks of the last solution.
  [[nodiscard]] Eigen::VectorXd extractInputs(const Eigen::VectorXd & theta) const;
  [[nodiscard]] Eigen::VectorXd extractCopy(const Eigen::VectorXd & theta, int agent_j) const;

  /// Value of the local objective *excluding* the consensus penalty, for logging.
  [[nodiscard]] double localObjective(const Eigen::VectorXd & theta) const;

  /// Read-only view of the assembled Hessian (upper-triangular CSC). Exposed for the
  /// unit tests that verify the PSD structure and that updateRho/updateConsensus touch
  /// exactly the entries they are documented to touch.
  [[nodiscard]] const Eigen::SparseMatrix<double> & hessian() const noexcept;

private:
  void buildHessian();       ///< P: tracking + effort + formation + rho*I on the y blocks.
  void buildLinearTerm();    ///< q: reference, formation offsets, consensus (z - lam).
  void buildConstraints();   ///< A, l, u in the row order documented at the top.

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_admm

#endif  // CPP_ADMM__PER_AGENT_QP_HPP_
