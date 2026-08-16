// Copyright (c) 2026, Ali-Eimaan. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// OSQP-backed per-agent QP. See per_agent_qp.hpp for the variable and constraint layout.

#include "cpp_admm/per_agent_qp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <osqp.h>

#include "cpp_admm/admm_kernel.hpp"  // AgentConfig

namespace cpp_admm
{

namespace
{

/// Repeat a per-timestep block `n` times, producing a (n * v.size()) vector.
Eigen::VectorXd tile_vec(const Eigen::VectorXd & v, int n)
{
  Eigen::VectorXd out(v.size() * n);
  for (int i = 0; i < n; ++i) {
    out.segment(i * v.size(), v.size()) = v;
  }
  return out;
}

/// Rebuild q_solve from the structural linear term and the consensus penalty.
void apply_consensus(
  Eigen::VectorXd & q_solve, const Eigen::VectorXd & q_base,
  const std::vector<Eigen::VectorXd> & z, const std::vector<Eigen::VectorXd> & lam,
  double rho, int block_size, int n_blocks)
{
  q_solve = q_base;
  for (int b = 0; b < n_blocks; ++b) {
    // theta = [U][y^{c_0}]...[y^{c_{M-1}}]; y-block b starts at (b + 1) * block_size.
    q_solve.segment((b + 1) * block_size, block_size).noalias() -= rho * (z[b] - lam[b]);
  }
}

/// Eigen stores sparse indices as 32-bit `int`, but this OSQP build uses the 64-bit
/// `OSQPInt` (`DLONG`). Convert a compressed Eigen matrix into OSQP CSC arrays. OSQP
/// copies these inside `osqp_setup`, so the returned vectors are only needed for the
/// duration of the setup call.
struct OsqpCsc
{
  std::vector<OSQPInt> p;
  std::vector<OSQPInt> i;
  std::vector<OSQPFloat> x;
};

OsqpCsc to_osqp_csc(const Eigen::SparseMatrix<double> & M)
{
  OsqpCsc out;
  const int cols = static_cast<int>(M.cols());
  out.p.reserve(static_cast<std::size_t>(cols) + 1);
  for (int c = 0; c <= cols; ++c) {
    out.p.push_back(static_cast<OSQPInt>(M.outerIndexPtr()[c]));
  }
  const int nnz = static_cast<int>(M.nonZeros());
  out.i.reserve(static_cast<std::size_t>(nnz));
  out.x.reserve(static_cast<std::size_t>(nnz));
  for (int k = 0; k < nnz; ++k) {
    out.i.push_back(static_cast<OSQPInt>(M.innerIndexPtr()[k]));
    out.x.push_back(static_cast<OSQPFloat>(M.valuePtr()[k]));
  }
  return out;
}

QpStatus map_status(OSQPInt value)
{
  switch (value) {
    case OSQP_SOLVED: return QpStatus::kSolved;
    case OSQP_SOLVED_INACCURATE: return QpStatus::kSolvedInaccurate;
    case OSQP_MAX_ITER_REACHED: return QpStatus::kMaxIterReached;
    case OSQP_TIME_LIMIT_REACHED: return QpStatus::kTimeLimitReached;
    case OSQP_PRIMAL_INFEASIBLE:
    case OSQP_PRIMAL_INFEASIBLE_INACCURATE: return QpStatus::kPrimalInfeasible;
    case OSQP_DUAL_INFEASIBLE:
    case OSQP_DUAL_INFEASIBLE_INACCURATE: return QpStatus::kDualInfeasible;
    default: return QpStatus::kError;
  }
}

}  // namespace

const char * toString(QpStatus status)
{
  switch (status) {
    case QpStatus::kSolved: return "solved";
    case QpStatus::kSolvedInaccurate: return "solved_inaccurate";
    case QpStatus::kMaxIterReached: return "max_iter_reached";
    case QpStatus::kTimeLimitReached: return "time_limit_reached";
    case QpStatus::kPrimalInfeasible: return "primal_infeasible";
    case QpStatus::kDualInfeasible: return "dual_infeasible";
    case QpStatus::kError: return "error";
  }
  return "unknown";
}

bool QpSolution::ok() const noexcept
{
  return status == QpStatus::kSolved || status == QpStatus::kSolvedInaccurate;
}

// ------------------------------------------------------------------ DoubleIntegrator

DoubleIntegrator::DoubleIntegrator(double dt, int dim)
: dt_(dt), dim_(dim)
{
  const int n = 2 * dim;
  a_ = Eigen::MatrixXd::Zero(n, n);
  b_ = Eigen::MatrixXd::Zero(n, dim);

  a_.topLeftCorner(dim, dim).setIdentity();
  a_.topRightCorner(dim, dim).setIdentity();
  a_.topRightCorner(dim, dim) *= dt;
  a_.bottomRightCorner(dim, dim).setIdentity();

  b_.topRows(dim).setIdentity();
  b_.topRows(dim) *= 0.5 * dt * dt;
  b_.bottomRows(dim).setIdentity();
  b_.bottomRows(dim) *= dt;
}

int DoubleIntegrator::nStates() const noexcept {return 2 * dim_;}
int DoubleIntegrator::nInputs() const noexcept {return dim_;}

const Eigen::MatrixXd & DoubleIntegrator::A() const noexcept {return a_;}
const Eigen::MatrixXd & DoubleIntegrator::B() const noexcept {return b_;}

void DoubleIntegrator::predictionMatrices(
  int horizon, Eigen::MatrixXd & phi,
  Eigen::MatrixXd & gamma) const
{
  const int n = 2 * dim_;
  const int m = dim_;
  const int B = horizon * m;

  phi.resize(horizon * n, n);
  gamma = Eigen::MatrixXd::Zero(horizon * n, B);

  // powers[0] = I, powers[k] = A^k.
  std::vector<Eigen::MatrixXd> powers(static_cast<std::size_t>(horizon) + 1);
  powers[0] = Eigen::MatrixXd::Identity(n, n);
  for (int k = 1; k <= horizon; ++k) {
    powers[static_cast<std::size_t>(k)] = a_ * powers[static_cast<std::size_t>(k) - 1];
  }

  for (int r = 0; r < horizon; ++r) {
    phi.block(r * n, 0, n, n) = powers[static_cast<std::size_t>(r) + 1];
    for (int s = 0; s <= r; ++s) {
      gamma.block(r * n, s * m, n, m) = powers[static_cast<std::size_t>(r - s)] * b_;
    }
  }
}

void DoubleIntegrator::positionPredictionMatrices(
  int horizon, Eigen::MatrixXd & phi_p, Eigen::MatrixXd & gamma_p) const
{
  Eigen::MatrixXd phi, gamma;
  predictionMatrices(horizon, phi, gamma);

  const int n = 2 * dim_;
  phi_p.resize(horizon * dim_, n);
  gamma_p.resize(horizon * dim_, horizon * dim_);
  for (int t = 0; t < horizon; ++t) {
    phi_p.middleRows(t * dim_, dim_) = phi.middleRows(t * n, dim_);
    gamma_p.middleRows(t * dim_, dim_) = gamma.middleRows(t * n, dim_);
  }
}

void DoubleIntegrator::velocityPredictionMatrices(
  int horizon, Eigen::MatrixXd & phi_v, Eigen::MatrixXd & gamma_v) const
{
  Eigen::MatrixXd phi, gamma;
  predictionMatrices(horizon, phi, gamma);

  const int n = 2 * dim_;
  phi_v.resize(horizon * dim_, n);
  gamma_v.resize(horizon * dim_, horizon * dim_);
  for (int t = 0; t < horizon; ++t) {
    phi_v.middleRows(t * dim_, dim_) = phi.middleRows(t * n + dim_, dim_);
    gamma_v.middleRows(t * dim_, dim_) = gamma.middleRows(t * n + dim_, dim_);
  }
}

// ------------------------------------------------------------------------ PerAgentQp

struct PerAgentQp::Impl
{
  AgentConfig config;
  QpSettings settings;
  DoubleIntegrator model;

  int horizon{0};
  int dim{0};
  int block_size{0};  // horizon * dim
  int n{0};           // (1 + M) * block_size
  int m{0};           // 3 * block_size

  std::vector<int> closed_nbhd;              // sorted closed neighborhood
  std::unordered_map<int, int> block_offset;  // agent id -> block index

  Eigen::MatrixXd phi_p, gamma_p, phi_v, gamma_v;
  Eigen::MatrixXd qU_per_x0;  // 2*q_velocity * gamma_v' * phi_v

  // QP data. P is upper-triangular CSC, A is CSC; both built once.
  Eigen::SparseMatrix<double> P, A;
  Eigen::VectorXd q_vel, q_ref, q_form;  // additive parts of the structural linear term
  Eigen::VectorXd q_base;                // q_vel + q_ref + q_form
  Eigen::VectorXd q_solve;               // q_base - rho*(z - lam), passed to OSQP
  Eigen::VectorXd l, u;                  // constraint bounds
  Eigen::VectorXd ones_B;

  std::vector<OSQPInt> rho_diag_indices;   // positions of rho*I entries in P.valuePtr()
  std::vector<OSQPFloat> rho_diag_base;    // P diagonal minus rho
  std::vector<OSQPFloat> rho_values;       // scratch for osqp_update_data_mat
  double rho_p{1.0};                        // rho currently baked into P

  // Consensus cache, keyed by block index.
  std::vector<Eigen::VectorXd> z_blocks;
  std::vector<Eigen::VectorXd> lam_blocks;
  double rho_q{1.0};

  // Current state/reference/offset caches.
  Eigen::VectorXd x0;
  Eigen::VectorXd reference;
  bool has_reference{false};
  std::unordered_map<int, Eigen::VectorXd> offsets;

  OSQPSolver * solver{nullptr};
  QpSolution solution;
  bool ready{false};

  Impl(double sample_time, int spatial_dim)
  : model(sample_time, spatial_dim) {}
};

PerAgentQp::PerAgentQp(const AgentConfig & config, QpSettings settings)
{
  impl_ = std::make_unique<Impl>(config.dt, config.dim);
  impl_->config = config;
  impl_->settings = settings;

  impl_->horizon = config.horizon;
  impl_->dim = config.dim;
  impl_->block_size = config.horizon * config.dim;
  impl_->closed_nbhd = config.closed_neighborhood();

  const int M = static_cast<int>(impl_->closed_nbhd.size());
  impl_->n = (1 + M) * impl_->block_size;
  impl_->m = 3 * impl_->block_size;
  for (int b = 0; b < M; ++b) {
    impl_->block_offset[impl_->closed_nbhd[static_cast<std::size_t>(b)]] = b;
  }

  impl_->model.positionPredictionMatrices(config.horizon, impl_->phi_p, impl_->gamma_p);
  impl_->model.velocityPredictionMatrices(config.horizon, impl_->phi_v, impl_->gamma_v);
  impl_->qU_per_x0 = 2.0 * config.q_velocity * impl_->gamma_v.transpose() * impl_->phi_v;

  impl_->x0 = Eigen::VectorXd::Zero(2 * config.dim);
  impl_->reference = Eigen::VectorXd::Zero(impl_->block_size);

  // Normalise offsets so every formation edge has a (possibly zero) offset.
  impl_->offsets = config.offsets;
  for (int j : config.neighbors) {
    if (impl_->offsets.count(j) == 0) {
      impl_->offsets[j] = Eigen::VectorXd::Zero(config.dim);
    }
  }

  impl_->q_vel = Eigen::VectorXd::Zero(impl_->n);
  impl_->q_ref = Eigen::VectorXd::Zero(impl_->n);
  impl_->q_form = Eigen::VectorXd::Zero(impl_->n);
  impl_->q_base = Eigen::VectorXd::Zero(impl_->n);
  impl_->q_solve = Eigen::VectorXd::Zero(impl_->n);
  impl_->l = Eigen::VectorXd::Zero(impl_->m);
  impl_->u = Eigen::VectorXd::Zero(impl_->m);
  impl_->ones_B = Eigen::VectorXd::Ones(impl_->block_size);

  impl_->z_blocks.resize(static_cast<std::size_t>(M));
  impl_->lam_blocks.resize(static_cast<std::size_t>(M));
  for (int b = 0; b < M; ++b) {
    impl_->z_blocks[static_cast<std::size_t>(b)] = Eigen::VectorXd::Zero(impl_->block_size);
    impl_->lam_blocks[static_cast<std::size_t>(b)] = Eigen::VectorXd::Zero(impl_->block_size);
  }
}

PerAgentQp::~PerAgentQp()
{
  if (impl_ && impl_->solver != nullptr) {
    osqp_cleanup(impl_->solver);
  }
}

void PerAgentQp::setup()
{
  if (impl_->ready) {
    throw std::logic_error("PerAgentQp::setup called twice");
  }

  buildHessian();
  buildLinearTerm();
  buildConstraints();

  OSQPSettings os;
  osqp_set_default_settings(&os);
  os.verbose = impl_->settings.verbose ? 1 : 0;
  os.warm_starting = impl_->settings.warm_start ? 1 : 0;
  os.polishing = impl_->settings.polish ? 1 : 0;
  os.adaptive_rho = impl_->settings.adaptive_rho ? 1 : 0;
  os.max_iter = static_cast<OSQPInt>(impl_->settings.max_iter);
  os.eps_abs = static_cast<OSQPFloat>(impl_->settings.eps_abs);
  os.eps_rel = static_cast<OSQPFloat>(impl_->settings.eps_rel);
  // OSQP rejects a non-positive time_limit; 0 means "no time limit" and keeps the
  // default (OSQP_TIME_LIMIT).
  if (impl_->settings.time_limit_s > 0.0) {
    os.time_limit = static_cast<OSQPFloat>(impl_->settings.time_limit_s);
  }
  // Scaling off keeps the value-only P updates byte-for-byte predictable, which the
  // tests rely on (and saves a scaling pass every control step on real hardware).
  os.scaling = 0;

  OSQPCscMatrix P_csc;
  OsqpCsc P = to_osqp_csc(impl_->P);
  P_csc.m = static_cast<OSQPInt>(impl_->n);
  P_csc.n = static_cast<OSQPInt>(impl_->n);
  P_csc.nzmax = static_cast<OSQPInt>(P.x.size());
  P_csc.p = P.p.data();
  P_csc.i = P.i.data();
  P_csc.x = P.x.data();
  P_csc.nz = -1;  // CSC format
  P_csc.owned = 0;

  OSQPCscMatrix A_csc;
  OsqpCsc A = to_osqp_csc(impl_->A);
  A_csc.m = static_cast<OSQPInt>(impl_->m);
  A_csc.n = static_cast<OSQPInt>(impl_->n);
  A_csc.nzmax = static_cast<OSQPInt>(A.x.size());
  A_csc.p = A.p.data();
  A_csc.i = A.i.data();
  A_csc.x = A.x.data();
  A_csc.nz = -1;
  A_csc.owned = 0;

  const OSQPInt rc = osqp_setup(
    &impl_->solver, &P_csc, impl_->q_solve.data(), &A_csc, impl_->l.data(), impl_->u.data(),
    static_cast<OSQPInt>(impl_->m), static_cast<OSQPInt>(impl_->n), &os);
  if (rc != 0) {
    throw std::runtime_error(std::string("osqp_setup failed: ") + osqp_error_message(rc));
  }

  impl_->ready = true;
}

bool PerAgentQp::isReady() const noexcept {return impl_->ready;}

void PerAgentQp::updateInitialState(const Eigen::VectorXd & x0)
{
  impl_->x0 = x0;
  const int B = impl_->block_size;

  // Velocity-damping linear term lives in the U block and depends on x0.
  impl_->q_vel.setZero();
  impl_->q_vel.head(B) = impl_->qU_per_x0 * x0;

  impl_->q_base = impl_->q_vel + impl_->q_ref + impl_->q_form;
  apply_consensus(
    impl_->q_solve, impl_->q_base, impl_->z_blocks, impl_->lam_blocks, impl_->rho_q, B,
    static_cast<int>(impl_->closed_nbhd.size()));

  const Eigen::VectorXd phi_p_x0 = impl_->phi_p * x0;
  const Eigen::VectorXd phi_v_x0 = impl_->phi_v * x0;
  impl_->l.head(B) = phi_p_x0;
  impl_->u.head(B) = phi_p_x0;
  impl_->l.segment(2 * B, B) = -impl_->config.v_max * impl_->ones_B - phi_v_x0;
  impl_->u.segment(2 * B, B) = impl_->config.v_max * impl_->ones_B - phi_v_x0;

  osqp_update_data_vec(impl_->solver, impl_->q_solve.data(), impl_->l.data(), impl_->u.data());
}

void PerAgentQp::updateReference(const Eigen::VectorXd & reference)
{
  const int B = impl_->block_size;
  const int self_block = impl_->block_offset.at(impl_->config.agent_id);

  impl_->has_reference = reference.size() != 0;
  impl_->q_ref.setZero();
  if (impl_->has_reference) {
    impl_->reference = reference;
    const int tail = impl_->horizon - 1;
    Eigen::VectorXd qr = -2.0 * impl_->config.q_position * reference;
    qr.segment(tail * impl_->dim, impl_->dim) -=
      2.0 * impl_->config.p_terminal * reference.segment(tail * impl_->dim, impl_->dim);
    impl_->q_ref.segment((self_block + 1) * B, B) = qr;
  } else {
    impl_->reference.setZero();
  }

  impl_->q_base = impl_->q_vel + impl_->q_ref + impl_->q_form;
  apply_consensus(
    impl_->q_solve, impl_->q_base, impl_->z_blocks, impl_->lam_blocks, impl_->rho_q, B,
    static_cast<int>(impl_->closed_nbhd.size()));
  osqp_update_data_vec(impl_->solver, impl_->q_solve.data(), nullptr, nullptr);
}

void PerAgentQp::updateOffsets(const std::unordered_map<int, Eigen::VectorXd> & offsets)
{
  const int B = impl_->block_size;
  const int self_block = impl_->block_offset.at(impl_->config.agent_id);
  const double w = impl_->config.w_formation;

  impl_->offsets = offsets;
  for (int j : impl_->config.neighbors) {
    if (impl_->offsets.count(j) == 0) {
      impl_->offsets[j] = Eigen::VectorXd::Zero(impl_->dim);
    }
  }

  impl_->q_form.setZero();
  for (int j : impl_->config.neighbors) {
    const Eigen::VectorXd d_tiled = tile_vec(impl_->offsets.at(j), impl_->horizon);
    const int bj = impl_->block_offset.at(j);
    impl_->q_form.segment((self_block + 1) * B, B) -= 2.0 * w * d_tiled;
    impl_->q_form.segment((bj + 1) * B, B) += 2.0 * w * d_tiled;
  }

  impl_->q_base = impl_->q_vel + impl_->q_ref + impl_->q_form;
  apply_consensus(
    impl_->q_solve, impl_->q_base, impl_->z_blocks, impl_->lam_blocks, impl_->rho_q, B,
    static_cast<int>(impl_->closed_nbhd.size()));
  osqp_update_data_vec(impl_->solver, impl_->q_solve.data(), nullptr, nullptr);
}

void PerAgentQp::updateConsensus(
  const std::unordered_map<int, Eigen::VectorXd> & z,
  const std::unordered_map<int, Eigen::VectorXd> & lam, double rho)
{
  const int B = impl_->block_size;
  const int M = static_cast<int>(impl_->closed_nbhd.size());
  for (int b = 0; b < M; ++b) {
    const int j = impl_->closed_nbhd[static_cast<std::size_t>(b)];
    const auto zit = z.find(j);
    const auto lit = lam.find(j);
    if (zit == z.end() || lit == lam.end()) {
      throw std::invalid_argument("updateConsensus missing z/lam for agent " + std::to_string(j));
    }
    impl_->z_blocks[static_cast<std::size_t>(b)] = zit->second;
    impl_->lam_blocks[static_cast<std::size_t>(b)] = lit->second;
  }
  impl_->rho_q = rho;

  apply_consensus(impl_->q_solve, impl_->q_base, impl_->z_blocks, impl_->lam_blocks, rho, B, M);
  osqp_update_data_vec(impl_->solver, impl_->q_solve.data(), nullptr, nullptr);
}

void PerAgentQp::updateRho(double rho)
{
  if (rho == impl_->rho_p) {
    return;
  }

  const std::size_t k = impl_->rho_diag_indices.size();
  for (std::size_t i = 0; i < k; ++i) {
    impl_->rho_values[i] = impl_->rho_diag_base[i] + rho;
    impl_->P.valuePtr()[impl_->rho_diag_indices[i]] = impl_->rho_values[i];
  }

  if (k > 0) {
    osqp_update_data_mat(
      impl_->solver, impl_->rho_values.data(), impl_->rho_diag_indices.data(),
      static_cast<OSQPInt>(k), nullptr, nullptr, 0);
  }
  impl_->rho_p = rho;
}

const QpSolution & PerAgentQp::solve()
{
  if (!impl_->ready) {
    throw std::logic_error("PerAgentQp::solve called before setup");
  }

  const OSQPInt rc = osqp_solve(impl_->solver);
  auto & sol = impl_->solution;

  if (sol.theta.size() != impl_->n) {
    sol.theta.resize(impl_->n);
  }
  sol.theta = Eigen::Map<const Eigen::VectorXd>(impl_->solver->solution->x, impl_->n);
  sol.iterations = static_cast<int>(impl_->solver->info->iter);
  sol.solve_time_ms = static_cast<double>(impl_->solver->info->solve_time) * 1000.0;
  sol.objective = static_cast<double>(impl_->solver->info->obj_val);
  sol.status = rc == 0 ? map_status(impl_->solver->info->status_val) : QpStatus::kError;
  return sol;
}

void PerAgentQp::warmStart(const Eigen::VectorXd & theta, const Eigen::VectorXd & dual)
{
  if (!impl_->ready) {
    throw std::logic_error("PerAgentQp::warmStart called before setup");
  }
  if (theta.size() != impl_->n || dual.size() != impl_->m) {
    throw std::invalid_argument("PerAgentQp::warmStart received wrong-sized vectors");
  }
  osqp_warm_start(impl_->solver, theta.data(), dual.data());
}

void PerAgentQp::resetWarmStart()
{
  if (impl_->ready) {
    osqp_cold_start(impl_->solver);
  }
}

int PerAgentQp::numVariables() const noexcept {return impl_->n;}
int PerAgentQp::numConstraints() const noexcept {return impl_->m;}
int PerAgentQp::inputOffset() const noexcept {return 0;}

int PerAgentQp::copyOffset(int agent_j) const
{
  return (impl_->block_offset.at(agent_j) + 1) * impl_->block_size;
}

Eigen::VectorXd PerAgentQp::extractInputs(const Eigen::VectorXd & theta) const
{
  return theta.segment(0, impl_->block_size);
}

Eigen::VectorXd PerAgentQp::extractCopy(const Eigen::VectorXd & theta, int agent_j) const
{
  return theta.segment(copyOffset(agent_j), impl_->block_size);
}

double PerAgentQp::localObjective(const Eigen::VectorXd & theta) const
{
  const auto & im = *impl_;
  const int B = im.block_size;
  const int dim = im.dim;

  const Eigen::VectorXd U = theta.head(B);
  double obj = 0.0;

  if (im.config.r_input != 0.0) {
    obj += im.config.r_input * U.squaredNorm();
  }

  const Eigen::VectorXd velocity = im.phi_v * im.x0 + im.gamma_v * U;
  if (im.config.q_velocity != 0.0) {
    obj += im.config.q_velocity * velocity.squaredNorm();
  }

  if (im.config.r_rate != 0.0) {
    double rate = 0.0;
    for (int t = 0; t < im.horizon - 1; ++t) {
      rate += (U.segment((t + 1) * dim, dim) - U.segment(t * dim, dim)).squaredNorm();
    }
    obj += im.config.r_rate * rate;
  }

  const int self_block = im.block_offset.at(im.config.agent_id);
  const Eigen::VectorXd y_self = theta.segment((self_block + 1) * B, B);

  if (im.has_reference) {
    const Eigen::VectorXd e = y_self - im.reference;
    if (im.config.q_position != 0.0) {
      obj += im.config.q_position * e.squaredNorm();
    }
    if (im.config.p_terminal != 0.0) {
      obj += im.config.p_terminal * e.tail(dim).squaredNorm();
    }
  }

  for (int j : im.config.neighbors) {
    const int bj = im.block_offset.at(j);
    const Eigen::VectorXd y_j = theta.segment((bj + 1) * B, B);
    const Eigen::VectorXd d_tiled = tile_vec(im.offsets.at(j), im.horizon);
    obj += im.config.w_formation * (y_self - y_j - d_tiled).squaredNorm();
  }

  return obj;
}

const Eigen::SparseMatrix<double> & PerAgentQp::hessian() const noexcept
{
  return impl_->P;
}

void PerAgentQp::buildHessian()
{
  const int B = impl_->block_size;
  const int n = impl_->n;
  const int dim = impl_->dim;
  const int horizon = impl_->horizon;
  const AgentConfig & cfg = impl_->config;
  const int M = static_cast<int>(impl_->closed_nbhd.size());
  const int self_block = impl_->block_offset.at(cfg.agent_id);
  const int num_neighbors = static_cast<int>(cfg.neighbors.size());

  Eigen::MatrixXd Pd = Eigen::MatrixXd::Zero(n, n);

  // --- U block ------------------------------------------------------------------
  Eigen::MatrixXd Puu = Eigen::MatrixXd::Zero(B, B);
  if (cfg.r_input != 0.0) {
    Puu.diagonal().array() += 2.0 * cfg.r_input;
  }
  if (cfg.r_rate != 0.0) {
    Eigen::MatrixXd DtD = Eigen::MatrixXd::Zero(B, B);
    for (int t = 0; t < horizon - 1; ++t) {
      for (int d = 0; d < dim; ++d) {
        const int i = t * dim + d;
        const int j = (t + 1) * dim + d;
        DtD(i, i) += 1.0;
        DtD(j, j) += 1.0;
        DtD(i, j) -= 1.0;
        DtD(j, i) -= 1.0;
      }
    }
    Puu += 2.0 * cfg.r_rate * DtD;
  }
  if (cfg.q_velocity != 0.0) {
    Puu += 2.0 * cfg.q_velocity * (impl_->gamma_v.transpose() * impl_->gamma_v);
  }
  Pd.block(0, 0, B, B) = Puu;

  // --- y block diagonals ---------------------------------------------------------
  for (int b = 0; b < M; ++b) {
    const int base = (b + 1) * B;
    double diag = impl_->rho_p;
    if (b == self_block) {
      diag += 2.0 * cfg.q_position;
      diag += 2.0 * cfg.w_formation * static_cast<double>(num_neighbors);
    } else {
      diag += 2.0 * cfg.w_formation;
    }
    for (int k = 0; k < B; ++k) {
      Pd(base + k, base + k) = diag;
    }
    if (b == self_block && cfg.p_terminal != 0.0) {
      for (int d = 0; d < dim; ++d) {
        const int k = (horizon - 1) * dim + d;
        Pd(base + k, base + k) += 2.0 * cfg.p_terminal;
      }
    }
  }

  // --- formation cross terms -----------------------------------------------------
  for (int j : cfg.neighbors) {
    const int bj = impl_->block_offset.at(j);
    for (int k = 0; k < B; ++k) {
      Pd((self_block + 1) * B + k, (bj + 1) * B + k) += -2.0 * cfg.w_formation;
      Pd((bj + 1) * B + k, (self_block + 1) * B + k) += -2.0 * cfg.w_formation;
    }
  }

  // OSQP wants the upper triangle only.
  Eigen::MatrixXd Pu = Pd.triangularView<Eigen::Upper>();
  impl_->P = Pu.sparseView();
  impl_->P.makeCompressed();

  // Record the positions of the rho*I diagonal entries, and their non-rho base.
  impl_->rho_diag_indices.clear();
  impl_->rho_diag_base.clear();
  for (int b = 0; b < M; ++b) {
    for (int k = 0; k < B; ++k) {
      const int c = (b + 1) * B + k;
      for (OSQPInt idx = impl_->P.outerIndexPtr()[c]; idx < impl_->P.outerIndexPtr()[c + 1];
        ++idx)
      {
        if (impl_->P.innerIndexPtr()[idx] == c) {
          impl_->rho_diag_indices.push_back(idx);
          impl_->rho_diag_base.push_back(impl_->P.valuePtr()[idx] - impl_->rho_p);
          break;
        }
      }
    }
  }
  impl_->rho_values.resize(impl_->rho_diag_indices.size());
}

void PerAgentQp::buildLinearTerm()
{
  impl_->x0.setZero();
  impl_->has_reference = false;
  impl_->reference.setZero();
  impl_->q_vel.setZero();
  impl_->q_ref.setZero();

  const int B = impl_->block_size;
  const int self_block = impl_->block_offset.at(impl_->config.agent_id);
  const double w = impl_->config.w_formation;

  impl_->q_form.setZero();
  for (int j : impl_->config.neighbors) {
    const Eigen::VectorXd d_tiled = tile_vec(impl_->offsets.at(j), impl_->horizon);
    const int bj = impl_->block_offset.at(j);
    impl_->q_form.segment((self_block + 1) * B, B) -= 2.0 * w * d_tiled;
    impl_->q_form.segment((bj + 1) * B, B) += 2.0 * w * d_tiled;
  }

  impl_->q_base = impl_->q_form;
  impl_->q_solve = impl_->q_base;
}

void PerAgentQp::buildConstraints()
{
  const int B = impl_->block_size;
  const int n = impl_->n;
  const int m = impl_->m;
  const int self_offset = (impl_->block_offset.at(impl_->config.agent_id) + 1) * B;

  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(m, n);
  // Dynamics equality: y_self - Gamma_p U = Phi_p x0.
  Ad.block(0, 0, B, B) = -impl_->gamma_p;
  Ad.block(0, self_offset, B, B) = Eigen::MatrixXd::Identity(B, B);
  // Input box.
  Ad.block(B, 0, B, B) = Eigen::MatrixXd::Identity(B, B);
  // Velocity box.
  Ad.block(2 * B, 0, B, B) = impl_->gamma_v;

  impl_->A = Ad.sparseView();
  impl_->A.makeCompressed();

  impl_->l.setZero();
  impl_->u.setZero();
  impl_->l.segment(B, B) = -impl_->config.u_max * impl_->ones_B;
  impl_->u.segment(B, B) = impl_->config.u_max * impl_->ones_B;
  impl_->l.segment(2 * B, B) = -impl_->config.v_max * impl_->ones_B;
  impl_->u.segment(2 * B, B) = impl_->config.v_max * impl_->ones_B;
}

}  // namespace cpp_admm
