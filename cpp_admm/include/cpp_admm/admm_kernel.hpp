// Production consensus-ADMM kernel: transport-agnostic, allocation-free in steady state.
//
// This is the piece that gets dropped into `transition-viable-swarm`. It must therefore
// depend on neither ROS nor ZeroMQ at the type level -- both are supplied through the
// ITransport interface below. `consensus_node.cpp` provides the ROS 2 implementation;
// `ZeroMqTransport` (declared here, implemented in admm_kernel.cpp) provides a standalone
// one for benchmarks that must run without a ROS graph.
//
// Real-time contract
// ------------------
//   * No heap allocation inside `iterate()`. Everything is sized in `configure()`.
//   * No unbounded blocking: `poll()` takes an explicit timeout and a missed message is a
//     normal, handled outcome -- never a reason to stall the control loop.
//   * `iterate()` is callable from a real-time thread; `configure()` is not.
//
// Correspondence with the Python reference: the update equations, residual definitions,
// and adaptive-rho rule are identical to `python/distributed_mpc_admm/consensus_admm.py`.
// `test/test_admm_kernel.cpp` pins that equivalence numerically -- if you change the math
// in one place, change it in both.

#ifndef CPP_ADMM__ADMM_KERNEL_HPP_
#define CPP_ADMM__ADMM_KERNEL_HPP_

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "cpp_admm/per_agent_qp.hpp"

namespace cpp_admm
{

/// One inter-agent packet. `payload` is a flattened (horizon * dim) position block in
/// row-major (time-major) order; the flattening convention is shared with the QP layout
/// and must not diverge from it.
struct NeighborMessage
{
  int sender{-1};
  int subject{-1};          ///< Agent whose trajectory this describes.
  int64_t admm_iteration{0};
  int64_t control_step{0};  ///< Rejects packets from a stale MPC step outright.
  Eigen::VectorXd payload;

  /// Serialised size in bytes, for the bandwidth accounting in the analysis notebooks.
  [[nodiscard]] std::size_t byte_size() const noexcept;
};

/// Which phase of the iteration a message belongs to. Kept explicit so a receiver can
/// never mistake a local copy for an agreed consensus value.
enum class MessageKind : uint8_t
{
  kLocalCopy = 0,   ///< y_i^j, sent from contributor i to subject j.
  kConsensus = 1,   ///< z^j, broadcast by subject j to its contributors.
};

/// Transport abstraction. Implementations must be non-blocking apart from the explicit
/// `poll` timeout, and must be safe to call from the same thread that runs `iterate()`.
class ITransport
{
public:
  virtual ~ITransport() = default;

  /// Fire-and-forget send. Returns false if the packet could not be queued (a full queue
  /// is a dropped packet, not an error -- the kernel is required to tolerate it).
  virtual bool publish(MessageKind kind, const NeighborMessage & message) = 0;

  /// Drain everything that has arrived, up to `timeout`. Returns the number appended to
  /// `out`. `out` is cleared first and must have been reserved by the caller.
  virtual std::size_t poll(
    MessageKind kind, std::chrono::microseconds timeout,
    std::vector<NeighborMessage> & out) = 0;

  /// Discard buffered traffic. Called at every control step boundary so that a slow agent
  /// cannot poison the next step with packets from the previous one.
  virtual void flush() = 0;
};

/// Loopback transport for unit tests: delivers to peer kernels in the same process, with
/// optional deterministic loss and delay so the asynchronous paths can be tested without
/// a network.
class InProcessTransport : public ITransport
{
public:
  InProcessTransport(int agent_id, double loss_prob, int max_delay, uint64_t seed);
  ~InProcessTransport() override;

  bool publish(MessageKind kind, const NeighborMessage & message) override;
  std::size_t poll(
    MessageKind kind, std::chrono::microseconds timeout,
    std::vector<NeighborMessage> & out) override;
  void flush() override;

  /// Wire a set of in-process transports together. Must be called before any publish.
  static void connect(const std::vector<InProcessTransport *> & peers);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// ZeroMQ transport: one PUB socket per agent, one SUB socket subscribed to each neighbor.
/// Topics are `"admm/<kind>/<subject>"`. Used for the no-ROS benchmark and as the fallback
/// on platforms where a full ROS 2 stack is not available.
class ZeroMqTransport : public ITransport
{
public:
  /// `endpoints` maps neighbor id -> endpoint string (e.g. "tcp://192.168.1.11:5561").
  ZeroMqTransport(
    int agent_id, std::string bind_endpoint,
    std::unordered_map<int, std::string> endpoints);
  ~ZeroMqTransport() override;

  bool publish(MessageKind kind, const NeighborMessage & message) override;
  std::size_t poll(
    MessageKind kind, std::chrono::microseconds timeout,
    std::vector<NeighborMessage> & out) override;
  void flush() override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;  ///< PIMPL keeps <zmq.hpp> out of this header.
};

/// Static configuration for one agent's kernel.
struct AgentConfig
{
  int agent_id{0};
  int n_agents{4};
  std::vector<int> neighbors;   ///< Open neighborhood, sorted ascending.
  int horizon{15};
  int dim{2};
  double dt{0.1};

  double q_position{1.0};
  double q_velocity{0.1};
  double r_input{0.05};
  double r_rate{0.0};
  double p_terminal{5.0};
  double w_formation{10.0};

  double u_max{3.0};
  double v_max{2.0};

  /// Desired relative offsets d_ij = p_i - p_j, keyed by neighbor id.
  std::unordered_map<int, Eigen::VectorXd> offsets;

  /// Closed neighborhood (neighbors plus self), sorted. Defines block ordering everywhere.
  [[nodiscard]] std::vector<int> closed_neighborhood() const;

  /// Throws std::invalid_argument on inconsistent sizes, duplicate or self neighbors,
  /// negative weights, or an offset key that is not a neighbor.
  void validate() const;
};

/// ADMM tuning. Mirrors ADMMOptions in the Python reference.
struct ADMMOptions
{
  double rho{1.0};
  int max_iterations{50};       ///< Deliberately small: this is a real-time budget, not a
                                ///< convergence target. See docs/README_math.md.
  double eps_abs{1e-4};
  double eps_rel{1e-3};
  double alpha{1.6};            ///< Over-relaxation.
  bool adaptive_rho{false};
  double mu{10.0};
  double tau{2.0};
  double rho_min{1e-4};
  double rho_max{1e4};
  bool warm_start{true};
  int check_every{1};
  std::chrono::microseconds poll_timeout{2000};
  /// Iterations an agent may run on stale neighbor data before it gives up and holds the
  /// previous control input. Zero disables the check (unsafe on real hardware).
  int max_staleness{5};
};

/// Per-iteration diagnostics, held in preallocated ring buffers.
struct ADMMStats
{
  int iterations{0};
  bool converged{false};
  double primal_residual{0.0};
  double dual_residual{0.0};
  double rho{1.0};
  double solve_time_ms{0.0};
  double qp_time_ms{0.0};        ///< Time inside OSQP only.
  double comm_time_ms{0.0};      ///< Time inside publish/poll only.
  int messages_sent{0};
  int messages_received{0};
  int messages_missing{0};       ///< Expected but not received this iteration.
  int max_staleness_seen{0};

  void reset() noexcept;
};

/// One agent's consensus-ADMM kernel.
///
/// Lifecycle: `AdmmKernel k{cfg, opts, transport}; k.configure(); ` then per control step
/// `k.setInitialState(x); k.setReference(ref); k.solve(); k.firstInput();`
class AdmmKernel
{
public:
  AdmmKernel(AgentConfig config, ADMMOptions options, ITransport * transport);
  ~AdmmKernel();

  AdmmKernel(const AdmmKernel &) = delete;
  AdmmKernel & operator=(const AdmmKernel &) = delete;

  /// Allocate every buffer, build the prediction matrices, and set up the OSQP workspace.
  /// Must be called once before any solve. Throws on invalid configuration.
  void configure();

  /// Current measured state, size 2*dim. Cheap; does not touch the QP structure.
  void setInitialState(const Eigen::VectorXd & x0);

  /// Position reference over the horizon, size horizon*dim. Pass an empty vector for
  /// pure followers.
  void setReference(const Eigen::VectorXd & reference);

  /// Update the formation offsets mid-run (a morph event). Rebuilds only the linear term
  /// of the QP, never the sparsity pattern.
  void setOffsets(const std::unordered_map<int, Eigen::VectorXd> & offsets);

  /// Rebuild for a changed neighborhood (a split or merge event). This *does* change the
  /// number of decision variables, so it reallocates and re-setups OSQP -- never call it
  /// from a real-time thread. Warm-start state for surviving neighbors is preserved.
  void setNeighbors(const std::vector<int> & neighbors);

  /// One full ADMM iteration: x-update, relaxation, exchange, z-update, broadcast,
  /// dual update, residuals. Returns true if the convergence test passed.
  bool iterate();

  /// Run `iterate()` until convergence or the iteration cap. Returns the stats block.
  const ADMMStats & solve();

  /// Time-shift (y, z, lambda) by one control step for warm starting. Called by the node
  /// after applying the input, not by `solve()`.
  void shiftWarmStart();

  /// Zero all iterates and flush the transport.
  void reset();

  /// First input of the optimal sequence, size dim. Valid after `solve()`.
  [[nodiscard]] Eigen::VectorXd firstInput() const;

  /// Full optimal input sequence, size horizon*dim.
  [[nodiscard]] const Eigen::VectorXd & inputs() const noexcept;

  /// This agent's agreed position trajectory z^i, size horizon*dim.
  [[nodiscard]] const Eigen::VectorXd & consensusTrajectory() const noexcept;

  [[nodiscard]] const ADMMStats & stats() const noexcept;
  [[nodiscard]] const AgentConfig & config() const noexcept;

  /// Advance the control-step counter used to reject stale packets.
  void setControlStep(int64_t step) noexcept;

private:
  // --- ADMM phases ---------------------------------------------------------------
  void xUpdate();               ///< Solve the local QP; fills y_.
  void relax();                 ///< y_hat = alpha*y + (1-alpha)*z.
  void exchangeLocalCopies();   ///< Publish y_hat_i^j to each j; collect y_hat_k^i from
                                ///< every contributor k. Missing entries are excluded from
                                ///< the average, never replaced by stale values.
  void zUpdate();               ///< Average over received contributions -> z^i.
  void broadcastConsensus();    ///< Publish z^i; collect z^j from each neighbor j. Here a
                                ///< missing value *does* fall back to the last known one.
  void dualUpdate();            ///< lambda += y_hat - z.
  void computeResiduals();
  void updateRho();             ///< Residual balancing; rescales lambda by 1/factor.

  // --- helpers -------------------------------------------------------------------
  /// Index of neighbor `j` within the closed-neighborhood block ordering.
  [[nodiscard]] int blockIndex(int j) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_admm

#endif  // CPP_ADMM__ADMM_KERNEL_HPP_
