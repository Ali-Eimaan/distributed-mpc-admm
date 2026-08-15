// Copyright (c) 2026, Ali-Eimaan. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// ROS 2 node wrapping one agent's AdmmKernel. See consensus_node.hpp for topics and QoS.

#include "cpp_admm/consensus_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/executors/single_threaded_executor.hpp>

namespace cpp_admm
{

namespace
{

constexpr char kCopyPrefix[] = "/admm/copy/";
constexpr char kConsensusPrefix[] = "/admm/consensus/";
constexpr char kGraphTopic[] = "/admm/graph";

/// ADMM traffic: best-effort, volatile, depth 1 (see consensus_node.hpp "QoS").
rclcpp::QoS admm_qos()
{
  return rclcpp::QoS(1).best_effort().durability_volatile();
}

std::string join(const std::vector<int> & values)
{
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << "]";
  return os.str();
}

/// Serialise a packet: payload in `data`, freshness/metadata in four labeled dims
/// (sender, subject, admm_iteration, control_step). Iteration/step are int64 but the
/// message schema only has uint32 dim sizes; the demo's counters never leave uint32 range.
void pack_message(const NeighborMessage & message, std_msgs::msg::Float64MultiArray & msg)
{
  msg.data.assign(message.payload.data(), message.payload.data() + message.payload.size());

  msg.layout.dim.resize(4);
  const char * labels[4] = {"sender", "subject", "admm_iteration", "control_step"};
  const std::uint32_t sizes[4] = {
    static_cast<std::uint32_t>(message.sender),
    static_cast<std::uint32_t>(message.subject),
    static_cast<std::uint32_t>(message.admm_iteration),
    static_cast<std::uint32_t>(message.control_step),
  };
  for (int i = 0; i < 4; ++i) {
    msg.layout.dim[static_cast<std::size_t>(i)].label = labels[i];
    msg.layout.dim[static_cast<std::size_t>(i)].size = sizes[i];
    msg.layout.dim[static_cast<std::size_t>(i)].stride = 0;
  }
}

bool unpack_message(const std_msgs::msg::Float64MultiArray & msg, NeighborMessage & out)
{
  if (msg.layout.dim.size() < 4) {
    return false;
  }
  out.sender = static_cast<int>(msg.layout.dim[0].size);
  out.subject = static_cast<int>(msg.layout.dim[1].size);
  out.admm_iteration = static_cast<std::int64_t>(msg.layout.dim[2].size);
  out.control_step = static_cast<std::int64_t>(msg.layout.dim[3].size);

  out.payload.resize(static_cast<Eigen::Index>(msg.data.size()));
  for (std::size_t i = 0; i < msg.data.size(); ++i) {
    out.payload[static_cast<Eigen::Index>(i)] = msg.data[i];
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------- RosTransport

struct RosTransport::Impl
{
  rclcpp::Node * node{nullptr};
  int agent_id{0};
  std::vector<int> neighbors;

  // The one executor, owned by ConsensusNode and re-entered by poll().
  rclcpp::executors::SingleThreadedExecutor * executor{nullptr};

  std::unordered_map<int, rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr>
  copy_pubs;    // one per neighbor
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr consensus_pub;  // own z

  std::unordered_map<int, rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr>
  copy_subs;        // one subscription, /admm/copy/<self>
  std::unordered_map<int, rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr>
  consensus_subs;    // one per neighbor, /admm/consensus/<j>

  std::mutex inbox_mutex;
  std::unordered_map<int, NeighborMessage> latest_copy;      // keyed by sender
  std::unordered_map<int, NeighborMessage> latest_consensus;  // keyed by subject
};

RosTransport::RosTransport(
  rclcpp::Node * node, int agent_id, const std::vector<int> & neighbors,
  rclcpp::executors::SingleThreadedExecutor * executor)
: impl_(std::make_unique<Impl>())
{
  impl_->node = node;
  impl_->agent_id = agent_id;
  impl_->executor = executor;

  // Subscribe to the copies addressed to us (subject == self); one fixed topic.
  const rclcpp::QoS qos = admm_qos();
  impl_->copy_subs[agent_id] = node->create_subscription<std_msgs::msg::Float64MultiArray>(
    std::string(kCopyPrefix) + std::to_string(agent_id), qos,
    [this, agent_id](std_msgs::msg::Float64MultiArray::SharedPtr msg) {
      this->onCopy(agent_id, msg);
    });

  // Our own consensus value (fixed topic; reconfigure does not touch it).
  impl_->consensus_pub = node->create_publisher<std_msgs::msg::Float64MultiArray>(
    std::string(kConsensusPrefix) + std::to_string(agent_id), qos);

  reconfigure(neighbors);
}

RosTransport::~RosTransport() = default;

bool RosTransport::publish(MessageKind kind, const NeighborMessage & message)
{
  std_msgs::msg::Float64MultiArray msg;
  pack_message(message, msg);

  if (kind == MessageKind::kLocalCopy) {
    auto it = impl_->copy_pubs.find(message.subject);
    if (it == impl_->copy_pubs.end()) {
      return false;
    }
    it->second->publish(msg);
    return true;
  }

  if (kind == MessageKind::kConsensus) {
    if (!impl_->consensus_pub) {
      return false;
    }
    impl_->consensus_pub->publish(msg);
    return true;
  }

  return false;
}

std::size_t RosTransport::poll(
  MessageKind kind, std::chrono::microseconds timeout, std::vector<NeighborMessage> & out)
{
  out.clear();

  const auto deadline = std::chrono::steady_clock::now() + timeout;

  // Do NOT sleep: the subscription callbacks only run while the executor is spinning.
  // We re-enter the single shared executor here; a sleeping poll would guarantee an empty
  // inbox and silently degrade the kernel to a fully asynchronous run.
  while (std::chrono::steady_clock::now() < deadline) {
    impl_->executor->spin_some();

    bool complete = true;
    {
      std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
      const auto & inbox =
        (kind == MessageKind::kLocalCopy) ? impl_->latest_copy : impl_->latest_consensus;
      for (int j : impl_->neighbors) {
        if (inbox.find(j) == inbox.end()) {
          complete = false;
          break;
        }
      }
    }

    if (complete) {
      break;
    }

    // spin_some() above returns immediately when there is no work; yield briefly so we
    // do not busy-loop the CPU while waiting for the next packet to arrive.
    rclcpp::sleep_for(std::chrono::microseconds(200));
  }

  // Drain once. Depth-1 semantics: a neighbor that sent twice in one iteration contributes
  // only its latest packet, and everything buffered is consumed by this control step.
  std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
  auto & inbox = (kind == MessageKind::kLocalCopy) ? impl_->latest_copy : impl_->latest_consensus;
  out.reserve(inbox.size());
  for (auto & entry : inbox) {
    out.push_back(std::move(entry.second));
  }
  inbox.clear();
  return out.size();
}

void RosTransport::flush()
{
  std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
  impl_->latest_copy.clear();
  impl_->latest_consensus.clear();
}

void RosTransport::reconfigure(const std::vector<int> & neighbors)
{
  const rclcpp::QoS qos = admm_qos();

  // Rebuild per-neighbor copy publishers (contributing to each neighbor's topic).
  impl_->copy_pubs.clear();
  for (int j : neighbors) {
    impl_->copy_pubs[j] = impl_->node->create_publisher<std_msgs::msg::Float64MultiArray>(
      std::string(kCopyPrefix) + std::to_string(j), qos);
  }

  // Rebuild per-neighbor consensus subscriptions (z^j from each neighbor).
  impl_->consensus_subs.clear();
  for (int j : neighbors) {
    impl_->consensus_subs[j] = impl_->node->create_subscription<std_msgs::msg::Float64MultiArray>(
      std::string(kConsensusPrefix) + std::to_string(j), qos,
      [this, j](std_msgs::msg::Float64MultiArray::SharedPtr msg) {this->onConsensus(j, msg);});
  }

  impl_->neighbors = neighbors;

  // Drop inbox entries for neighbors that are no longer adjacent.
  std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
  for (auto it = impl_->latest_copy.begin(); it != impl_->latest_copy.end(); ) {
    if (std::find(neighbors.begin(), neighbors.end(), it->first) == neighbors.end()) {
      it = impl_->latest_copy.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = impl_->latest_consensus.begin(); it != impl_->latest_consensus.end(); ) {
    if (std::find(neighbors.begin(), neighbors.end(), it->first) == neighbors.end()) {
      it = impl_->latest_consensus.erase(it);
    } else {
      ++it;
    }
  }
}

void RosTransport::onCopy(int, const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  NeighborMessage message;
  if (!unpack_message(*msg, message)) {
    return;
  }

  std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
  // We subscribed to /admm/copy/<self>, so subject must be us, and the contributor must be
  // a current neighbor.
  if (message.subject != impl_->agent_id) {
    return;
  }
  if (std::find(impl_->neighbors.begin(), impl_->neighbors.end(), message.sender) ==
    impl_->neighbors.end())
  {
    return;
  }
  impl_->latest_copy[message.sender] = std::move(message);
}

void RosTransport::onConsensus(int subject, const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  NeighborMessage message;
  if (!unpack_message(*msg, message)) {
    return;
  }

  std::lock_guard<std::mutex> lock(impl_->inbox_mutex);
  // The publisher of /admm/consensus/<j> must be j itself.
  if (message.subject != subject) {
    return;
  }
  if (std::find(impl_->neighbors.begin(), impl_->neighbors.end(), message.subject) ==
    impl_->neighbors.end())
  {
    return;
  }
  impl_->latest_consensus[message.subject] = std::move(message);
}

// --------------------------------------------------------------------- ConsensusNode

struct ConsensusNode::Impl
{
  AgentConfig config;
  ADMMOptions options;
  std::unique_ptr<RosTransport> transport;
  std::unique_ptr<AdmmKernel> kernel;

  // The single executor: created here, handed to RosTransport, spun by spin().
  rclcpp::executors::SingleThreadedExecutor executor;

  rclcpp::Publisher<geometry_msgs::msg::AccelStamped>::SharedPtr cmd_pub;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr diag_pub;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr state_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr reference_sub;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr graph_sub;
  rclcpp::TimerBase::SharedPtr timer;

  std::string reference_topic;
  bool is_leader{false};
  double control_rate_hz{10.0};

  Eigen::VectorXd x0;        // current state, 2*dim
  Eigen::VectorXd reference;  // leader reference over the horizon, horizon*dim
  bool have_state{false};
  bool have_reference{false};
  bool safe_state{false};
  bool need_fresh_state{false};

  int64_t control_step{0};
  bool in_control_step{false};

  // Deferred topology update received mid-solve (setNeighbors is not real-time safe).
  std::vector<int> pending_neighbors;
  bool has_pending_graph{false};

  rclcpp::Time last_state_stamp;
};

ConsensusNode::ConsensusNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("consensus_node", options), impl_(std::make_unique<Impl>())
{
  impl_->config = this->loadConfig();
  impl_->options = this->loadOptions();

  impl_->is_leader = this->declare_parameter<bool>("is_leader", false);
  impl_->reference_topic = this->declare_parameter<std::string>("reference_topic", "");
  impl_->control_rate_hz = this->declare_parameter<double>("control_rate_hz", 10.0);

  if (std::abs(1.0 / impl_->control_rate_hz - impl_->config.dt) > 1e-9) {
    RCLCPP_WARN(
      this->get_logger(),
      "control_rate_hz (%.3f Hz) does not match 1/dt (%.3f Hz); the timer uses "
      "control_rate_hz and the model uses dt",
      impl_->control_rate_hz, 1.0 / impl_->config.dt);
  }

  // Transport and kernel. The transport hands its subscriptions to the same executor we
  // spin in spin(); all node entities exist before add_node() below.
  impl_->transport = std::make_unique<RosTransport>(
    this, impl_->config.agent_id, impl_->config.neighbors, &impl_->executor);
  impl_->kernel = std::make_unique<AdmmKernel>(
    impl_->config, impl_->options, impl_->transport.get());
  impl_->kernel->configure();

  impl_->x0 = Eigen::VectorXd::Zero(2 * impl_->config.dim);
  impl_->reference = Eigen::VectorXd::Zero(impl_->config.horizon * impl_->config.dim);

  // Publishers.
  impl_->cmd_pub =
    this->create_publisher<geometry_msgs::msg::AccelStamped>("~/cmd", rclcpp::QoS(1));
  impl_->diag_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "~/diagnostics", rclcpp::QoS(1));

  // Subscriptions. State/reference are reliable depth-1 (they carry the current estimate,
  // not an iterate); ADMM traffic inside the transport stays best-effort.
  const rclcpp::QoS state_qos = rclcpp::QoS(1);
  impl_->state_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "~/state", state_qos,
    [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {this->onState(msg);});

  if (impl_->is_leader && !impl_->reference_topic.empty()) {
    impl_->reference_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      impl_->reference_topic, state_qos,
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {this->onReference(msg);});
  }

  impl_->graph_sub = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    kGraphTopic, state_qos,
    [this](const std_msgs::msg::Int32MultiArray::SharedPtr msg) {this->onGraphUpdate(msg);});

  // Control timer at control_rate_hz.
  const auto period = std::chrono::milliseconds(
    static_cast<std::int64_t>(std::lround(1000.0 / impl_->control_rate_hz)));
  impl_->timer = this->create_wall_timer(period, [this]() {this->controlStep();});

  // Register the node (and every callback group created above) with our executor.
  impl_->executor.add_node(this->get_node_base_interface());

  RCLCPP_INFO(
    this->get_logger(), "agent %d up: neighbors %s, leader=%s",
    impl_->config.agent_id, join(impl_->config.neighbors).c_str(),
    impl_->is_leader ? "yes" : "no");
}

ConsensusNode::~ConsensusNode() = default;

void ConsensusNode::spin()
{
  impl_->executor.spin();
}

AgentConfig ConsensusNode::loadConfig()
{
  AgentConfig config;

  config.agent_id = static_cast<int>(this->declare_parameter<std::int64_t>("agent_id"));
  config.n_agents = static_cast<int>(this->declare_parameter<std::int64_t>("n_agents"));
  config.dim = 2;

  const std::vector<std::int64_t> neighbors64 =
    this->declare_parameter<std::vector<std::int64_t>>("neighbors", std::vector<std::int64_t>{});
  config.neighbors.clear();
  config.neighbors.reserve(neighbors64.size());
  for (const auto value : neighbors64) {
    config.neighbors.push_back(static_cast<int>(value));
  }

  config.horizon = static_cast<int>(this->declare_parameter<std::int64_t>("horizon", 15));
  config.dt = this->declare_parameter<double>("dt", 0.1);

  config.q_position = this->declare_parameter<double>("q_position", 1.0);
  config.q_velocity = this->declare_parameter<double>("q_velocity", 0.1);
  config.r_input = this->declare_parameter<double>("r_input", 0.05);
  config.r_rate = this->declare_parameter<double>("r_rate", 0.0);
  config.p_terminal = this->declare_parameter<double>("p_terminal", 5.0);
  config.w_formation = this->declare_parameter<double>("w_formation", 10.0);

  config.u_max = this->declare_parameter<double>("u_max", 3.0);
  config.v_max = this->declare_parameter<double>("v_max", 2.0);

  // formation_offsets: flat (neighbor, dx, dy) triples.
  const std::vector<double> flat = this->declare_parameter<std::vector<double>>(
    "formation_offsets", std::vector<double>{});
  if (flat.size() % 3 != 0) {
    throw std::invalid_argument("formation_offsets length must be a multiple of 3");
  }
  for (std::size_t i = 0; i < flat.size(); i += 3) {
    const int neighbor = static_cast<int>(std::lround(flat[i]));
    Eigen::VectorXd offset(2);
    offset << flat[i + 1], flat[i + 2];
    config.offsets[neighbor] = offset;
  }

  // Throws on a bad config: a silently defaulted neighbor list produces a controller that
  // looks fine and converges to the wrong formation.
  config.validate();
  return config;
}

ADMMOptions ConsensusNode::loadOptions()
{
  ADMMOptions options;
  options.rho = this->declare_parameter<double>("rho", 1.0);
  options.max_iterations =
    static_cast<int>(this->declare_parameter<std::int64_t>("max_iterations", 50));
  options.alpha = this->declare_parameter<double>("alpha", 1.6);
  options.adaptive_rho = this->declare_parameter<bool>("adaptive_rho", false);
  options.max_staleness =
    static_cast<int>(this->declare_parameter<std::int64_t>("max_staleness", 5));
  // poll_timeout stays at the kernel default (2 ms): it is the per-iteration real-time
  // budget, not a launch knob.
  return options;
}

void ConsensusNode::controlStep()
{
  // Re-entered through the transport's spin_some() when a timer overrun fires while the
  // previous control step is still running; skip rather than recurse.
  if (impl_->in_control_step) {
    return;
  }
  impl_->in_control_step = true;

  const rclcpp::Time now = this->now();
  const auto period = std::chrono::milliseconds(
    static_cast<std::int64_t>(std::lround(1000.0 / impl_->control_rate_hz)));

  // (1) Gate: no state yet, or still waiting for a fresh sample after a safe-state entry.
  if (!impl_->have_state || impl_->need_fresh_state) {
    this->enterSafeState(impl_->have_state ? "waiting for fresh state" : "no state yet");
    impl_->in_control_step = false;
    return;
  }
  // (1b) Stale state: more than three control periods since the last estimate.
  if (impl_->last_state_stamp.nanoseconds() > 0 &&
    now - impl_->last_state_stamp > rclcpp::Duration(3 * period))
  {
    this->enterSafeState("stale state");
    impl_->in_control_step = false;
    return;
  }

  // Recover on the first good step after a safe-state hold: the stored warm start is stale
  // by an unknown amount, so reset it before solving.
  if (impl_->safe_state) {
    impl_->safe_state = false;
    impl_->kernel->reset();
    RCLCPP_INFO(this->get_logger(), "recovered from safe state; warm start reset");
  }

  // (2) New control step.
  impl_->transport->flush();
  impl_->kernel->setControlStep(++impl_->control_step);

  // (3) Seed the solve from the current state and (for leaders) the reference.
  impl_->kernel->setInitialState(impl_->x0);
  if (impl_->is_leader && impl_->have_reference) {
    impl_->kernel->setReference(impl_->reference);
  } else {
    impl_->kernel->setReference(Eigen::VectorXd());
  }

  // (4) Solve.
  const ADMMStats & stats = impl_->kernel->solve();

  // (5) A non-converged ADMM iterate is a suboptimal feasible input, not garbage: log it
  // but still apply it. Only a failed QP (zero iterations) warrants the safe state.
  if (!stats.converged) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "ADMM not converged in %d iterations (primal %.2e, dual %.2e)",
      stats.iterations, stats.primal_residual, stats.dual_residual);
  }
  if (stats.iterations == 0) {
    this->enterSafeState("local QP failed");
    impl_->in_control_step = false;
    return;
  }

  // (6) Apply the first input, shift the warm start, publish diagnostics.
  this->publishCommand(impl_->kernel->firstInput());
  impl_->kernel->shiftWarmStart();
  this->publishDiagnostics(stats);

  // Apply a topology change deferred from mid-solve.
  if (impl_->has_pending_graph) {
    impl_->has_pending_graph = false;
    this->applyGraphUpdate(impl_->pending_neighbors);
  }

  impl_->in_control_step = false;
}

void ConsensusNode::onState(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  const double px = msg->pose.position.x;
  const double py = msg->pose.position.y;

  double vx = 0.0;
  double vy = 0.0;
  if (impl_->have_state) {
    // Numerically differentiate the pose for velocity. A pose at 10 Hz is noisy; if an
    // Odometry estimate is available, subscribe to that instead.
    const rclcpp::Time stamp(msg->header.stamp);
    const double dt = (stamp - impl_->last_state_stamp).seconds();
    if (dt > 1e-6) {
      vx = (px - impl_->x0[0]) / dt;
      vy = (py - impl_->x0[1]) / dt;
    }
  }

  impl_->x0[0] = px;
  impl_->x0[1] = py;
  impl_->x0[2] = vx;
  impl_->x0[3] = vy;

  impl_->have_state = true;
  impl_->last_state_stamp = rclcpp::Time(msg->header.stamp);
  impl_->need_fresh_state = false;
}

void ConsensusNode::onReference(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  const int horizon = impl_->config.horizon;
  const int dim = impl_->config.dim;
  if (impl_->reference.size() != horizon * dim) {
    impl_->reference = Eigen::VectorXd::Zero(horizon * dim);
  }
  // A constant reference over the horizon: the simplest meaningful leader command.
  for (int k = 0; k < horizon; ++k) {
    impl_->reference[2 * k] = msg->pose.position.x;
    impl_->reference[2 * k + 1] = msg->pose.position.y;
  }
  impl_->have_reference = true;
}

void ConsensusNode::onGraphUpdate(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  if (msg->data.size() % 2 != 0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "ignoring malformed /admm/graph (odd length %zu)", msg->data.size());
    return;
  }

  const int self = impl_->config.agent_id;
  std::vector<int> neighbors;
  for (std::size_t i = 0; i < msg->data.size(); i += 2) {
    const int a = static_cast<int>(msg->data[i]);
    const int b = static_cast<int>(msg->data[i + 1]);
    if (a == self) {
      neighbors.push_back(b);
    } else if (b == self) {
      neighbors.push_back(a);
    }
  }
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

  if (neighbors == impl_->config.neighbors) {
    return;
  }

  // setNeighbors reallocates and re-setups OSQP: never call it from inside a solve.
  if (impl_->in_control_step) {
    impl_->pending_neighbors = std::move(neighbors);
    impl_->has_pending_graph = true;
    return;
  }

  this->applyGraphUpdate(neighbors);
}

void ConsensusNode::applyGraphUpdate(const std::vector<int> & neighbors)
{
  RCLCPP_INFO(
    this->get_logger(), "topology update %s -> %s", join(impl_->config.neighbors).c_str(),
    join(neighbors).c_str());
  impl_->config.neighbors = neighbors;
  impl_->transport->reconfigure(neighbors);
  impl_->kernel->setNeighbors(neighbors);
}

void ConsensusNode::publishCommand(const Eigen::VectorXd & u)
{
  geometry_msgs::msg::AccelStamped cmd;
  cmd.header.stamp = this->now();
  if (u.size() >= 2) {
    cmd.accel.linear.x = u[0];
    cmd.accel.linear.y = u[1];
  }
  impl_->cmd_pub->publish(cmd);
}

void ConsensusNode::publishDiagnostics(const ADMMStats & stats)
{
  std_msgs::msg::Float64MultiArray msg;
  msg.data = {
    static_cast<double>(stats.iterations),
    stats.primal_residual,
    stats.dual_residual,
    stats.rho,
    stats.solve_time_ms,
    stats.qp_time_ms,
    stats.comm_time_ms,
    static_cast<double>(stats.messages_sent),
    static_cast<double>(stats.messages_received),
    static_cast<double>(stats.messages_missing),
    static_cast<double>(stats.max_staleness_seen),
    stats.converged ? 1.0 : 0.0,
  };
  msg.layout.dim.resize(1);
  msg.layout.dim[0].label = "admm_stats";
  msg.layout.dim[0].size = static_cast<std::uint32_t>(msg.data.size());
  msg.layout.dim[0].stride = static_cast<std::uint32_t>(msg.data.size());
  impl_->diag_pub->publish(msg);
}

void ConsensusNode::enterSafeState(const std::string & reason)
{
  geometry_msgs::msg::AccelStamped cmd;
  cmd.header.stamp = this->now();
  impl_->cmd_pub->publish(cmd);

  if (!impl_->safe_state) {
    impl_->safe_state = true;
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "entering safe state: %s", reason.c_str());
  }
  // Recover only when a fresh state sample arrives.
  impl_->need_fresh_state = true;
}

}  // namespace cpp_admm

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  int exit_code = 0;
  try {
    auto node = std::make_shared<cpp_admm::ConsensusNode>();
    node->spin();
  } catch (const std::exception & error) {
    RCLCPP_ERROR(rclcpp::get_logger("consensus_node"), "fatal: %s", error.what());
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
