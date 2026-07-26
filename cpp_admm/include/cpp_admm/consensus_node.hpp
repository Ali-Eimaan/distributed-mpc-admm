// ROS 2 node running one agent's ADMM updates.
//
// One process per agent. The node owns an AdmmKernel and implements ITransport on top of
// ROS 2 topics, so a 4-agent run is four processes that exchange only neighbor traffic --
// there is no central coordinator anywhere in this package.
//
// Topics
// ------
//   subscribe  ~/state                      geometry_msgs/PoseStamped (own state estimate)
//   subscribe  /admm/copy/<subject>         std_msgs/Float64MultiArray  (y_i^subject)
//   subscribe  /admm/consensus/<neighbor>   std_msgs/Float64MultiArray  (z^neighbor)
//   subscribe  /admm/graph                  std_msgs/Int32MultiArray    (topology updates)
//   publish    /admm/copy/<neighbor>        std_msgs/Float64MultiArray
//   publish    /admm/consensus/<agent_id>   std_msgs/Float64MultiArray
//   publish    ~/cmd                        geometry_msgs/AccelStamped  (first input)
//   publish    ~/diagnostics                std_msgs/Float64MultiArray  (ADMMStats)
//
// Float64MultiArray is used rather than a custom .msg to keep this package free of a
// message-generation dependency; the layout field carries (subject, iteration, step).
// If this graduates into `transition-viable-swarm`, replace it with a typed message --
// the untyped array has no schema and will eventually be mismatched silently.
//
// QoS
// ---
// ADMM traffic uses best-effort, volatile, depth 1. Reliable QoS is the wrong choice: a
// retransmitted stale iterate is worse than a dropped one, because the kernel handles a
// miss explicitly but cannot tell that a late arrival is late.
//
// Threading
// ---------
// Single-threaded executor, one timer at the control rate. The ADMM iteration loop spins
// inside the timer callback and polls the subscription queues directly, so `poll()` must
// pump the executor rather than sleep -- see the note in consensus_node.cpp.

#ifndef CPP_ADMM__CONSENSUS_NODE_HPP_
#define CPP_ADMM__CONSENSUS_NODE_HPP_

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/accel_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include "cpp_admm/admm_kernel.hpp"

namespace cpp_admm
{

/// ITransport implementation backed by ROS 2 publishers and subscriptions.
/// Owned by ConsensusNode; declared here so it can be unit-tested against a mock node.
class RosTransport : public ITransport
{
public:
  RosTransport(rclcpp::Node * node, int agent_id, const std::vector<int> & neighbors);
  ~RosTransport() override;

  bool publish(MessageKind kind, const NeighborMessage & message) override;
  std::size_t poll(
    MessageKind kind, std::chrono::microseconds timeout,
    std::vector<NeighborMessage> & out) override;
  void flush() override;

  /// Re-create publishers and subscriptions after a topology change.
  void reconfigure(const std::vector<int> & neighbors);

private:
  void onCopy(int subject, const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void onConsensus(int subject, const std_msgs::msg::Float64MultiArray::SharedPtr msg);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// One agent. Parameters (all declared in the constructor, all overridable from launch):
///
///   agent_id            int      required
///   n_agents            int      required
///   neighbors           int[]    required, open neighborhood
///   horizon             int      15
///   dt                  double   0.1
///   control_rate_hz     double   10.0
///   rho                 double   1.0
///   max_iterations      int      50
///   alpha               double   1.6
///   adaptive_rho        bool     false
///   formation_offsets   double[] flattened (neighbor, dx, dy) triples
///   is_leader           bool     false
///   reference_topic     string   "" (leaders only)
///   u_max, v_max        double   3.0, 2.0
///   q_position ... w_formation   cost weights, see AgentConfig
class ConsensusNode : public rclcpp::Node
{
public:
  explicit ConsensusNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ConsensusNode() override;

private:
  /// Read every parameter into an AgentConfig and validate it. Throws on a bad config --
  /// failing at startup is correct here; a silently defaulted neighbor list produces a
  /// controller that looks fine and converges to the wrong formation.
  AgentConfig loadConfig();
  ADMMOptions loadOptions();

  /// Timer callback at `control_rate_hz`: read state, run the ADMM solve, publish the
  /// first input, shift the warm start, publish diagnostics.
  void controlStep();

  void onState(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void onReference(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  /// Topology update: `[i0, j0, i1, j1, ...]` edge list. Recomputes this agent's
  /// neighborhood, reconfigures the transport, and calls `AdmmKernel::setNeighbors`.
  /// Deliberately not real-time safe -- it reallocates. Rate-limit it upstream.
  void onGraphUpdate(const std_msgs::msg::Int32MultiArray::SharedPtr msg);

  void publishCommand(const Eigen::VectorXd & u);
  void publishDiagnostics(const ADMMStats & stats);

  /// Zero command plus a throttled warning. Entered when the state estimate is stale, the
  /// QP fails, or `max_staleness` is exceeded. Every one of those must be a defined,
  /// logged state rather than a stale input silently held forever.
  void enterSafeState(const std::string & reason);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_admm

#endif  // CPP_ADMM__CONSENSUS_NODE_HPP_
