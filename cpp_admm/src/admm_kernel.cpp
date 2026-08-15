// Copyright (c) 2026, Ali-Eimaan. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Implementation of the transport-agnostic consensus-ADMM kernel.
//
// See admm_kernel.hpp for the contract and docs/derivations/consensus_admm_derivation.tex
// for the math. Keep the phase ordering in `iterate()` identical to the Python reference;
// test/test_admm_kernel.cpp compares the two iterate-by-iterate.

#include "cpp_admm/admm_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <utility>

#ifdef CPP_ADMM_WITH_ZMQ
#include <zmq.hpp>
#include <zmq_addon.hpp>
#endif

namespace cpp_admm
{

namespace
{

/// Serialised size of the fixed-width header (sender, subject, two int64s).
constexpr std::size_t kHeaderBytes =
  sizeof(int) + sizeof(int) + sizeof(int64_t) + sizeof(int64_t);

/// Clamp `v` into [lo, hi].
double clamp_value(double v, double lo, double hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

// ---------------------------------------------------------------------------- messages

std::size_t NeighborMessage::byte_size() const noexcept
{
  return kHeaderBytes + static_cast<std::size_t>(payload.size()) * sizeof(double);
}

// ------------------------------------------------------------------------ AgentConfig

std::vector<int> AgentConfig::closed_neighborhood() const
{
  std::vector<int> nb = neighbors;
  nb.push_back(agent_id);
  std::sort(nb.begin(), nb.end());
  nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
  return nb;
}

void AgentConfig::validate() const
{
  auto fail = [](const std::string & what) {throw std::invalid_argument(what);};

  if (agent_id < 0 || agent_id >= n_agents) {
    fail("agent_id " + std::to_string(agent_id) + " out of range [0, " +
         std::to_string(n_agents) + ")");
  }
  if (horizon <= 0) {
    fail("horizon must be positive");
  }
  if (dim != 2 && dim != 3) {
    fail("dim must be 2 or 3");
  }
  if (dt <= 0.0) {
    fail("dt must be positive");
  }
  if (q_position < 0.0 || q_velocity < 0.0 || r_input < 0.0 || r_rate < 0.0 ||
    p_terminal < 0.0 || w_formation < 0.0)
  {
    fail("cost weights must be non-negative");
  }
  if (u_max <= 0.0 || v_max <= 0.0) {
    fail("limits must be positive");
  }

  std::vector<int> sorted = neighbors;
  std::sort(sorted.begin(), sorted.end());
  for (int j : sorted) {
    if (j == agent_id) {
      fail("neighbors must not contain self");
    }
    if (j < 0 || j >= n_agents) {
      fail("neighbor " + std::to_string(j) + " out of range");
    }
  }
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    fail("duplicate neighbor");
  }

  for (const auto & [j, offset] : offsets) {
    if (std::find(neighbors.begin(), neighbors.end(), j) == neighbors.end()) {
      fail("offset key " + std::to_string(j) + " is not a neighbor");
    }
    if (offset.size() != dim) {
      fail("offset for neighbor " + std::to_string(j) + " has wrong size");
    }
  }
}

// ---------------------------------------------------------------------------- stats

void ADMMStats::reset() noexcept
{
  const double keep_rho = rho;
  *this = ADMMStats{};
  rho = keep_rho;
}

// -------------------------------------------------------------- InProcessTransport

struct InProcessTransport::Impl
{
  int agent_id{0};
  double loss_prob{0.0};
  int max_delay{0};
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> loss_dist{0.0, 1.0};
  std::uniform_int_distribution<int> delay_dist{0, 0};

  // Per-kind inboxes; each entry is (arrival_iteration, message).
  std::deque<std::pair<int64_t, NeighborMessage>> inbox[2];

  // Delivery clock: the highest admm_iteration this transport has published. Messages
  // with arrival_iteration <= clock are eligible for delivery on the next poll.
  int64_t clock{0};

  static std::vector<Impl *> & peers()
  {
    static std::vector<Impl *> registry;
    return registry;
  }
};

InProcessTransport::InProcessTransport(
  int agent_id, double loss_prob, int max_delay, uint64_t seed)
{
  impl_ = std::make_unique<Impl>();
  impl_->agent_id = agent_id;
  impl_->loss_prob = loss_prob;
  impl_->max_delay = max_delay;
  impl_->rng = std::mt19937_64(seed);
  impl_->loss_dist = std::uniform_real_distribution<double>(0.0, 1.0);
  impl_->delay_dist = std::uniform_int_distribution<int>(0, max_delay);
}

InProcessTransport::~InProcessTransport()
{
  if (!impl_) {
    return;
  }
  auto & peers = Impl::peers();
  peers.erase(std::remove(peers.begin(), peers.end(), impl_.get()), peers.end());
}

bool InProcessTransport::publish(MessageKind kind, const NeighborMessage & message)
{
  auto & peers = Impl::peers();
  impl_->clock = std::max(impl_->clock, message.admm_iteration);

  if (impl_->loss_dist(impl_->rng) < impl_->loss_prob) {
    return false;
  }
  const int64_t arrival = message.admm_iteration + impl_->delay_dist(impl_->rng);
  const std::size_t k = static_cast<std::size_t>(kind);
  for (Impl * peer : peers) {
    if (peer == impl_.get()) {
      continue;
    }
    peer->inbox[k].emplace_back(arrival, message);
  }
  return true;
}

std::size_t InProcessTransport::poll(
  MessageKind kind, std::chrono::microseconds /*timeout*/, std::vector<NeighborMessage> & out)
{
  out.clear();
  const std::size_t k = static_cast<std::size_t>(kind);
  auto & queue = impl_->inbox[k];
  const int64_t now = impl_->clock;

  std::size_t delivered = 0;
  for (auto it = queue.begin(); it != queue.end(); ) {
    if (it->first <= now) {
      out.push_back(std::move(it->second));
      it = queue.erase(it);
      ++delivered;
    } else {
      ++it;
    }
  }
  return delivered;
}

void InProcessTransport::flush()
{
  impl_->inbox[0].clear();
  impl_->inbox[1].clear();
}

void InProcessTransport::connect(const std::vector<InProcessTransport *> & peers)
{
  auto & registry = Impl::peers();
  registry.clear();
  registry.reserve(peers.size());
  for (InProcessTransport * peer : peers) {
    if (peer != nullptr) {
      registry.push_back(peer->impl_.get());
    }
  }
}

// ----------------------------------------------------------------- ZeroMqTransport

#ifdef CPP_ADMM_WITH_ZMQ
namespace
{

std::string topic_for(MessageKind kind, int subject)
{
  return "admm/" + std::to_string(static_cast<int>(kind)) + "/" + std::to_string(subject);
}

int kind_from_topic(const std::string & topic)
{
  // "admm/<kind>/<subject>"; std::stoi stops at the first '/'.
  return std::stoi(topic.substr(5));
}

void serialize_message(const NeighborMessage & m, std::vector<unsigned char> & buf)
{
  buf.resize(kHeaderBytes + static_cast<std::size_t>(m.payload.size()) * sizeof(double));
  unsigned char * p = buf.data();
  std::memcpy(p, &m.sender, sizeof(int));
  std::memcpy(p + sizeof(int), &m.subject, sizeof(int));
  std::memcpy(p + 2 * sizeof(int), &m.admm_iteration, sizeof(int64_t));
  std::memcpy(p + 2 * sizeof(int) + sizeof(int64_t), &m.control_step, sizeof(int64_t));
  if (m.payload.size() > 0) {
    std::memcpy(
      p + kHeaderBytes, m.payload.data(),
          static_cast<std::size_t>(m.payload.size()) * sizeof(double));
  }
}

NeighborMessage deserialize_message(const unsigned char * data, std::size_t size)
{
  NeighborMessage m;
  std::memcpy(&m.sender, data, sizeof(int));
  std::memcpy(&m.subject, data + sizeof(int), sizeof(int));
  std::memcpy(&m.admm_iteration, data + 2 * sizeof(int), sizeof(int64_t));
  std::memcpy(&m.control_step, data + 2 * sizeof(int) + sizeof(int64_t), sizeof(int64_t));
  const std::size_t n = (size - kHeaderBytes) / sizeof(double);
  m.payload.resize(static_cast<Eigen::Index>(n));
  if (n > 0) {
    std::memcpy(m.payload.data(), data + kHeaderBytes, n * sizeof(double));
  }
  return m;
}

}  // namespace

struct ZeroMqTransport::Impl
{
  zmq::context_t context{1};
  zmq::socket_t pub;
  zmq::socket_t sub;
};

ZeroMqTransport::ZeroMqTransport(
  int agent_id, std::string bind_endpoint, std::unordered_map<int, std::string> endpoints)
{
  impl_ = std::make_unique<Impl>();
  impl_->pub = zmq::socket_t(impl_->context, zmq::socket_type::pub);
  impl_->sub = zmq::socket_t(impl_->context, zmq::socket_type::sub);

  // Bound both queues so a backlog never turns into unbounded staleness. A receive
  // high-water mark, rather than ZMQ_CONFLATE, is used because the shared SUB socket
  // carries *two* message kinds; CONFLATE would silently discard one kind in favour of
  // whichever arrived last.
  impl_->pub.set(zmq::sockopt::sndhwm, 1);
  impl_->sub.set(zmq::sockopt::rcvhwm, 64);
  impl_->sub.set(zmq::sockopt::linger, 0);

  impl_->pub.bind(bind_endpoint);

  // We must receive local copies addressed to us (from every contributor) and consensus
  // values broadcast by each neighbor.
  impl_->sub.set(zmq::sockopt::subscribe, topic_for(MessageKind::kLocalCopy, agent_id));
  for (const auto & [neighbor, endpoint] : endpoints) {
    impl_->sub.connect(endpoint);
    impl_->sub.set(zmq::sockopt::subscribe, topic_for(MessageKind::kConsensus, neighbor));
  }
}

ZeroMqTransport::~ZeroMqTransport() = default;

bool ZeroMqTransport::publish(MessageKind kind, const NeighborMessage & message)
{
  std::vector<unsigned char> buf;
  serialize_message(message, buf);

  const std::string topic = topic_for(kind, message.subject);
  zmq::message_t topic_msg(topic.data(), topic.size());
  zmq::message_t payload_msg(buf.data(), buf.size());

  const auto first = impl_->pub.send(topic_msg, zmq::send_flags::sndmore);
  const auto second = impl_->pub.send(payload_msg, zmq::send_flags::none);
  return first.has_value() && second.has_value();
}

std::size_t ZeroMqTransport::poll(
  MessageKind kind, std::chrono::microseconds timeout, std::vector<NeighborMessage> & out)
{
  out.clear();

  zmq_pollitem_t item{impl_->sub.handle(), 0, ZMQ_POLLIN, 0};
  const long timeout_ms = static_cast<long>((timeout.count() + 999) / 1000);
  const int rc = zmq::poll(&item, 1, std::chrono::milliseconds(timeout_ms));
  if (rc <= 0 || (item.revents & ZMQ_POLLIN) == 0) {
    return 0;
  }

  std::size_t received = 0;
  while (true) {
    std::vector<zmq::message_t> frames;
    if (!zmq::recv_multipart(impl_->sub, std::back_inserter(frames), zmq::recv_flags::dontwait)) {
      break;
    }
    if (frames.size() < 2) {
      continue;
    }
    const std::string topic(static_cast<const char *>(frames[0].data()), frames[0].size());
    if (kind_from_topic(topic) != static_cast<int>(kind)) {
      continue;
    }
    out.push_back(deserialize_message(
      static_cast<const unsigned char *>(frames[1].data()), frames[1].size()));
    ++received;
  }
  return received;
}

void ZeroMqTransport::flush()
{
  // Drain anything left in the SUB queue; PUB is fire-and-forget and has nothing to drop.
  zmq_pollitem_t item{impl_->sub.handle(), 0, ZMQ_POLLIN, 0};
  while (zmq::poll(&item, 1, std::chrono::milliseconds(0)) > 0 &&
    (item.revents & ZMQ_POLLIN) != 0)
  {
    std::vector<zmq::message_t> frames;
    if (!zmq::recv_multipart(impl_->sub, std::back_inserter(frames), zmq::recv_flags::dontwait)) {
      break;
    }
  }
}

#else  // !CPP_ADMM_WITH_ZMQ

struct ZeroMqTransport::Impl
{
};

ZeroMqTransport::ZeroMqTransport(int, std::string, std::unordered_map<int, std::string>)
{
  throw std::runtime_error("ZeroMqTransport requires a build with CPP_ADMM_WITH_ZMQ enabled");
}

ZeroMqTransport::~ZeroMqTransport() = default;

bool ZeroMqTransport::publish(MessageKind, const NeighborMessage &)
{
  throw std::logic_error("ZeroMqTransport unavailable without CPP_ADMM_WITH_ZMQ");
}

std::size_t ZeroMqTransport::poll(
  MessageKind, std::chrono::microseconds, std::vector<NeighborMessage> &)
{
  throw std::logic_error("ZeroMqTransport unavailable without CPP_ADMM_WITH_ZMQ");
}

void ZeroMqTransport::flush()
{
  throw std::logic_error("ZeroMqTransport unavailable without CPP_ADMM_WITH_ZMQ");
}

#endif  // CPP_ADMM_WITH_ZMQ

// ---------------------------------------------------------------------- AdmmKernel

struct AdmmKernel::Impl
{
  AgentConfig config;
  ADMMOptions options;
  ITransport * transport{nullptr};

  std::unique_ptr<PerAgentQp> qp;

  std::vector<int> closed_nbhd;                  // block ordering (sorted closed neighborhood)
  std::unordered_map<int, int> block_index;      // agent id -> index within closed_nbhd

  int horizon{0};
  int dim{0};
  int B{0};       // horizon * dim (one trajectory block)
  int M{0};       // |closed_nbhd|
  int n_states{0};

  // Iteration state, keyed by neighbor id (and self). Every Eigen vector is sized in
  // configure() and never resized afterwards.
  std::unordered_map<int, Eigen::VectorXd> y;              // local copies
  std::unordered_map<int, Eigen::VectorXd> y_hat;          // relaxed copies
  std::unordered_map<int, Eigen::VectorXd> lam;            // scaled duals
  std::unordered_map<int, Eigen::VectorXd> z_received;     // received consensus (self == z_self)
  std::unordered_map<int, Eigen::VectorXd> z_received_prev;
  Eigen::VectorXd z_self;
  Eigen::VectorXd z_self_prev;
  Eigen::VectorXd inputs;
  Eigen::VectorXd x0;
  Eigen::VectorXd reference;
  bool has_reference{false};

  std::unordered_map<int, Eigen::VectorXd> offsets;

  // z-update scratch: received (y_hat + lam) contributions, keyed by neighbor id.
  std::unordered_map<int, Eigen::VectorXd> contributions;
  std::vector<int> received_contributors;
  std::vector<char> contrib_present;

  std::unordered_map<int, int64_t> last_seen_iteration;

  std::vector<NeighborMessage> rx_buffer;   // reserved once, reused every poll
  NeighborMessage tx_msg;                   // reused publish buffer
  Eigen::VectorXd scratch_B;                // horizon*dim scratch

  ADMMStats stats;
  int64_t control_step{0};
  int iteration{0};
  double rho{1.0};
  double rho_prev_qp{1.0};   // rho currently baked into the QP's Hessian
};

AdmmKernel::AdmmKernel(AgentConfig config, ADMMOptions options, ITransport * transport)
{
  impl_ = std::make_unique<Impl>();
  impl_->config = std::move(config);
  impl_->options = options;
  impl_->transport = transport;
}

AdmmKernel::~AdmmKernel() = default;

void AdmmKernel::configure()
{
  impl_->config.validate();
  impl_->closed_nbhd = impl_->config.closed_neighborhood();

  impl_->horizon = impl_->config.horizon;
  impl_->dim = impl_->config.dim;
  impl_->B = impl_->horizon * impl_->dim;
  impl_->n_states = 2 * impl_->dim;
  impl_->M = static_cast<int>(impl_->closed_nbhd.size());

  for (int b = 0; b < impl_->M; ++b) {
    impl_->block_index[impl_->closed_nbhd[static_cast<std::size_t>(b)]] = b;
  }

  impl_->x0 = Eigen::VectorXd::Zero(impl_->n_states);
  impl_->reference = Eigen::VectorXd::Zero(impl_->B);
  impl_->has_reference = false;
  impl_->inputs = Eigen::VectorXd::Zero(impl_->B);
  impl_->z_self = Eigen::VectorXd::Zero(impl_->B);
  impl_->z_self_prev = Eigen::VectorXd::Zero(impl_->B);
  impl_->scratch_B = Eigen::VectorXd::Zero(impl_->B);

  // Normalise offsets so every formation edge has a (possibly zero) offset.
  impl_->offsets = impl_->config.offsets;
  for (int j : impl_->config.neighbors) {
    if (impl_->offsets.count(j) == 0) {
      impl_->offsets[j] = Eigen::VectorXd::Zero(impl_->dim);
    }
  }

  for (int j : impl_->closed_nbhd) {
    impl_->y[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->y_hat[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->lam[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->z_received[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->z_received_prev[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->contributions[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->last_seen_iteration[j] = 0;
  }

  impl_->received_contributors.assign(static_cast<std::size_t>(impl_->M), -1);
  impl_->contrib_present.assign(static_cast<std::size_t>(impl_->M), 0);

  impl_->rx_buffer.reserve(static_cast<std::size_t>(impl_->M));
  impl_->tx_msg.payload = Eigen::VectorXd::Zero(impl_->B);

  impl_->rho = impl_->options.rho;
  impl_->rho_prev_qp = impl_->options.rho;

  impl_->qp = std::make_unique<PerAgentQp>(impl_->config, impl_->options.qp_settings);
  impl_->qp->setup();
  // Sync the QP Hessian to the initial rho (PerAgentQp builds with rho_p == 1.0).
  impl_->qp->updateRho(impl_->options.rho);
}

void AdmmKernel::setInitialState(const Eigen::VectorXd & x0)
{
  impl_->x0 = x0;
}

void AdmmKernel::setReference(const Eigen::VectorXd & reference)
{
  impl_->reference = reference;
  impl_->has_reference = reference.size() != 0;
}

void AdmmKernel::setOffsets(const std::unordered_map<int, Eigen::VectorXd> & offsets)
{
  impl_->offsets = offsets;
  for (int j : impl_->config.neighbors) {
    if (impl_->offsets.count(j) == 0) {
      impl_->offsets[j] = Eigen::VectorXd::Zero(impl_->dim);
    }
  }
  impl_->qp->updateOffsets(impl_->offsets);
}

void AdmmKernel::setNeighbors(const std::vector<int> & neighbors)
{
  // Preserve surviving y/lam entries, then rebuild the neighborhood and the QP. This
  // reallocates (a topology change alters the number of decision variables), so it must
  // never be called from a real-time thread.
  std::vector<int> new_nbhd = neighbors;
  new_nbhd.push_back(impl_->config.agent_id);
  std::sort(new_nbhd.begin(), new_nbhd.end());
  new_nbhd.erase(std::unique(new_nbhd.begin(), new_nbhd.end()), new_nbhd.end());

  std::unordered_map<int, Eigen::VectorXd> y_new;
  std::unordered_map<int, Eigen::VectorXd> lam_new;
  for (int j : new_nbhd) {
    auto yit = impl_->y.find(j);
    auto lit = impl_->lam.find(j);
    y_new[j] = (yit != impl_->y.end()) ? yit->second : Eigen::VectorXd::Zero(impl_->B);
    lam_new[j] = (lit != impl_->lam.end()) ? lit->second : Eigen::VectorXd::Zero(impl_->B);
  }
  impl_->y = std::move(y_new);
  impl_->lam = std::move(lam_new);

  impl_->config.neighbors = neighbors;
  std::sort(impl_->config.neighbors.begin(), impl_->config.neighbors.end());

  // Prune offsets for dropped neighbors so validate() does not reject a topology change,
  // and re-normalise the kernel's own offsets view for the surviving neighborhood.
  {
    std::unordered_map<int, Eigen::VectorXd> pruned_offsets;
    for (const auto & [j, d] : impl_->config.offsets) {
      if (std::find(impl_->config.neighbors.begin(), impl_->config.neighbors.end(), j) !=
        impl_->config.neighbors.end())
      {
        pruned_offsets[j] = d;
      }
    }
    impl_->config.offsets = std::move(pruned_offsets);
  }
  impl_->offsets = impl_->config.offsets;
  for (int j : impl_->config.neighbors) {
    if (impl_->offsets.count(j) == 0) {
      impl_->offsets[j] = Eigen::VectorXd::Zero(impl_->dim);
    }
  }

  impl_->config.validate();

  // Rebuild the neighborhood-derived maps from scratch.
  impl_->closed_nbhd = impl_->config.closed_neighborhood();
  impl_->M = static_cast<int>(impl_->closed_nbhd.size());
  impl_->block_index.clear();
  for (int b = 0; b < impl_->M; ++b) {
    impl_->block_index[impl_->closed_nbhd[static_cast<std::size_t>(b)]] = b;
  }

  for (int j : impl_->closed_nbhd) {
    impl_->y_hat[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->z_received[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->z_received_prev[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->contributions[j] = Eigen::VectorXd::Zero(impl_->B);
    impl_->last_seen_iteration[j] = 0;
  }
  impl_->received_contributors.assign(static_cast<std::size_t>(impl_->M), -1);
  impl_->contrib_present.assign(static_cast<std::size_t>(impl_->M), 0);
  impl_->rx_buffer.clear();
  impl_->rx_buffer.reserve(static_cast<std::size_t>(impl_->M));

  impl_->qp = std::make_unique<PerAgentQp>(impl_->config, impl_->options.qp_settings);
  impl_->qp->setup();
  impl_->qp->updateRho(impl_->rho);
  impl_->rho_prev_qp = impl_->rho;
}

bool AdmmKernel::iterate()
{
  // Snapshot the previous consensus view for the dual residual (mirrors the Python
  // z_prev capture at the top of each iteration).
  for (int j : impl_->closed_nbhd) {
    impl_->z_received_prev[j] = impl_->z_received[j];
  }
  impl_->z_self_prev = impl_->z_self;

  // Per-iteration message accounting; max_staleness_seen stays a running maximum.
  impl_->stats.messages_sent = 0;
  impl_->stats.messages_received = 0;
  impl_->stats.messages_missing = 0;

  xUpdate();
  relax();
  exchangeLocalCopies();
  zUpdate();
  broadcastConsensus();
  dualUpdate();
  computeResiduals();
  if (impl_->options.adaptive_rho) {
    updateRho();
  }

  ++impl_->iteration;
  impl_->stats.iterations = impl_->iteration;
  return impl_->stats.converged;
}

const ADMMStats & AdmmKernel::solve()
{
  impl_->stats.reset();
  impl_->stats.rho = impl_->rho;
  impl_->iteration = 0;

  for (int k = 0; k < impl_->options.max_iterations; ++k) {
    if (iterate()) {
      break;
    }
    if (impl_->options.max_staleness > 0 &&
      impl_->stats.max_staleness_seen > impl_->options.max_staleness)
    {
      break;
    }
  }
  return impl_->stats;
}

void AdmmKernel::shiftWarmStart()
{
  const int d = impl_->dim;
  const int B = impl_->B;

  auto shift_block = [&](Eigen::VectorXd & block) {
      if (block.size() != B) {
        return;
      }
      for (int t = 0; t < impl_->horizon - 1; ++t) {
        block.segment(t * d, d) = block.segment((t + 1) * d, d);
      }
      block.segment((impl_->horizon - 1) * d, d) = block.segment((impl_->horizon - 2) * d, d);
    };

  for (int j : impl_->closed_nbhd) {
    shift_block(impl_->y[j]);
    shift_block(impl_->lam[j]);
    shift_block(impl_->z_received[j]);
  }
  shift_block(impl_->z_self);
}

void AdmmKernel::reset()
{
  for (int j : impl_->closed_nbhd) {
    impl_->y[j].setZero();
    impl_->y_hat[j].setZero();
    impl_->lam[j].setZero();
    impl_->z_received[j].setZero();
    impl_->z_received_prev[j].setZero();
    impl_->contributions[j].setZero();
    impl_->last_seen_iteration[j] = 0;
  }
  impl_->z_self.setZero();
  impl_->z_self_prev.setZero();
  impl_->inputs.setZero();
  impl_->x0.setZero();
  impl_->reference.setZero();
  impl_->has_reference = false;
  impl_->iteration = 0;

  impl_->stats.reset();
  impl_->stats.rho = impl_->rho;
  if (impl_->transport != nullptr) {
    impl_->transport->flush();
  }
}

Eigen::VectorXd AdmmKernel::firstInput() const
{
  return impl_->inputs.head(impl_->dim);
}

const Eigen::VectorXd & AdmmKernel::inputs() const noexcept
{
  return impl_->inputs;
}

const Eigen::VectorXd & AdmmKernel::consensusTrajectory() const noexcept
{
  return impl_->z_self;
}

const Eigen::VectorXd & AdmmKernel::localCopy(int j) const
{
  return impl_->y.at(j);
}

const Eigen::VectorXd & AdmmKernel::dual(int j) const
{
  return impl_->lam.at(j);
}

const ADMMStats & AdmmKernel::stats() const noexcept
{
  return impl_->stats;
}

const AgentConfig & AdmmKernel::config() const noexcept
{
  return impl_->config;
}

void AdmmKernel::setControlStep(int64_t step) noexcept
{
  impl_->control_step = step;
}

// ------------------------------------------------------------------- private phases

void AdmmKernel::xUpdate()
{
  impl_->qp->updateInitialState(impl_->x0);
  if (impl_->has_reference) {
    impl_->qp->updateReference(impl_->reference);
  } else {
    impl_->qp->updateReference(Eigen::VectorXd{});
  }
  if (impl_->rho != impl_->rho_prev_qp) {
    impl_->qp->updateRho(impl_->rho);
    impl_->rho_prev_qp = impl_->rho;
  }
  impl_->qp->updateConsensus(impl_->z_received, impl_->lam, impl_->rho);

  const QpSolution & sol = impl_->qp->solve();
  impl_->stats.qp_time_ms = sol.solve_time_ms;

  if (!sol.ok()) {
    // Hold the previous y and inputs; never propagate a failed inner solve into the
    // consensus step.
    return;
  }

  impl_->inputs = sol.theta.head(impl_->B);
  for (int j : impl_->closed_nbhd) {
    impl_->y[j] = sol.theta.segment(impl_->qp->copyOffset(j), impl_->B);
  }
}

void AdmmKernel::relax()
{
  const double alpha = impl_->options.alpha;
  if (alpha == 1.0) {
    for (int j : impl_->closed_nbhd) {
      impl_->y_hat[j] = impl_->y[j];
    }
    return;
  }

  for (int j : impl_->closed_nbhd) {
    impl_->y_hat[j] = alpha * impl_->y[j];
    impl_->y_hat[j] += (1.0 - alpha) * impl_->z_received[j];
  }
}

void AdmmKernel::exchangeLocalCopies()
{
  const int agent_id = impl_->config.agent_id;
  const int M = impl_->M;

  // Reset the contribution flags.
  for (int b = 0; b < M; ++b) {
    impl_->contrib_present[static_cast<std::size_t>(b)] = 0;
  }

  // Publish (y_hat + lam) to every neighbor j (subject = j). The payload already includes
  // the scaled dual so the receiving subject can average y_hat + lam directly.
  if (impl_->transport != nullptr) {
    for (int j : impl_->closed_nbhd) {
      if (j == agent_id) {
        continue;
      }
      impl_->scratch_B = impl_->y_hat[j];
      impl_->scratch_B += impl_->lam[j];
      impl_->tx_msg.sender = agent_id;
      impl_->tx_msg.subject = j;
      impl_->tx_msg.admm_iteration = impl_->iteration;
      impl_->tx_msg.control_step = impl_->control_step;
      impl_->tx_msg.payload = impl_->scratch_B;
      if (impl_->transport->publish(MessageKind::kLocalCopy, impl_->tx_msg)) {
        ++impl_->stats.messages_sent;
      }
    }
  }

  // Collect contributions from other contributors, subject == self.
  impl_->rx_buffer.clear();
  if (impl_->transport != nullptr) {
    impl_->transport->poll(
      MessageKind::kLocalCopy, impl_->options.poll_timeout, impl_->rx_buffer);
  }

  int n_received = 0;
  for (const NeighborMessage & msg : impl_->rx_buffer) {
    if (msg.subject != agent_id) {
      continue;
    }
    if (msg.control_step != impl_->control_step) {
      continue;
    }
    auto it = impl_->block_index.find(msg.sender);
    if (it == impl_->block_index.end()) {
      continue;
    }
    const int b = it->second;
    impl_->contributions[msg.sender] = msg.payload;
    impl_->contrib_present[static_cast<std::size_t>(b)] = 1;
    impl_->received_contributors[static_cast<std::size_t>(n_received)] = msg.sender;
    ++n_received;
  }
  impl_->stats.messages_received += n_received;
  impl_->stats.messages_missing +=
    static_cast<int>(impl_->config.neighbors.size()) - n_received;
}

void AdmmKernel::zUpdate()
{
  const int agent_id = impl_->config.agent_id;
  const int M = impl_->M;

  // Self contribution is always present.
  impl_->z_self = impl_->y_hat[agent_id];
  impl_->z_self += impl_->lam[agent_id];
  int count = 1;

  for (int b = 0; b < M; ++b) {
    const int j = impl_->closed_nbhd[static_cast<std::size_t>(b)];
    if (j == agent_id || !impl_->contrib_present[static_cast<std::size_t>(b)]) {
      continue;
    }
    impl_->z_self += impl_->contributions[j];
    ++count;
  }
  impl_->z_self /= static_cast<double>(count);

  // Keep the received view of our own trajectory consistent with z_self.
  impl_->z_received[agent_id] = impl_->z_self;
}

void AdmmKernel::broadcastConsensus()
{
  const int agent_id = impl_->config.agent_id;

  if (impl_->transport != nullptr) {
    impl_->tx_msg.sender = agent_id;
    impl_->tx_msg.subject = agent_id;
    impl_->tx_msg.admm_iteration = impl_->iteration;
    impl_->tx_msg.control_step = impl_->control_step;
    impl_->tx_msg.payload = impl_->z_self;
    if (impl_->transport->publish(MessageKind::kConsensus, impl_->tx_msg)) {
      ++impl_->stats.messages_sent;
    }
  }

  impl_->rx_buffer.clear();
  if (impl_->transport != nullptr) {
    impl_->transport->poll(
      MessageKind::kConsensus, impl_->options.poll_timeout, impl_->rx_buffer);
  }

  int n_received = 0;
  for (const NeighborMessage & msg : impl_->rx_buffer) {
    if (msg.subject == agent_id) {
      continue;  // our own broadcast, already reflected in z_self
    }
    if (impl_->block_index.count(msg.subject) == 0) {
      continue;  // not a neighbor
    }
    if (msg.control_step != impl_->control_step) {
      continue;
    }
    impl_->z_received[msg.subject] = msg.payload;
    impl_->last_seen_iteration[msg.subject] = impl_->iteration;
    ++n_received;
  }

  impl_->stats.messages_received += n_received;

  // For each neighbor we did not hear from, fall back to the last-known z^j and record
  // how stale it is.
  int missing = 0;
  for (int j : impl_->closed_nbhd) {
    if (j == agent_id) {
      continue;
    }
    if (impl_->last_seen_iteration[j] != impl_->iteration) {
      ++missing;
      const int64_t staleness = impl_->iteration - impl_->last_seen_iteration[j];
      if (staleness > impl_->stats.max_staleness_seen) {
        impl_->stats.max_staleness_seen = static_cast<int>(staleness);
      }
    }
  }
  impl_->stats.messages_missing += missing;
}

void AdmmKernel::dualUpdate()
{
  for (int j : impl_->closed_nbhd) {
    impl_->lam[j] += impl_->y_hat[j];
    impl_->lam[j] -= impl_->z_received[j];
  }
}

void AdmmKernel::computeResiduals()
{
  double primal_sq = 0.0;
  double dual_sq = 0.0;
  double y_norm_sq = 0.0;
  double z_norm_sq = 0.0;
  double lam_norm_sq = 0.0;
  int n_dual = 0;

  for (int j : impl_->closed_nbhd) {
    primal_sq += (impl_->y[j] - impl_->z_received[j]).squaredNorm();
    dual_sq += (impl_->z_received[j] - impl_->z_received_prev[j]).squaredNorm();
    y_norm_sq += impl_->y[j].squaredNorm();
    z_norm_sq += impl_->z_received[j].squaredNorm();
    lam_norm_sq += impl_->lam[j].squaredNorm();
    n_dual += impl_->B;
  }

  const double primal = std::sqrt(primal_sq);
  const double dual = impl_->rho * std::sqrt(dual_sq);

  const double sqrt_n = std::sqrt(static_cast<double>(n_dual));
  const double eps_primal =
    sqrt_n * impl_->options.eps_abs +
    impl_->options.eps_rel * std::max(std::sqrt(y_norm_sq), std::sqrt(z_norm_sq));
  const double eps_dual =
    sqrt_n * impl_->options.eps_abs +
    impl_->options.eps_rel * impl_->rho * std::sqrt(lam_norm_sq);

  impl_->stats.primal_residual = primal;
  impl_->stats.dual_residual = dual;
  impl_->stats.converged = primal <= eps_primal && dual <= eps_dual;
}

void AdmmKernel::updateRho()
{
  const double primal = impl_->stats.primal_residual;
  const double dual = impl_->stats.dual_residual;

  double factor = 1.0;
  if (primal > impl_->options.mu * dual) {
    factor = impl_->options.tau;
  } else if (dual > impl_->options.mu * primal) {
    factor = 1.0 / impl_->options.tau;
  }

  const double new_rho = clamp_value(
    impl_->rho * factor, impl_->options.rho_min, impl_->options.rho_max);
  const double actual = new_rho / impl_->rho;

  if (actual != 1.0) {
    for (int j : impl_->closed_nbhd) {
      impl_->lam[j] /= actual;
    }
    impl_->rho = new_rho;
    impl_->stats.rho = impl_->rho;
    impl_->qp->updateRho(impl_->rho);
    impl_->rho_prev_qp = impl_->rho;
  }
}

int AdmmKernel::blockIndex(int j) const
{
  const auto it = impl_->block_index.find(j);
  if (it == impl_->block_index.end()) {
    throw std::out_of_range("unknown agent id " + std::to_string(j));
  }
  return it->second;
}

}  // namespace cpp_admm
