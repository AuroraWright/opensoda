#include "transducer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace tflite::ops::builtin {
TfLiteRegistration* Register_BATCH_MATMUL_REF();
}  // namespace tflite::ops::builtin

namespace soda {
namespace {

int Dim(const TfLiteTensor* t, int i) { return t->dims->data[i]; }

constexpr float kNoCost = 1e20f;  // greco's infinite cost.

float LogAdd(float a, float b) {
  if (a < b) std::swap(a, b);
  if (b == -std::numeric_limits<float>::infinity()) return a;
  return a + std::log1p(std::exp(b - a));
}

// -log(e^-a + e^-b), computed as greco merges path costs.
float CostAdd(float a, float b) {
  const float lo = a >= b ? b : a;
  const float hi = a >= b ? a : b;
  return lo - std::log1p(std::exp(-(hi - lo)));
}

}  // namespace

TfLiteModel::TfLiteModel() = default;
TfLiteModel::~TfLiteModel() = default;

bool TfLiteModel::Load(const std::string& path, int num_threads,
                       std::string* error) {
  model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!model_) {
    *error = "cannot load TFLite model " + path;
    return false;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // libsoda's float BATCH_MATMUL accumulates in order without fused
  // multiply-adds, like TFLite's reference kernel. The optimized kernel's
  // rounding differs, which the encoder's dynamic int8 quantization amplifies.
  resolver.AddBuiltin(tflite::BuiltinOperator_BATCH_MATMUL,
                      tflite::ops::builtin::Register_BATCH_MATMUL_REF(), 1, 4);
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder.SetNumThreads(num_threads);
  if (builder(&interpreter_) != kTfLiteOk || !interpreter_) {
    *error = "cannot build interpreter for " + path;
    return false;
  }
  return true;
}

bool StreamingEncoder::Init(const std::string& path, int num_threads,
                            std::string* error) {
  if (!model_.Load(path, num_threads, error)) return false;
  tflite::Interpreter* interp = model_.interpreter();
  const auto& inputs = interp->inputs();
  const auto& outputs = interp->outputs();
  if (inputs.size() < 2 || outputs.size() < 2) {
    *error = "unexpected encoder signature";
    return false;
  }
  input_tensor_ = inputs[0];
  output_tensor_ = outputs[0];
  input_dim_ = Dim(interp->tensor(input_tensor_), 2);
  output_dim_ = Dim(interp->tensor(output_tensor_), 2);
  // ".../previous_state.NN" feeds back from ".../next_state.NN". Some
  // encoders also take the recognition domain as an int32 "domain_id".
  auto state_index = [](const char* name) -> std::string {
    const std::string n = name ? name : "";
    const size_t dot = n.rfind('.');
    const bool state = n.find("previous_state") != std::string::npos ||
                       n.find("next_state") != std::string::npos;
    return state && dot != std::string::npos ? n.substr(dot + 1) : std::string();
  };
  std::map<std::string, int> next_states;
  for (size_t i = 1; i < outputs.size(); ++i) {
    const std::string k = state_index(interp->tensor(outputs[i])->name);
    if (!k.empty()) next_states[k] = outputs[i];
  }
  const bool by_name = next_states.size() == outputs.size() - 1;
  for (size_t i = 1; i < inputs.size(); ++i) {
    const TfLiteTensor* in = interp->tensor(inputs[i]);
    if (in->name && std::string(in->name) == "domain_id") {
      domain_tensor_ = inputs[i];
      continue;
    }
    int output = -1;
    if (by_name) {
      auto it = next_states.find(state_index(in->name));
      if (it != next_states.end()) output = it->second;
    } else if (i < outputs.size()) {
      output = outputs[i];
    }
    if (output < 0) {
      *error = std::string("encoder input ") + (in->name ? in->name : "?") + " has no state output";
      return false;
    }
    State s{inputs[i], output, {}};
    const TfLiteTensor* out = interp->tensor(s.output_tensor);
    if (in->type != out->type || in->bytes != out->bytes) {
      *error = "encoder state " + std::to_string(i) + " shape mismatch";
      return false;
    }
    s.data.assign(in->bytes, 0);
    states_.push_back(std::move(s));
  }
  batch_time_ = 0;
  return true;
}

void StreamingEncoder::Reset() {
  for (State& s : states_) std::fill(s.data.begin(), s.data.end(), 0);
}

bool StreamingEncoder::Encode(const std::vector<std::vector<float>>& frames,
                              std::vector<std::vector<float>>* out,
                              std::string* error) {
  if (frames.empty()) return true;
  tflite::Interpreter* interp = model_.interpreter();
  const int t = static_cast<int>(frames.size());
  if (t != batch_time_) {
    interp->ResizeInputTensor(input_tensor_, {1, t, input_dim_});
    if (interp->AllocateTensors() != kTfLiteOk) {
      *error = "encoder tensor allocation failed";
      return false;
    }
    batch_time_ = t;
  }
  float* in = interp->typed_tensor<float>(input_tensor_);
  for (int i = 0; i < t; ++i) {
    std::memcpy(in + i * input_dim_, frames[i].data(), input_dim_ * sizeof(float));
  }
  for (const State& s : states_) {
    std::memcpy(interp->tensor(s.input_tensor)->data.raw, s.data.data(), s.data.size());
  }
  if (domain_tensor_ >= 0) interp->typed_tensor<int32_t>(domain_tensor_)[0] = domain_id_;
  if (interp->Invoke() != kTfLiteOk) {
    *error = "encoder invocation failed";
    return false;
  }
  for (State& s : states_) {
    std::memcpy(s.data.data(), interp->tensor(s.output_tensor)->data.raw, s.data.size());
  }
  const TfLiteTensor* o = interp->tensor(output_tensor_);
  const int out_t = Dim(o, 1);
  const float* od = o->data.f;
  for (int i = 0; i < out_t; ++i) {
    out->emplace_back(od + i * output_dim_, od + (i + 1) * output_dim_);
  }
  return true;
}

bool PredictionNetwork::Init(const std::string& path, int num_threads,
                             std::string* error) {
  if (!model_.Load(path, num_threads, error)) return false;
  tflite::Interpreter* interp = model_.interpreter();
  const size_t inputs = interp->inputs().size();
  recurrent_ = inputs > 2;
  if (inputs < 2 || (recurrent_ && interp->outputs().size() != inputs) ||
      interp->AllocateTensors() != kTfLiteOk) {
    *error = "unexpected prediction network signature";
    return false;
  }
  const TfLiteTensor* out = interp->tensor(interp->outputs()[0]);
  output_dim_ = Dim(out, out->dims->size - 1);
  batch_ = recurrent_ ? Dim(interp->tensor(interp->inputs()[0]), 0) : 1;
  return true;
}

const std::vector<float>& PredictionNetwork::Get(int prev_label, int label) {
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(prev_label)) << 32) |
                       static_cast<uint32_t>(label);
  auto it = cache_.find(key);
  if (it != cache_.end()) return it->second;
  tflite::Interpreter* interp = model_.interpreter();
  interp->typed_tensor<int32_t>(interp->inputs()[0])[0] = label;
  interp->typed_tensor<int32_t>(interp->inputs()[1])[0] = prev_label;
  interp->Invoke();
  const float* o = interp->typed_tensor<float>(interp->outputs()[0]);
  return cache_.emplace(key, std::vector<float>(o, o + output_dim_)).first->second;
}

void PredictionNetwork::Outputs(std::vector<Hypothesis>& hyps,
                                std::vector<const std::vector<float>*>* out) {
  out->clear();
  if (!recurrent_) {
    for (const Hypothesis& h : hyps) out->push_back(&Get(h.prev_label, h.last_label));
    return;
  }
  tflite::Interpreter* interp = model_.interpreter();
  const auto& inputs = interp->inputs();
  const auto& outputs = interp->outputs();
  std::vector<Hypothesis*> todo;
  for (Hypothesis& h : hyps) {
    if (!h.decoder_state) todo.push_back(&h);
  }
  // Rows are independent; unused rows of a batch repeat the first.
  for (size_t b = 0; b < todo.size(); b += batch_) {
    const size_t n = std::min(todo.size() - b, static_cast<size_t>(batch_));
    for (int row = 0; row < batch_; ++row) {
      const Hypothesis& h = *todo[b + (row < static_cast<int>(n) ? row : 0)];
      interp->typed_tensor<int32_t>(inputs[0])[row] = h.last_label;
      for (size_t i = 1; i < inputs.size(); ++i) {
        TfLiteTensor* t = interp->tensor(inputs[i]);
        const size_t dim = t->bytes / sizeof(float) / batch_;
        float* dst = t->data.f + row * dim;
        if (h.parent_state) {
          std::memcpy(dst, h.parent_state->states[i - 1].data(), dim * sizeof(float));
        } else {
          std::fill(dst, dst + dim, 0.0f);
        }
      }
    }
    interp->Invoke();
    for (size_t row = 0; row < n; ++row) {
      auto state = std::make_shared<DecoderState>();
      const float* o = interp->tensor(outputs[0])->data.f + row * output_dim_;
      state->output.assign(o, o + output_dim_);
      for (size_t i = 1; i < outputs.size(); ++i) {
        const TfLiteTensor* t = interp->tensor(outputs[i]);
        const size_t dim = t->bytes / sizeof(float) / batch_;
        state->states.emplace_back(t->data.f + row * dim, t->data.f + (row + 1) * dim);
      }
      todo[b + row]->decoder_state = std::move(state);
    }
  }
  for (const Hypothesis& h : hyps) out->push_back(&h.decoder_state->output);
}

bool JointNetwork::Init(const std::string& path, int num_threads,
                        std::string* error, bool logits) {
  if (!model_.Load(path, num_threads, error)) return false;
  logits_ = logits;
  tflite::Interpreter* interp = model_.interpreter();
  if (interp->inputs().size() != 2 || interp->AllocateTensors() != kTfLiteOk) {
    *error = "unexpected joint network signature";
    return false;
  }
  const TfLiteTensor* enc = interp->tensor(interp->inputs()[0]);
  rank3_ = enc->dims->size == 3;
  encoder_dim_ = Dim(enc, enc->dims->size - 1);
  const TfLiteTensor* dec = interp->tensor(interp->inputs()[1]);
  decoder_dim_ = Dim(dec, dec->dims->size - 1);
  output_dim_ = Dim(interp->tensor(interp->outputs()[0]), 1);
  batch_ = rank3_ ? Dim(dec, 0) : 1;
  return true;
}

void JointNetwork::Run(const float* encoder_frame,
                       const std::vector<const std::vector<float>*>& decoder_outputs,
                       std::vector<float>* probs) {
  tflite::Interpreter* interp = model_.interpreter();
  const int b = static_cast<int>(decoder_outputs.size());
  if (rank3_) {
    // A fixed batch of decoder outputs against one encoder frame; unused
    // rows repeat the first.
    float* enc = interp->typed_tensor<float>(interp->inputs()[0]);
    float* dec = interp->typed_tensor<float>(interp->inputs()[1]);
    std::memcpy(enc, encoder_frame, encoder_dim_ * sizeof(float));
    probs->resize(static_cast<size_t>(b) * output_dim_);
    for (int start = 0; start < b; start += batch_) {
      const int n = std::min(batch_, b - start);
      for (int i = 0; i < batch_; ++i) {
        std::memcpy(dec + i * decoder_dim_, decoder_outputs[start + (i < n ? i : 0)]->data(),
                    decoder_dim_ * sizeof(float));
      }
      interp->Invoke();
      const float* o = interp->typed_tensor<float>(interp->outputs()[0]);
      std::copy(o, o + n * output_dim_, probs->data() + start * output_dim_);
    }
    return;
  }
  if (b != batch_) {
    interp->ResizeInputTensor(interp->inputs()[0], {b, encoder_dim_});
    interp->ResizeInputTensor(interp->inputs()[1], {b, decoder_dim_});
    interp->AllocateTensors();
    batch_ = b;
  }
  float* enc = interp->typed_tensor<float>(interp->inputs()[0]);
  float* dec = interp->typed_tensor<float>(interp->inputs()[1]);
  for (int i = 0; i < b; ++i) {
    std::memcpy(enc + i * encoder_dim_, encoder_frame, encoder_dim_ * sizeof(float));
    std::memcpy(dec + i * decoder_dim_, decoder_outputs[i]->data(),
                decoder_dim_ * sizeof(float));
  }
  interp->Invoke();
  const float* o = interp->typed_tensor<float>(interp->outputs()[0]);
  probs->assign(o, o + b * output_dim_);
}

BeamSearchOptions SearchOptions(const DecoderConfig& d) {
  BeamSearchOptions bo;
  bo.max_arcs = d.max_arcs;
  bo.min_arcs = d.min_arcs;
  bo.local_beam = d.local_beam;
  bo.histogram_scale = d.histogram_scale;
  bo.histogram_buckets = d.histogram_buckets;
  bo.word_pen = d.word_pen;
  bo.max_label_cost = d.max_model_cost;
  bo.max_expansions = std::max(1, d.max_expansions);
  bo.temperature = d.smoothing_temperature;
  bo.max_model_cost_to_influence = d.max_model_cost_to_influence;
  bo.prune_on_min_cost = d.prune_on_min_cost;
  return bo;
}

BeamSearch::BeamSearch(PredictionNetwork* decoder, JointNetwork* joint,
                       const BeamSearchOptions& options,
                       const DecoderConfig& config)
    : decoder_(decoder), joint_(joint), options_(options), config_(config) {
  Reset();
}

void BeamSearch::Reset() {
  Hypothesis h;
  h.prev_label = config_.decoder_initial_value;
  h.last_label = config_.sos_id;
  beam_.assign(1, h);
  ended_.clear();
  best_ = 0;
  best_cost_ = kNoCost;  // The first frame has no beam.
  frame_ = 0;
}

void BeamSearch::Resume() {
  Hypothesis h;
  const Hypothesis& best = Best();
  h.prev_label = best.prev_label;
  h.last_label = best.last_label;
  h.decoder_state = best.decoder_state;
  beam_.assign(1, h);
  ended_.clear();
  best_ = 0;
  best_cost_ = kNoCost;
  frame_ = 0;
}

void BeamSearch::Score(const std::vector<float>& encoder_frame,
                       std::vector<Hypothesis>& hyps,
                       std::vector<std::vector<float>>* costs) {
  const int vocab = joint_->output_dim();
  const float inv_temp = 1.0f / options_.temperature;
  std::vector<const std::vector<float>*> dec;
  decoder_->Outputs(hyps, &dec);
  joint_->Run(encoder_frame.data(), dec, &probs_);
  costs->resize(hyps.size());
  for (size_t i = 0; i < hyps.size(); ++i) {
    // Temperature smoothing: -log of p^(1/T), renormalized.
    const float* p = probs_.data() + i * vocab;
    std::vector<float>& c = (*costs)[i];
    c.resize(vocab);
    float norm = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < vocab; ++k) {
      c[k] = (joint_->logits() ? p[k] : std::log(std::max(p[k], 1e-30f))) * inv_temp;
      norm = LogAdd(norm, c[k]);
    }
    for (int k = 0; k < vocab; ++k) c[k] = norm - c[k];
  }
}

float BeamSearch::HistogramThreshold(const std::vector<Hypothesis>& hyps, float best,
                                     float threshold) const {
  const int max_arcs = options_.max_arcs;
  const int min_arcs = options_.min_arcs;
  if (static_cast<int>(hyps.size()) <= max_arcs && min_arcs <= 0) return threshold;
  // Bins of width histogram_scale / histogram_buckets, rounded, from `best`.
  const int buckets = options_.histogram_buckets;
  const float width = options_.histogram_scale / static_cast<float>(buckets);
  const float scale = static_cast<float>(buckets) / options_.histogram_scale;
  std::vector<int> counts(buckets, 0);
  for (const Hypothesis& h : hyps) {
    const float b = (PruningCost(h) - best) * scale + 0.5f;
    int k = 0;
    if (std::isnan(b) || std::trunc(b) < -2147483648.0f) {
      k = 0;
    } else if (b >= 2147483648.0f) {
      continue;
    } else {
      k = std::max(0, static_cast<int>(b));
    }
    if (k < buckets) ++counts[k];
  }
  // The first bin where more than `limit` hypotheses have been counted.
  auto bin = [&](int limit) {
    int total = 0;
    for (int k = 0; k < buckets; ++k) {
      total += counts[k];
      if (total > limit) return k;
    }
    return buckets;
  };
  if (static_cast<int>(hyps.size()) > max_arcs) {
    threshold = std::min(threshold, best + static_cast<float>(bin(max_arcs)) * width);
  }
  if (min_arcs >= 1) {
    threshold = std::max(threshold, best + static_cast<float>(bin(min_arcs)) * width);
  }
  return threshold;
}

void BeamSearch::Advance(const std::vector<float>& encoder_frame) {
  const int vocab = joint_->output_dim();
  const float max_cost = options_.max_label_cost > 0 ? options_.max_label_cost : kNoCost;
  // Hypotheses expand labels while within local_beam of the previous frame's
  // best; each round's new hypotheses are pruned relative to the best cost
  // seen on this frame.
  const float threshold = best_cost_ + options_.local_beam;
  float best = best_cost_;
  std::vector<Hypothesis> round = std::move(beam_);
  const std::vector<Hypothesis> entering = round;
  std::vector<Hypothesis> blanks;  // Successors into the next frame.
  std::vector<std::vector<float>> costs;
  for (int expansion = 0; expansion < options_.max_expansions && !round.empty(); ++expansion) {
    Score(encoder_frame, round, &costs);
    const bool expand = expansion < options_.max_expansions - 1;
    std::vector<Hypothesis> next;
    for (size_t i = 0; i < round.size(); ++i) {
      const Hypothesis& h = round[i];
      const std::vector<float>& c = costs[i];
      if (c[0] < kNoCost) {
        Hypothesis b = h;
        b.cost = c[0] + h.cost;
        b.model_cost = c[0] + h.model_cost;
        blanks.push_back(std::move(b));
      }
      if (!expand || !(PruningCost(h) < threshold)) continue;
      // Hypotheses that entered the frame and extend h: their labels after
      // h's carry alignments still alive in libsoda's prefix tree.
      std::vector<const Hypothesis*> extensions;
      for (const Hypothesis& g : entering) {
        if (g.labels.size() > h.labels.size() &&
            std::equal(h.labels.begin(), h.labels.end(), g.labels.begin())) {
          extensions.push_back(&g);
        }
      }
      if (config_.eos_id > 0 && h.last_label == config_.eos_id) continue;  // Ended.
      for (int k = 1; k < vocab; ++k) {
        float cost = c[k];
        if (k == config_.sos_id) continue;
        if (k == config_.eos_id) {
          // </S> can only end an utterance that has content.
          if (!(config_.eos_penalty > 0) || h.labels.empty()) continue;
          cost -= std::log(config_.eos_penalty);
        }
        float bias = 0.0f;
        int bias_state = 0;
        if (biaser_ != nullptr && !biaser_->empty()) {
          bias = biaser_->Step(h.bias_state, k, &bias_state);
          if (!(cost <= options_.max_model_cost_to_influence)) bias = 0.0f;
        }
        const float limit_cost = options_.prune_on_min_cost ? std::min(cost, cost + bias) : cost;
        if (!(cost < kNoCost) || !(limit_cost <= max_cost)) continue;
        Hypothesis e;
        e.labels.reserve(h.labels.size() + 1);
        e.labels = h.labels;
        e.labels.push_back(k);
        e.label_frames = h.label_frames;
        e.label_frames.push_back(frame_);
        e.prev_label = h.last_label;
        e.last_label = k;
        e.decoder_state = nullptr;
        e.parent_state = h.decoder_state;
        e.model_cost = (h.model_cost + cost) + options_.word_pen;
        e.cost = (h.cost + cost) + options_.word_pen + bias;
        e.bias_state = bias_state;
        // Another alignment of these labels ended the previous frame; the
        // merged hypothesis keeps that earlier alignment if it survived the
        // previous frame's pruning (as libsoda's word times show), and its
        // cost either way.
        auto it = ended_.find(e.labels);
        if (it != ended_.end()) {
          if (it->second.kept) e.label_frames = it->second.label_frames;
          e.cost = CostAdd(e.cost, it->second.cost);
          e.model_cost = CostAdd(e.model_cost, it->second.model_cost);
        } else {
          // Otherwise a surviving longer hypothesis may still hold these
          // labels' node from an earlier frame, which the expansion reuses
          // with its time (ja-JP: `▁g ma i l ▁ co m` takes the bare ▁ of a
          // `▁g ma i l ▁ .` emitted 7 frames before `co`).
          for (const Hypothesis* g : extensions) {
            if (g->labels[h.labels.size()] != k) continue;
            e.label_frames.assign(g->label_frames.begin(), g->label_frames.begin() + e.labels.size());
            break;
          }
        }
        best = std::min(best, PruningCost(e));
        next.push_back(std::move(e));
      }
    }
    const float t = HistogramThreshold(next, best, threshold);
    round.clear();
    for (Hypothesis& h : next) {
      if (PruningCost(h) < t) round.push_back(std::move(h));
    }
  }

  // Blank successors with the same labels merge into the cheaper one (the
  // first on ties).
  std::map<std::vector<int>, size_t> index;
  float next_best = kNoCost;
  for (Hypothesis& b : blanks) {
    next_best = std::min(next_best, PruningCost(b));
    auto [it, inserted] = index.emplace(b.labels, beam_.size());
    if (inserted) {
      beam_.push_back(std::move(b));
    } else if (b.cost < beam_[it->second].cost) {
      beam_[it->second] = std::move(b);
    }
  }
  ended_.clear();
  for (const Hypothesis& h : beam_) ended_.emplace(h.labels, Ended{h.cost, h.model_cost, h.label_frames});
  const float t = HistogramThreshold(beam_, next_best, next_best + options_.local_beam);
  size_t b = 0;  // The best hypothesis, if every one is pruned.
  for (size_t i = 1; i < beam_.size(); ++i) {
    if (beam_[i].cost < beam_[b].cost) b = i;
  }
  if (!(PruningCost(beam_[b]) < t)) {
    // greco drops every hypothesis when more than max_arcs fall in the best
    // one's histogram bin; keep the best.
    beam_ = {std::move(beam_[b])};
  } else {
    beam_.erase(std::remove_if(beam_.begin(), beam_.end(),
                               [&](const Hypothesis& h) { return !(PruningCost(h) < t); }),
                beam_.end());
  }
  for (const Hypothesis& h : beam_) ended_[h.labels].kept = true;
  best_ = 0;
  for (size_t i = 1; i < beam_.size(); ++i) {
    if (beam_[i].cost < beam_[best_].cost) best_ = i;
  }
  best_cost_ = next_best;
  ++frame_;
}

}  // namespace soda
