#include "langid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

#include "proto_wire.h"

namespace soda {

namespace {

constexpr int kSampleRate = 16000;

}  // namespace

void AutoGainControl::Init(const AgcConfig& config, int sample_rate) {
  config_ = config;
  decay_ = std::exp2(-1.0f / (sample_rate * config.half_life_sec));
  const int delay = config.delay_ms * sample_rate / 1000;
  // The level settles over the attack window when one is given.
  const int window = config.attack_window_ms > 0
                         ? config.attack_window_ms * sample_rate / 1000
                         : delay;
  attack_ = window < 1 ? 1.0f : 1.0f / window;
  delay_line_.assign(delay, 0.0f);
  Reset();
}

void AutoGainControl::Reset() {
  peak_ = level_ = config_.floor;
  std::fill(delay_line_.begin(), delay_line_.end(), 0.0f);
  delay_pos_ = 0;
}

void AutoGainControl::Process(float* samples, size_t n) {
  if (!config_.activated) return;
  for (size_t i = 0; i < n; ++i) {
    const float x = samples[i];
    peak_ *= decay_;
    const float magnitude = std::fabs(x) / 32768.0f;
    // As greco does: a louder sample sets the peak even below the floor.
    if (peak_ < magnitude) {
      peak_ = magnitude;
    } else if (peak_ < config_.floor) {
      peak_ = config_.floor;
    }
    level_ += attack_ * ((peak_ - level_) + (peak_ - level_));
    const float gain =
        (config_.level_target + level_ * 0.1f) / (level_ + config_.curvature) *
        (config_.gain_scale /
         (std::exp(config_.lower_target_transition * (config_.lower_level_target - level_)) +
          1.0f));
    // The gain applies to the sample delay_ms back.
    float delayed = x;
    if (!delay_line_.empty()) {
      delayed = delay_line_[delay_pos_];
      delay_line_[delay_pos_] = x;
      delay_pos_ = (delay_pos_ + 1) % delay_line_.size();
    }
    samples[i] = gain * delayed;
  }
}

std::unique_ptr<LanguageIdentifier> LanguageIdentifier::Create(
    const std::string& models_dir, std::string* error) {
  std::unique_ptr<LanguageIdentifier> l(new LanguageIdentifier());
  if (!l->Init(models_dir, error)) return nullptr;
  return l;
}

bool LanguageIdentifier::Init(const std::string& models_dir, std::string* error) {
  if (!LoadLangIdConfig(models_dir + "/configs/LANGUAGE_IDENTIFICATION.config", &config_,
                        error)) {
    return false;
  }
  if (config_.agc.activated && config_.agc.estimator != 3) {
    *error = "unsupported AGC gain estimator " + std::to_string(config_.agc.estimator);
    return false;
  }
  agc_.Init(config_.agc, kSampleRate);
  const FrameStackConfig& stack = config_.frontend.stack1;
  const int dim = stack.input_size * (stack.left_context + 1 + stack.right_context);
  if (!frontend_.Init(config_.frontend, kSampleRate, std::vector<float>(dim, 0.0f),
                      std::vector<float>(dim, 1.0f), error)) {
    return false;
  }
  frontend_.set_stacked_tap(&stacked_);
  frame_ms_ = config_.frontend.frame_step_ms * stack.subsample;
  if (!vad_.Init(models_dir, config_.vad, error)) return false;
  endpointer_.Init(config_.vad.endpointer);

  if (!model_.Load(models_dir + "/" + config_.model_file, 1, error)) return false;
  tflite::Interpreter* interp = model_.interpreter();
  if (interp->AllocateTensors() != kTfLiteOk) {
    *error = "cannot allocate the language identification model";
    return false;
  }
  const auto& inputs = interp->inputs();
  const auto& outputs = interp->outputs();
  if (inputs.size() != outputs.size() || inputs.empty()) {
    *error = "unexpected language identification model signature";
    return false;
  }
  input_tensor_ = inputs[0];
  output_tensor_ = outputs[0];
  const TfLiteTensor* in = interp->tensor(input_tensor_);
  const TfLiteTensor* out = interp->tensor(output_tensor_);
  input_dim_ = in->dims->data[in->dims->size - 1];
  num_outputs_ = out->dims->data[out->dims->size - 1];
  if (in->dims->data[0] != config_.frame_batch_size || input_dim_ != dim) {
    *error = "language identification model input does not match its config";
    return false;
  }
  // Recurrent states: input i pairs with output i.
  for (size_t i = 1; i < inputs.size(); ++i) {
    StateTensor s{inputs[i], outputs[i], {}};
    if (interp->tensor(s.input)->bytes != interp->tensor(s.output)->bytes) {
      *error = "language identification state " + std::to_string(i) + " mismatch";
      return false;
    }
    s.data.assign(interp->tensor(s.input)->bytes, 0);
    states_.push_back(std::move(s));
  }

  // greco.ApplicationSpecificParams: profile (2) {lang (1), node (2)}.
  std::string params;
  const std::string params_path = models_dir + "/" + config_.application_params_file;
  if (!ReadFile(params_path, &params)) {
    *error = "cannot read " + params_path;
    return false;
  }
  node_language_.assign(num_outputs_, "");
  ProtoReader r(params);
  while (r.Next()) {
    if (r.field() != 2 || r.wire_type() != kLengthDelimited) continue;
    ProtoReader p(r.bytes());
    std::string lang;
    std::vector<int> nodes;
    while (p.Next()) {
      if (p.field() == 1) lang = p.str();
      if (p.field() == 2) nodes.push_back(static_cast<int>(p.int64()));
    }
    for (int n : nodes) {
      if (n >= 0 && n < num_outputs_) node_language_[n] = lang;
    }
  }
  return true;
}

void LanguageIdentifier::Reset() {
  agc_.Reset();
  frontend_.Reset();
  vad_.Reset();
  endpointer_.Reset();
  stacked_.clear();
  waiting_.clear();
  frames_seen_ = frames_decided_ = 0;
  in_segment_ = false;
  held_.clear();
  silence_frames_ = 0;
  batch_.clear();
  ResetClassifier();
  last_argmax_ = -1;
  continuity_ = 0;
  scored_ = false;
  last_event_.reset();
  pending_.clear();
}

void LanguageIdentifier::Finish() {
  // libsoda always reports once more at the end: it classifies a partial batch
  // filled with copies of its last frame, or else repeats its last event.
  if (!batch_.empty()) {
    while (static_cast<int>(batch_.size()) < config_.frame_batch_size) batch_.push_back(batch_.back());
    RunClassifier();
  } else if (last_event_) {
    pending_.push_back(*last_event_);
  }
  Deliver();
}

void LanguageIdentifier::Deliver() {
  bool later_confident = false;
  for (auto it = pending_.rbegin(); it != pending_.rend(); ++it) {
    it->superseded = it->confidence_level == 3 && later_confident;
    if (it->confidence_level == 3) later_confident = true;
  }
  if (callback_) {
    for (const LanguageEvent& e : pending_) callback_(e);
  }
  pending_.clear();
}

void LanguageIdentifier::ResetClassifier() {
  for (StateTensor& s : states_) std::fill(s.data.begin(), s.data.end(), 0);
  frames_since_reset_ = 0;
}

void LanguageIdentifier::Accept(const float* samples, size_t n) {
  std::vector<float> audio(samples, samples + n);
  agc_.Process(audio.data(), audio.size());
  std::vector<std::vector<float>> unused;
  frontend_.Accept(audio.data(), audio.size(), &unused);
  std::vector<bool> decisions;
  for (auto& frame : stacked_) {
    // The endpointer decides each frame with a delay; frames wait for it.
    endpointer_.Process(vad_.Process(frame));
    waiting_.push_back(std::move(frame));
    ++frames_seen_;
    decisions.clear();
    endpointer_.TakeDecisions(&decisions);
    for (bool speech : decisions) {
      if (waiting_.empty()) break;
      ProcessFrame(waiting_.front(), speech);
      waiting_.erase(waiting_.begin());
      ++frames_decided_;
    }
  }
  stacked_.clear();
  Deliver();
}

void LanguageIdentifier::ProcessFrame(const std::vector<float>& features, bool speech) {
  // greco's SegmenterStream: silence after speech is held back. If speech
  // resumes within reset_interval_ms, the held frames are classified too and
  // the classifier keeps its state; a longer silence drops them and resets.
  if (speech) {
    if (!in_segment_) {
      // A partial batch left from the last segment: libsoda classifies it
      // without reporting the result, then resets the classifier.
      batch_.clear();
      if (config_.reset_segment_state) ResetClassifier();
    }
    in_segment_ = true;
    silence_frames_ = 0;
    for (auto& held : held_) Classify(std::move(held));
    held_.clear();
    Classify(features);
    return;
  }
  if (!in_segment_) return;
  held_.push_back(features);
  if (++silence_frames_ * frame_ms_ > config_.reset_interval_ms) {
    // A partial batch stays until speech resumes or the audio ends.
    in_segment_ = false;
    held_.clear();
  }
}

void LanguageIdentifier::Classify(std::vector<float> features) {
  batch_.push_back(std::move(features));
  if (static_cast<int>(batch_.size()) >= config_.frame_batch_size) RunClassifier();
}

void LanguageIdentifier::RunClassifier() {
  // reset_every_n_frame counts classifier calls, not frames.
  if (config_.reset_every_n_frame > 0 &&
      frames_since_reset_ >= config_.reset_every_n_frame * config_.frame_batch_size) {
    ResetClassifier();
  }
  tflite::Interpreter* interp = model_.interpreter();
  float* in = interp->typed_tensor<float>(input_tensor_);
  for (size_t i = 0; i < batch_.size(); ++i) {
    std::memcpy(in + i * input_dim_, batch_[i].data(), input_dim_ * sizeof(float));
  }
  frames_since_reset_ += static_cast<long>(batch_.size());
  batch_.clear();
  for (const StateTensor& s : states_) {
    std::memcpy(interp->tensor(s.input)->data.raw, s.data.data(), s.data.size());
  }
  if (interp->Invoke() != kTfLiteOk) return;
  for (StateTensor& s : states_) {
    std::memcpy(s.data.data(), interp->tensor(s.output)->data.raw, s.data.size());
  }
  Score(interp->tensor(output_tensor_)->data.f);
}

void LanguageIdentifier::Score(const float* probs) {
  // libsoda reports nothing for the first call of the stream.
  if (!scored_) {
    scored_ = true;
    return;
  }
  const int n = num_outputs_;
  std::vector<float> p(probs, probs + n);
  for (float& v : p) {
    if (std::isnan(v)) v = 0.0f;
  }
  // Confidence features (langid_confidence_lib.cc).
  const int argmax = static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
  if (argmax != last_argmax_) {
    last_argmax_ = argmax;
    continuity_ = 0;
  } else {
    ++continuity_;
  }
  float negentropy = static_cast<float>(std::log(static_cast<double>(n)));
  for (float v : p) {
    if (v > 1e-10f) negentropy += v * std::log(v);
  }
  std::vector<float> sorted = p;
  std::partial_sort(sorted.begin(), sorted.begin() + std::min(2, n), sorted.end(),
                    std::greater<float>());
  const float top = sorted[0];
  const float delta = n > 1 ? sorted[0] - sorted[1] : sorted[0];
  const LangIdConfidenceConfig& c = config_.confidence;
  float continuity = static_cast<float>(continuity_);
  if (c.continuity_upper_bound >= 0.0f) continuity = std::min(continuity, c.continuity_upper_bound);
  const float score = top * c.prob_weight + delta * c.prob_delta_weight +
                      negentropy * c.negentropy_weight + continuity * c.continuity_weight;

  LanguageEvent e;
  if (!c.has_thresholds) {
    e.confidence_level = 0;
  } else if (node_language_[argmax].empty() || score < c.confident_threshold) {
    e.confidence_level = 1;
  } else {
    e.confidence_level = score >= c.highly_confident_threshold ? 3 : 2;
  }
  // The language with the highest-scoring mapped node.
  int best = -1;
  for (int k = 0; k < n; ++k) {
    if (!node_language_[k].empty() && (best < 0 || p[k] > p[best])) best = k;
  }
  if (best < 0) return;
  e.language = node_language_[best];
  e.audio_ms = static_cast<int64_t>(frames_decided_ * frame_ms_);
  last_event_ = e;
  pending_.push_back(e);
}

}  // namespace soda
