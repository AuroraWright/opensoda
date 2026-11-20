#include "speaker_turn.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

namespace soda {

namespace {

constexpr int kSampleRate = 16000;

std::string Lowercase(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// Bytewise edit distance.
int Levenshtein(const std::string& a, const std::string& b) {
  std::vector<int> row(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) row[j] = static_cast<int>(j);
  for (size_t i = 1; i <= a.size(); ++i) {
    int diagonal = row[0];
    row[0] = static_cast<int>(i);
    for (size_t j = 1; j <= b.size(); ++j) {
      const int up = row[j];
      row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] != b[j - 1] ? 1 : 0)});
      diagonal = up;
    }
  }
  return row[b.size()];
}

// Of the words around `center`, the one whose surroundings best match the
// turn's context: the lowercase raw text before it (with its leading space)
// against the turn's left context, and from it on against the right one.
size_t AlignToContext(const SpeakerTurn& turn, const std::vector<SpeakerTurnWord>& words,
                      size_t center) {
  auto piece = [&](size_t k) {
    return (words[k].leading_space ? " " : "") + Lowercase(words[k].raw);
  };
  const size_t left_size = turn.left.size(), right_size = turn.right.size();
  int best = std::numeric_limits<int>::max();
  size_t best_index = center;
  for (size_t i = center == 0 ? 0 : center - 1; i <= center + 1 && i < words.size(); ++i) {
    std::string left = words[i].leading_space ? " " : "";
    for (size_t k = i; k > 0 && left.size() < left_size;) left = piece(--k) + left;
    if (left.size() > left_size) left = left.substr(left.size() - left_size);
    std::string right;
    for (size_t k = i; k < words.size() && right.size() < right_size; ++k) right += piece(k);
    if (right.size() > right_size) right.resize(right_size);
    const int score = Levenshtein(left, turn.left.substr(left_size - left.size())) +
                      Levenshtein(right, turn.right.substr(0, right.size()));
    if (score < best || (i == center && score == best)) {
      best = score;
      best_index = i;
    }
  }
  return best_index;
}

}  // namespace

std::vector<bool> PlaceSpeakerTurns(const std::vector<SpeakerTurn>& turns,
                                    const std::vector<SpeakerTurnWord>& words) {
  const size_t n = words.size();
  std::vector<bool> change(n, false);
  if (n == 0) return change;
  // Each turn's search starts a word before where the last one's ended.
  size_t from = 0;
  for (const SpeakerTurn& turn : turns) {
    const int64_t t = turn.time_ms * 1000;
    size_t center = from;  // The word starting closest to the turn.
    size_t after = from;   // The first word starting at or after it.
    float closest = std::numeric_limits<float>::max();
    for (; after < n; ++after) {
      const int64_t start = words[after].start_ms * 1000;
      const float distance = static_cast<float>(start >= t ? start - t : t - start);
      if (closest > distance) {
        closest = distance;
        center = after;
      }
      if (start >= t) break;
    }
    change[AlignToContext(turn, words, std::min(center, n - 1))] = true;
    from = after > 0 ? after - 1 : 0;
  }
  // Marks snap to sentence starts: one on a sentence's second word moves to
  // its first, and one on its last word to the next one's first (or is
  // dropped at the end of the result).
  auto ends_sentence = [&](size_t k) {
    const std::string& w = words[k].text;
    return !w.empty() && std::strchr(".!?", w.back()) != nullptr;
  };
  for (size_t k = 1; k < n; ++k) {
    if (ends_sentence(k - 1) || !change[k]) continue;
    if (k == 1 || ends_sentence(k - 2)) {
      change[k] = false;
      change[k - 1] = true;
    } else if (ends_sentence(k)) {
      change[k] = false;
      if (k + 1 < n) change[k + 1] = true;
    }
  }
  return change;
}

std::unique_ptr<SpeakerTurnDetector> SpeakerTurnDetector::Create(const std::string& models_dir,
                                                                 std::string* error) {
  std::unique_ptr<SpeakerTurnDetector> d(new SpeakerTurnDetector());
  if (!d->Init(models_dir, error)) return nullptr;
  return d;
}

bool SpeakerTurnDetector::Init(const std::string& models_dir, std::string* error) {
  if (!LoadSpeakerTurnConfig(models_dir + "/configs/ONDEVICE_SPEAKER_TURN_DETECTION.config",
                             &config_, error)) {
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
  if (!LoadMeanStdDev(models_dir + "/" + config_.mean_stddev_file, &mean_, &stddev_, error)) {
    return false;
  }
  if (static_cast<int>(mean_.size()) != dim) {
    *error = "speaker turn mean/stddev does not match the front end";
    return false;
  }

  if (!encoder_.Load(models_dir + "/" + config_.encoder_file, 1, error)) return false;
  tflite::Interpreter* interp = encoder_.interpreter();
  if (interp->AllocateTensors() != kTfLiteOk) {
    *error = "cannot allocate the speaker turn encoder";
    return false;
  }
  const auto& inputs = interp->inputs();
  const auto& outputs = interp->outputs();
  if (inputs.size() != outputs.size() || inputs.empty()) {
    *error = "unexpected speaker turn encoder signature";
    return false;
  }
  const TfLiteTensor* in = interp->tensor(inputs[0]);
  const TfLiteTensor* out = interp->tensor(outputs[0]);
  input_dim_ = in->dims->data[in->dims->size - 1];
  output_dim_ = out->dims->data[out->dims->size - 1];
  if (in->dims->data[0] != config_.frame_batch_size || input_dim_ != dim) {
    *error = "speaker turn encoder input does not match its config";
    return false;
  }
  // Recurrent states (and their lengths): input i pairs with output i.
  for (size_t i = 1; i < inputs.size(); ++i) {
    StateTensor s{inputs[i], outputs[i], {}};
    if (interp->tensor(s.input)->bytes != interp->tensor(s.output)->bytes) {
      *error = "speaker turn encoder state " + std::to_string(i) + " mismatch";
      return false;
    }
    s.data.assign(interp->tensor(s.input)->bytes, 0);
    states_.push_back(std::move(s));
  }

  if (!decoder_.Init(models_dir + "/" + config_.decoder_file, 1, error) ||
      !joint_.Init(models_dir + "/" + config_.joint_file, 1, error, /*logits=*/true)) {
    return false;
  }
  // A text symbol table: "symbol<TAB>id" per line.
  std::ifstream syms(models_dir + "/" + config_.symbols_file);
  std::string line;
  while (std::getline(syms, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;
    const int id = std::atoi(line.c_str() + tab + 1);
    if (id < 0 || id > 100000) continue;
    if (id >= static_cast<int>(symbols_.size())) symbols_.resize(id + 1);
    symbols_[id] = line.substr(0, tab);
  }
  if (static_cast<int>(symbols_.size()) != joint_.output_dim() ||
      config_.speaker_change_id >= static_cast<int>(symbols_.size())) {
    *error = "speaker turn symbols do not match the joint network";
    return false;
  }
  search_ = std::make_unique<BeamSearch>(&decoder_, &joint_, SearchOptions(config_.decoder),
                                         config_.decoder);
  return true;
}

void SpeakerTurnDetector::Reset() {
  agc_.Reset();
  frontend_.Reset();
  vad_.Reset();
  endpointer_.Reset();
  stacked_.clear();
  waiting_.clear();
  batch_.clear();
  last_input_.clear();
  for (StateTensor& s : states_) std::fill(s.data.begin(), s.data.end(), 0);
  encoded_ = 0;
  decoded_ = 0;
  frames_decided_ = 0;
  segmenter_ = SegmenterState();
  segment_start_ms_ = 0;
  turns_.clear();
  search_->Reset();
}

void SpeakerTurnDetector::Accept(const float* samples, size_t n) {
  std::vector<float> audio(samples, samples + n);
  agc_.Process(audio.data(), audio.size());
  std::vector<std::vector<float>> unused;
  frontend_.Accept(audio.data(), audio.size(), &unused);
  std::vector<bool> decisions;
  for (auto& frame : stacked_) {
    // The endpointer decides each frame with a delay; frames wait for it.
    endpointer_.Process(vad_.Process(frame));
    for (int k = 0; k < input_dim_; ++k) frame[k] = (frame[k] - mean_[k]) / stddev_[k];
    waiting_.push_back(std::move(frame));
    decisions.clear();
    endpointer_.TakeDecisions(&decisions);
    for (bool speech : decisions) {
      if (waiting_.empty()) break;
      if (SegmentEnds(static_cast<int64_t>(frames_decided_ * frame_ms_), speech)) EndSegment();
      ++frames_decided_;
      Push(std::move(waiting_.front()));
      waiting_.pop_front();
    }
  }
  stacked_.clear();
}

bool SpeakerTurnDetector::SegmentEnds(int64_t t, bool speech) {
  // greco's SegmenterStream: a segment ends once the silence after its
  // speech exceeds reset_interval_ms, less and less of it from
  // impatience_start_ms on, and is cut at impatience_limit_ms.
  SegmenterState& g = segmenter_;
  if (speech) g.last_speech_ms = t;
  bool check_silence = false;
  if (g.state == SegmenterState::kIdle) {
    if (speech) g.state = SegmenterState::kSpeech;
  } else if (g.state == SegmenterState::kSpeech) {
    if (!speech && !config_.vad.endpointer.never_transition_to_silence) {
      g.state = SegmenterState::kSilence;
      check_silence = true;
    }
  } else if (speech) {
    g.state = SegmenterState::kSpeech;
  } else {
    check_silence = true;
  }
  const int64_t length = t - g.segment_start_ms;
  const int start = config_.vad.impatience_start_ms;
  const int limit = config_.vad.impatience_limit_ms;
  bool end = limit > 0 && length >= limit;
  if (check_silence && !end) {
    double required = config_.reset_interval_ms;
    if (start > 0 && limit > start && length > start) {
      required *= static_cast<double>(limit - length) / (limit - start);
    }
    end = required < static_cast<double>(t - g.last_speech_ms);
  }
  if (end) g.state = SegmenterState::kIdle;
  if (g.state == SegmenterState::kIdle) g.segment_start_ms = t;
  return end;
}

void SpeakerTurnDetector::EndSegment() {
  // The search ends where the encoder's output does: a partial batch goes to
  // the next segment. Like libsoda (enable_resume_search), the next segment
  // keeps the decoder's context.
  for (SpeakerTurn& t : SegmentTurns()) turns_.push_back(std::move(t));
  search_->Resume();
  segment_start_ms_ = static_cast<int64_t>(decoded_ * frame_ms_);
}

void SpeakerTurnDetector::Push(std::vector<float> frame) {
  last_input_ = frame;
  batch_.push_back(std::move(frame));
  if (static_cast<int>(batch_.size()) == config_.frame_batch_size) Encode();
}

void SpeakerTurnDetector::Finish() {
  // Frames still awaiting a decision end the segment; libsoda then repeats
  // the last frame and fills the batch with it.
  for (auto& frame : waiting_) Push(std::move(frame));
  waiting_.clear();
  if (last_input_.empty()) return;
  const std::vector<float> last = last_input_;
  for (int i = 0; i < config_.last_frame_repetitions; ++i) Push(last);
  if (!batch_.empty()) {
    while (static_cast<int>(batch_.size()) < config_.frame_batch_size) batch_.push_back(last);
    Encode();
  }
  last_input_.clear();
}

void SpeakerTurnDetector::Encode() {
  tflite::Interpreter* interp = encoder_.interpreter();
  float* in = interp->typed_tensor<float>(interp->inputs()[0]);
  for (size_t i = 0; i < batch_.size(); ++i) {
    std::memcpy(in + i * input_dim_, batch_[i].data(), input_dim_ * sizeof(float));
  }
  const int frames = static_cast<int>(batch_.size());
  batch_.clear();
  for (const StateTensor& s : states_) {
    std::memcpy(interp->tensor(s.input)->data.raw, s.data.data(), s.data.size());
  }
  if (interp->Invoke() != kTfLiteOk) return;
  for (StateTensor& s : states_) {
    std::memcpy(s.data.data(), interp->tensor(s.output)->data.raw, s.data.size());
  }
  const float* out = interp->typed_tensor<float>(interp->outputs()[0]);
  for (int i = 0; i < frames; ++i, ++encoded_) {
    // The first activations are dropped (num_initial_activations_ignored).
    if (encoded_ < config_.initial_activations_ignored) continue;
    search_->Advance(std::vector<float>(out + i * output_dim_, out + (i + 1) * output_dim_));
    ++decoded_;
  }
}

std::vector<SpeakerTurn> SpeakerTurnDetector::Turns() const {
  std::vector<SpeakerTurn> turns = turns_;
  for (SpeakerTurn& t : SegmentTurns()) turns.push_back(std::move(t));
  return turns;
}

std::vector<SpeakerTurn> SpeakerTurnDetector::SegmentTurns() const {
  // greco's SpeakerTurnStream: the labels split into segments at each
  // speaker change label and wherever a segment would outlast
  // max_speaker_segment_ms.
  const Hypothesis& best = search_->Best();
  const std::vector<int>& labels = best.labels;
  const int change_id = config_.speaker_change_id;
  auto is_label = [&](int id) { return id != 0 && id != change_id; };
  auto text = [&](int id) { return symbols_[id] == "<space>" ? std::string(" ") : symbols_[id]; };
  struct Segment {
    int64_t start = 0, end = 0;
    bool change = false;
    int change_index = -1;  // Of the change label.
    std::string text;       // Its labels.
    bool has_labels = false;
  };
  std::vector<Segment> segments(1);
  for (size_t i = 0; i < labels.size(); ++i) {
    Segment* s = &segments.back();
    if (is_label(labels[i])) {
      s->text += text(labels[i]);
      s->has_labels = true;
    }
    const int64_t t = std::max<int64_t>(
        0, config_.time_offset_ms + static_cast<int64_t>(best.label_frames[i] * frame_ms_));
    s->end = t;
    const bool too_long =
        config_.max_speaker_segment_ms > 0 && t - s->start > config_.max_speaker_segment_ms;
    if (labels[i] == change_id || too_long) {
      Segment next;
      next.start = next.end = t;
      next.change = labels[i] == change_id;
      next.change_index = static_cast<int>(i);
      segments.push_back(std::move(next));
    }
  }
  // A change needs words after it, and none whose segment is only a short
  // response ("mhm") counts: neither it nor the change back after it.
  std::set<std::string> ignored(config_.short_responses_to_ignore.begin(),
                                config_.short_responses_to_ignore.end());
  for (size_t i = 0; i < segments.size(); ++i) {
    Segment& s = segments[i];
    if (!s.has_labels || s.start == s.end) s.change = false;
    std::string t = s.text;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return std::tolower(c); });
    const size_t b = t.find_first_not_of(" \t\n\r\f\v");
    t = b == std::string::npos ? "" : t.substr(b, t.find_last_not_of(" \t\n\r\f\v") - b + 1);
    if (ignored.count(t) != 0) {
      s.change = false;
      if (i + 1 < segments.size()) segments[i + 1].change = false;
    }
  }
  std::vector<SpeakerTurn> turns;
  for (const Segment& s : segments) {
    if (!s.change) continue;
    SpeakerTurn turn;
    turn.time_ms = segment_start_ms_ + s.start;
    const int n = static_cast<int>(labels.size());
    for (int k = 1; k <= config_.context_labels; ++k) {
      const int l = s.change_index - k, r = s.change_index + k;
      if (l >= 0 && is_label(labels[l])) turn.left = text(labels[l]) + turn.left;
      if (r < n && is_label(labels[r])) turn.right += text(labels[r]);
    }
    turns.push_back(std::move(turn));
  }
  return turns;
}

}  // namespace soda
