// Streaming RNN-T (HAT) model built from the language pack's TFLite pieces:
//
//   large-encoder.tflite        causal Conformer; [1,T,482] -> [1,T,512], with
//                               85 recurrent state tensors fed back each call
//   large-decoder.tflite        stateless prediction network over the last two
//                               labels; label + previous label -> [B,640]
//   large-joint_posterior.tflite  encoder [B,512] + decoder [B,640] ->
//                               P(blank, label_1..label_N) as [B,N+1]
//
// Symbol 0 (<epsilon>) is the blank; output index i is symbol id i.

#ifndef SODA_TRANSDUCER_H_
#define SODA_TRANSDUCER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "biasing.h"
#include "config.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace soda {

class TfLiteModel {
 public:
  TfLiteModel();
  ~TfLiteModel();
  bool Load(const std::string& path, int num_threads, std::string* error);
  tflite::Interpreter* interpreter() { return interpreter_.get(); }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

class StreamingEncoder {
 public:
  bool Init(const std::string& path, int num_threads, std::string* error);
  // Zeroes the recurrent state.
  void Reset();
  // Encodes T frames of input_dim() features, carrying state across calls.
  bool Encode(const std::vector<std::vector<float>>& frames,
              std::vector<std::vector<float>>* out, std::string* error);

  int input_dim() const { return input_dim_; }
  int output_dim() const { return output_dim_; }
  // The recognition domain, for encoders that take one.
  void set_domain_id(int id) { domain_id_ = id; }

 private:
  struct State {
    int input_tensor;
    int output_tensor;
    std::vector<char> data;
  };

  TfLiteModel model_;
  int input_tensor_ = -1;
  int output_tensor_ = -1;
  int input_dim_ = 0;
  int output_dim_ = 0;
  int batch_time_ = 0;  // Time dimension the interpreter is allocated for.
  int domain_tensor_ = -1;
  int domain_id_ = 0;
  std::vector<State> states_;
};

// A recurrent prediction network's state after a hypothesis's labels, and
// its output.
struct DecoderState {
  std::vector<float> output;
  std::vector<std::vector<float>> states;
};

struct Hypothesis {
  std::vector<int> labels;
  std::vector<int> label_frames;  // Encoder frame index of each label.
  float cost = 0.0f;              // Accumulated cost (-log p, smoothed).
  float model_cost = 0.0f;        // The same without biasing.
  int prev_label = 0;             // Decoder state: the label before last.
  int last_label = 0;
  int bias_state = 0;             // ContextBiaser state after the labels.
  // Recurrent prediction networks: the state after last_label, computed
  // when the hypothesis is first scored from the one before it (null:
  // zeros).
  std::shared_ptr<const DecoderState> decoder_state;
  std::shared_ptr<const DecoderState> parent_state;
};

class PredictionNetwork {
 public:
  bool Init(const std::string& path, int num_threads, std::string* error);
  // Output for `label` with `prev_label` as the decoder state. Cached.
  const std::vector<float>& Get(int prev_label, int label);
  // The output for a hypothesis's labels. Recurrent networks first compute
  // the states the hypotheses lack.
  void Outputs(std::vector<Hypothesis>& hyps, std::vector<const std::vector<float>*>* out);
  int output_dim() const { return output_dim_; }
  void ClearCache() { cache_.clear(); }

 private:
  TfLiteModel model_;
  int output_dim_ = 0;
  // A recurrent network (SODA_SPEAKER_TURN.scd.decoder) takes the label and
  // its states and returns its output and the new states, input i + 1 being
  // fed back from output i + 1, for a fixed batch of hypotheses.
  bool recurrent_ = false;
  int batch_ = 1;
  std::unordered_map<uint64_t, std::vector<float>> cache_;
};

class JointNetwork {
 public:
  // `logits`: the network outputs logits rather than probabilities.
  bool Init(const std::string& path, int num_threads, std::string* error,
            bool logits = false);
  // Writes rows of output_dim() probabilities (or logits), one per decoder
  // output.
  void Run(const float* encoder_frame,
           const std::vector<const std::vector<float>*>& decoder_outputs,
           std::vector<float>* probs);
  int output_dim() const { return output_dim_; }
  bool logits() const { return logits_; }

 private:
  TfLiteModel model_;
  bool logits_ = false;
  // Inputs of rank 3 ([1, 1, D] and [B, 1, D]): one encoder frame against
  // a fixed batch of decoder outputs.
  bool rank3_ = false;
  int encoder_dim_ = 0;
  int decoder_dim_ = 0;
  int output_dim_ = 0;
  int batch_ = 0;
};

struct BeamSearchOptions {
  // FstSearchParams: histogram pruning keeps about max_arcs hypotheses (and
  // at least min_arcs) within local_beam of the best, in histogram_buckets
  // bins over histogram_scale. word_pen is added to every label's cost.
  int max_arcs = 3;
  int min_arcs = 1;
  float local_beam = 5.0f;
  float histogram_scale = 5.0f;
  int histogram_buckets = 100;
  float word_pen = 0.0f;
  // DualSearchParams: labels costlier than this are not expanded (0: no
  // limit), and a frame expands labels at most max_expansions - 1 times.
  float max_label_cost = 0.0f;
  int max_expansions = 100;
  float temperature = 1.0f;
  float max_model_cost_to_influence = 1e20f;
  bool prune_on_min_cost = false;
};

BeamSearchOptions SearchOptions(const DecoderConfig& d);

// greco's breadth-first transducer search (speech/decoder2/dual,
// BREADTH_FIRST strategy) over a graph that accepts any label sequence.
//
// On each frame, every hypothesis takes the blank into the next frame, and
// those within local_beam of the previous frame's best also take every label
// whose cost is under max_label_cost. The new hypotheses are pruned with a
// cost histogram and expanded again, until none are left. A label that
// reaches the label sequence of a hypothesis the frame started with adds that
// hypothesis's probability. The blank successors are then merged by label
// sequence, keeping the cheaper, and pruned with the histogram again.
class BeamSearch {
 public:
  BeamSearch(PredictionNetwork* decoder, JointNetwork* joint,
             const BeamSearchOptions& options, const DecoderConfig& config);
  // Starts a new segment from <S>.
  void Reset();
  // Starts a new segment that keeps the best hypothesis's decoder context
  // (its last two labels) but none of its labels.
  void Resume();
  void Advance(const std::vector<float>& encoder_frame);
  // Adds the biaser's costs to labels (null: none). It must outlive the
  // search.
  void set_biaser(const ContextBiaser* biaser) { biaser_ = biaser; }
  // Best hypothesis so far.
  const Hypothesis& Best() const { return beam_[best_]; }
  int frames() const { return frame_; }

 private:
  // Label costs for each hypothesis in `hyps` on this frame.
  void Score(const std::vector<float>& encoder_frame, std::vector<Hypothesis>& hyps,
             std::vector<std::vector<float>>* costs);
  // The pruning threshold from greco's ScoreHistogram for `hyps` relative to
  // `best`, starting from `threshold`.
  float HistogramThreshold(const std::vector<Hypothesis>& hyps, float best, float threshold) const;
  // The cost pruning compares: with prune_on_min_cost, the cheaper of the
  // biased and unbiased costs.
  float PruningCost(const Hypothesis& h) const {
    return options_.prune_on_min_cost ? std::min(h.cost, h.model_cost) : h.cost;
  }

  PredictionNetwork* decoder_;
  JointNetwork* joint_;
  BeamSearchOptions options_;
  DecoderConfig config_;
  const ContextBiaser* biaser_ = nullptr;
  std::vector<Hypothesis> beam_;
  // The costs of the label sequences the previous frame ended with, before
  // pruning.
  struct Ended {
    float cost, model_cost;
    std::vector<int> label_frames;
    bool kept = false;  // Survived the frame's pruning.
  };
  std::map<std::vector<int>, Ended> ended_;
  size_t best_ = 0;
  float best_cost_ = 0.0f;  // Of the hypotheses entering this frame.
  int frame_ = 0;
  std::vector<float> probs_;
};

}  // namespace soda

#endif  // SODA_TRANSDUCER_H_
