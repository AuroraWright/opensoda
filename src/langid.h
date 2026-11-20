// Spoken language identification, a port of libsoda's LANGUAGE_IDENTIFICATION
// pipeline:
//
//   AutoGainControl -> 32 ms log-mel frames (128 channels) -> FrameStacker
//   -> VAD (its own LSTM endpointer) -> speech frames only
//   -> langid/ONDEVICE_langid.tflite (streaming, 2 frames per call)
//   -> per-frame language and confidence level.
//
// Each classifier output yields one event: the highest-scoring language among
// those the application parameters map, and a confidence level from
// greco's langid_confidence_lib (a weighted sum of the probability margin,
// the negentropy of the output and the continuity of its argmax).

#ifndef SODA_LANGID_H_
#define SODA_LANGID_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config.h"
#include "endpointer.h"
#include "frontend.h"
#include "transducer.h"

namespace soda {

struct LanguageEvent {
  std::string language;  // Lowercase BCP 47, as libsoda reports it ("ja-jp").
  // SodaLangIdEvent.confidence_level: 0 unknown, 1 not confident,
  // 2 confident, 3 highly confident.
  int confidence_level = 0;
  int64_t audio_ms = 0;  // End of the audio the decision covers.
  // A later level-3 event came from the same chunk of audio. libsoda
  // considers a language switch only on the last one.
  bool superseded = false;
};

// greco's AutoGainControl with the SAMPLE_LEVEL_TARGET_RANGE estimator:
// a decaying peak follower feeds a smoothed level, and the gain steers that
// level towards a target, fading out below lower_level_target.
class AutoGainControl {
 public:
  void Init(const AgcConfig& config, int sample_rate);
  void Reset();
  // Scales int16-scale samples in place.
  void Process(float* samples, size_t n);

 private:
  AgcConfig config_;
  float decay_ = 0.0f;   // Per-sample peak decay.
  float attack_ = 1.0f;  // Level smoothing rate.
  float peak_ = 0.0f;
  float level_ = 0.0f;
  std::vector<float> delay_line_;
  size_t delay_pos_ = 0;
};

class LanguageIdentifier {
 public:
  using Callback = std::function<void(const LanguageEvent&)>;

  // `models_dir` is the SODAModels directory.
  static std::unique_ptr<LanguageIdentifier> Create(const std::string& models_dir,
                                                    std::string* error);

  void set_callback(Callback cb) { callback_ = std::move(cb); }
  // Mono 16 kHz samples on the int16 scale.
  void Accept(const float* samples, size_t n);
  // At the end of the audio: reports a last event.
  void Finish();
  void Reset();

 private:
  LanguageIdentifier() = default;
  bool Init(const std::string& models_dir, std::string* error);
  void ProcessFrame(const std::vector<float>& features, bool speech);
  void Classify(std::vector<float> features);
  void RunClassifier();
  void ResetClassifier();
  void Score(const float* probs);
  void Deliver();

  LangIdConfig config_;
  AutoGainControl agc_;
  Frontend frontend_;
  VoiceActivityDetector vad_;
  Endpointer endpointer_;
  TfLiteModel model_;
  Callback callback_;

  std::vector<std::vector<float>> stacked_;  // Frontend tap.
  std::vector<std::vector<float>> waiting_;  // Frames awaiting a VAD decision.
  long frames_seen_ = 0;                     // VAD frames processed.
  long frames_decided_ = 0;                  // Frames with a decision.
  bool in_segment_ = false;
  std::vector<std::vector<float>> held_;  // Silence awaiting more speech.
  long silence_frames_ = 0;
  std::vector<std::vector<float>> batch_;
  long frames_since_reset_ = 0;
  bool scored_ = false;  // A classifier call has been made.
  std::optional<LanguageEvent> last_event_;
  std::vector<LanguageEvent> pending_;  // Events of the current chunk.
  float frame_ms_ = 30.0f;

  struct StateTensor {
    int input;
    int output;
    std::vector<char> data;
  };
  int input_tensor_ = -1;
  int output_tensor_ = -1;
  int input_dim_ = 0;
  int num_outputs_ = 0;
  std::vector<StateTensor> states_;

  // Output node -> language, for the mapped nodes.
  std::vector<std::string> node_language_;
  int last_argmax_ = -1;
  int continuity_ = 0;
};

}  // namespace soda

#endif  // SODA_LANGID_H_
