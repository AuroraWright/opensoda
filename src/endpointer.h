// Voice activity detection and endpointing, following greco's "vad.*" streams:
//
//   stacked filterbanks (240 dims, 30 ms) -> AppendClusterIdStream (one-hot)
//   -> LstmComputeStream (acousticmodel/*.endpointer_portable_lstm_model)
//   -> EndpointerStream (speech/non-speech state machine)
//
// The endpointer model files are TFLite flatbuffers XORed with a one-byte key.

#ifndef SODA_ENDPOINTER_H_
#define SODA_ENDPOINTER_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace soda {

class VoiceActivityDetector {
 public:
  bool Init(const std::string& models_dir, const VadConfig& config, std::string* error);
  // Clears the LSTM state.
  void Reset();
  // Speech posterior for one stacked-filterbank frame (before normalization).
  float Process(const std::vector<float>& frame);

 private:
  VadConfig config_;
  std::string model_data_;  // Decoded flatbuffer; must outlive model_.
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::vector<float> mean_, stddev_;
  int input_dim_ = 0;
  int frames_since_reset_ = 0;
  int frames_below_ = 0;
};

// greco's frame-counting endpointer (EndpointerParams.type = 1, the
// default), a descendant of Chromium's EnergyEndpointer: a frame is "on" when
// its speech posterior exceeds the decision threshold, and the state moves on
// counts of "on" frames within sliding windows.
//
// Its output is a per-frame speech decision, emitted with a delay: detected
// utterances are padded by endpoint_margin on both sides, so a pause shorter
// than about twice the margin stays inside one utterance.
class Endpointer {
 public:
  enum State { kPreSpeech, kPossibleOnset, kSpeechPresent, kPossibleOffset };

  void Init(const EndpointerConfig& config);
  void Reset();
  // Advances by one input frame. Returns true when the output decision has
  // just turned from speech to silence; end_frame() is then the index of the
  // first silent output frame.
  bool Process(float speech_posterior);
  State state() const { return state_; }
  long end_frame() const { return end_frame_; }
  // Output frames decided so far, and the decision given to the newest ones.
  long frames_output() const { return frames_output_; }
  bool last_output() const { return last_output_; }
  // Moves the decisions made since the last call, one per output frame, in
  // order, to `out`.
  void TakeDecisions(std::vector<bool>* out) {
    out->insert(out->end(), decisions_.begin(), decisions_.end());
    decisions_.clear();
  }

 private:
  int CountOn(int window) const;
  // Emits `speech` for output frames up to `last` (inclusive). Returns true
  // if that ends an utterance.
  bool Emit(long last, bool speech);
  // The last frame that the current utterance's output may extend to.
  long SpeechOutputBound(long frame) const;

  EndpointerConfig config_;
  // Window lengths and thresholds in frames.
  int onset_window_ = 0, offset_window_ = 0, speech_on_window_ = 0;
  int onset_detect_ = 0, onset_confirm_ = 0, offset_threshold_ = 0, on_maintain_ = 0;
  int offset_confirm_frames_ = 0;
  int margin_ = 0, margin_before_ = 0, margin_after_ = 0;
  size_t ring_size_ = 0;
  std::deque<bool> history_;
  long frames_ = 0;
  State state_ = kPreSpeech;
  long frames_output_ = 0;
  bool last_output_ = false;
  long onset_start_ = 0;  // First frame of a possible onset.
  long pending_ = 0;      // Speech frames still owed after an offset.
  long end_frame_ = -1;
  std::vector<bool> decisions_;  // Not yet taken.
};

}  // namespace soda

#endif  // SODA_ENDPOINTER_H_
