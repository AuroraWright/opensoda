// Speaker change detection, a port of libsoda's
// ONDEVICE_SPEAKER_TURN_DETECTION pipeline (en-US packs):
//
//   AutoGainControl -> 32 ms log-mel frames (128 channels) -> FrameStacker
//   (4 frames, every third) -> diarization/SODA_SPEAKER_TURN.scd.encoder
//   (streaming, 3 frames per call) -> a character transducer (an LSTM
//   prediction network) whose transcript marks speaker changes with '#'.
//
// The pipeline's own VAD and segmenter pass every frame on to the encoder
// (skip_frame_filtering), but end the search's segments after a second of
// silence; the search then resumes with its context. greco's
// SpeakerTurnStream turns the best hypothesis into turns: each '#' starts one
// at its frame's time, with the labels around it as context, which libsoda
// uses to place the change among the words.
//
// Like libsoda, times count the search's frames: the encoder's first four
// outputs are dropped, so they are 120 ms early.

#ifndef SODA_SPEAKER_TURN_H_
#define SODA_SPEAKER_TURN_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "endpointer.h"
#include "frontend.h"
#include "langid.h"
#include "transducer.h"

namespace soda {

struct SpeakerTurn {
  int64_t time_ms = 0;  // Audio time of the change.
  // Up to context_labels characters of the transcript on each side.
  std::string left;
  std::string right;
};

// A recognized word, as libsoda's diarization sees it.
struct SpeakerTurnWord {
  std::string text;  // Formatted.
  std::string raw;
  int64_t start_ms = 0;
  bool leading_space = false;  // Inner words of languages written with spaces.
};

// libsoda's placement of speaker turns among a result's words (its
// diarization processor, with context alignment and sentence snapping on).
// Each turn marks the word whose start is closest to its time, or a
// neighbour of it whose surrounding text better matches the turn's context;
// a mark then moves to the start of the sentence it is next to. Returns
// whether each word starts a new speaker's speech.
std::vector<bool> PlaceSpeakerTurns(const std::vector<SpeakerTurn>& turns,
                                    const std::vector<SpeakerTurnWord>& words);

class SpeakerTurnDetector {
 public:
  // `models_dir` is the SODAModels directory; fails if the pack has no
  // speaker turn model.
  static std::unique_ptr<SpeakerTurnDetector> Create(const std::string& models_dir,
                                                     std::string* error);

  // Mono 16 kHz samples on the int16 scale.
  void Accept(const float* samples, size_t n);
  // At the end of the audio: encodes the frames held back.
  void Finish();
  void Reset();
  // The speaker changes in the best hypothesis so far.
  std::vector<SpeakerTurn> Turns() const;

 private:
  SpeakerTurnDetector() = default;
  bool Init(const std::string& models_dir, std::string* error);
  // Whether the segment ends before the frame at `t` ms with this decision.
  bool SegmentEnds(int64_t t, bool speech);
  void EndSegment();
  void Push(std::vector<float> frame);
  void Encode();
  // The turns in the current segment's best hypothesis.
  std::vector<SpeakerTurn> SegmentTurns() const;

  SpeakerTurnConfig config_;
  AutoGainControl agc_;
  Frontend frontend_;
  VoiceActivityDetector vad_;
  Endpointer endpointer_;
  std::vector<float> mean_, stddev_;
  TfLiteModel encoder_;
  struct StateTensor {
    int input;
    int output;
    std::vector<char> data;
  };
  std::vector<StateTensor> states_;
  int input_dim_ = 0;
  int output_dim_ = 0;
  PredictionNetwork decoder_;
  JointNetwork joint_;
  std::unique_ptr<BeamSearch> search_;
  std::vector<std::string> symbols_;
  float frame_ms_ = 30.0f;

  std::vector<std::vector<float>> stacked_;  // Frontend tap.
  std::deque<std::vector<float>> waiting_;   // Frames awaiting a VAD decision.
  std::vector<std::vector<float>> batch_;    // Frames awaiting the encoder.
  std::vector<float> last_input_;            // The newest frame, for Finish().
  long frames_decided_ = 0;
  long encoded_ = 0;  // Encoder output frames so far.
  long decoded_ = 0;  // Frames searched.
  struct SegmenterState {
    enum { kIdle, kSpeech, kSilence } state = kIdle;
    int64_t last_speech_ms = 0;
    int64_t segment_start_ms = 0;
  } segmenter_;
  int64_t segment_start_ms_ = 0;    // Of the search's segment.
  std::vector<SpeakerTurn> turns_;  // Of the segments before it.
};

}  // namespace soda

#endif  // SODA_SPEAKER_TURN_H_
