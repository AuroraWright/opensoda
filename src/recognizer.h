// Streaming speech recognizer over a SODA language pack.

#ifndef SODA_RECOGNIZER_H_
#define SODA_RECOGNIZER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "biasing.h"
#include "config.h"
#include "endpointer.h"
#include "frontend.h"
#include "langid.h"
#include "pqrnn_spoken_punctuation.h"
#include "punctuator.h"
#include "spoken_punctuation.h"
#include "fst_normalizer.h"
#include "roadrunner_capitalizer.h"
#include "speaker_turn.h"
#include "truecaser.h"
#include "symbol_table.h"
#include "transducer.h"
#include "vad_frontend.h"

namespace soda {

struct RecognizerOptions {
  // Either a language pack version directory (containing SODAModels/) or the
  // SODAModels directory itself.
  std::string language_pack_dir;
  // Pipeline config name inside SODAModels/configs, without ".config".
  // Empty picks the pack's CONTINUOUS config, as libsoda does in both modes:
  // LARGE where the pack has it (ja-JP, en-US, de-DE), else MEDIUM (ru-RU,
  // ko-KR, hi-IN and others).
  std::string config_name;
  // Recognition domain, selecting the acoustic cluster id ("CAPTION" for Live
  // Caption, "IME" for dictation). Empty keeps the config's forced id.
  std::string domain = "CAPTION";
  int sample_rate = 16000;
  int channel_count = 1;
  // TFLite threads per model. The encoder runs a few frames at a time, too
  // little work to split: a second thread only spins waiting (about doubling
  // the CPU time) without making recognition faster. Results do not depend
  // on it.
  int num_threads = 1;
  // Hypotheses kept per frame; 0 uses the config's FstSearchParams.max_arcs
  // (3 in ja-JP), as libsoda's histogram pruning does.
  int beam = 0;
  // Segment speech with the pack's voice activity detector when it has one:
  // a result becomes final when the endpointer detects the end of speech.
  bool use_vad = true;
  // Without a VAD, a segment is finalized after this much time with no new
  // labels. Zero disables it; results are then final only on Finish().
  int endpoint_silence_ms = 900;
  // Punctuate results with the pack's punctuation model, if present.
  bool enable_punctuation = true;
  // Mask offensive words ("s***") with the pack's offensive word normalizer,
  // like libsoda's mask_offensive_words. Raw texts stay unmasked.
  bool mask_offensive_words = false;
  // Identify the spoken language with the pack's LANGUAGE_IDENTIFICATION
  // pipeline, reporting it through the language callback.
  bool enable_language_id = false;
  // Keep up with live audio as libsoda does: when recognition restarts after
  // a final with more than this much audio received but not yet recognized
  // (counting the caller's queued audio, see set_queued_audio_callback), it
  // skips to the last max_backlog_ms of it. libsoda keeps 1000 ms, or 5000 ms
  // with a multilang config. 0 recognizes all the audio, however late.
  int max_backlog_ms = 0;
  // Mark the words where another speaker takes over, as libsoda's
  // enable_speaker_change_detection does, with the pack's
  // ONDEVICE_SPEAKER_TURN_DETECTION model (en-US packs; others ignore it).
  // As in libsoda, segments are then cut after 30 s of speech, not 60.
  bool speaker_change_detection = false;
  // Hold back the unfinished sentence at the end of each final, as libsoda's
  // text formatter does without a multilang_config's languages (with the
  // VAD and punctuation on): a final ends after its last word that ends a
  // sentence, punctuated as a partial is, and the words after it start the
  // next partials and final. They are sent on their own, punctuated as a
  // final, after 3 s of audio without a partial, and at the end of the
  // audio.
  bool hold_unfinished_sentences = false;
  // Phrases to bias recognition toward, with their boosts (packs with a
  // class_substitution_twiddler context model, such as en-US; others ignore
  // them).
  std::vector<BiasPhrase> phrases;
};

struct WordTiming {
  std::string text;      // With punctuation.
  std::string raw_text;  // As recognized.
  int64_t start_ms = 0;
  // The word began at a bare word boundary piece. In packs punctuated by
  // character, libsoda's final raw text puts no space before it (ja-JP:
  // `テスト@ Gmailcomです` from `▁テ スト ▁ @ ▁g ma i l ▁ co m ▁です`).
  bool joined = false;
  // SODA's AsrConfidence bucket: 0 (UNKNOWN) or 1 (LOW), as libsoda reports.
  int confidence = 0;
  // Another speaker starts with this word (speaker_change_detection).
  bool speaker_change = false;
  // Bytes at the start and end of raw_text from marks of their own that
  // joined it (dictated punctuation: "hello" and "," in "hello,"), which
  // libsoda's punctuation model does not see.
  size_t marks_before = 0, marks_after = 0;
};

struct RecognitionEvent {
  enum Type { kPartial, kFinal };
  Type type = kPartial;
  std::string text;  // Punctuated, with spaces only where the script uses them.
  // libsoda's formatted hypothesis: punctuated tokens joined by spaces, where
  // tokens are words in a final result and, unless `spaced_words`,
  // characters in a partial one.
  std::string spaced_text;
  std::string raw_text;  // Unpunctuated, as `text`.
  // Whether the language is written with spaces between words. Then
  // partials list words, and libsoda starts each result after a session's
  // first final with a space (`leading_space`).
  bool spaced_words = false;
  bool leading_space = false;
  int64_t start_ms = 0;  // Audio time of the segment start.
  int64_t end_ms = 0;
  bool end_of_audio = false;
  std::vector<WordTiming> words;
};

// The segmenter's start and end of speech, and the end of the audio, as
// libsoda reports them.
struct EndpointEvent {
  enum Type { kStartOfSpeech, kEndOfSpeech, kEndOfAudio };
  Type type = kStartOfSpeech;
  int64_t audio_ms = 0;
};

// libsoda's audio level report, every 20 ms of audio at the model rate:
// the RMS of the 16-bit samples (full scale 1), smoothed with a 100 ms time
// constant, and in the first report after new voice activity output, the
// latest VAD frame's speech posterior. After the end of the audio, a last
// report carries only the posterior.
struct AudioLevelEvent {
  bool has_rms = true;
  float rms = 0.0f;
  bool has_speech_level = false;
  float speech_level = 0.0f;
  int64_t speech_level_ms = 0;  // Start of the posterior's VAD frame.
};

class Recognizer {
 public:
  using Callback = std::function<void(const RecognitionEvent&)>;

  static std::unique_ptr<Recognizer> Create(const RecognizerOptions& options,
                                            std::string* error);

  void set_callback(Callback cb) { callback_ = std::move(cb); }
  // Receives start and end of speech and end of audio events (with a VAD
  // only).
  void set_endpoint_callback(std::function<void(const EndpointEvent&)> cb) {
    endpoint_callback_ = std::move(cb);
  }
  // Receives audio level reports (with a VAD only).
  void set_audio_level_callback(std::function<void(const AudioLevelEvent&)> cb) {
    audio_level_callback_ = std::move(cb);
  }
  // Receives language identification events when enabled.
  void set_language_callback(LanguageIdentifier::Callback cb);
  // Returns how much audio, in sample frames at the input rate, the caller
  // has received but not yet passed to AddAudio (for max_backlog_ms).
  void set_queued_audio_callback(std::function<int64_t()> cb) { queued_audio_ = std::move(cb); }
  // Receives every encoder input frame (normalized features), for debugging.
  void set_feature_tap(std::function<void(const std::vector<float>&)> tap) {
    feature_tap_ = std::move(tap);
  }
  // Receives, per VAD frame, its end time, speech posterior and endpointer
  // state, for debugging.
  void set_vad_tap(std::function<void(int64_t ms, float posterior, int state)> tap) {
    vad_tap_ = std::move(tap);
  }

  // Interleaved little-endian 16-bit PCM at the configured rate/channels.
  void AddAudio(const int16_t* samples, size_t count);
  // Mono float samples on the int16 scale at the configured rate.
  void AddAudioFloat(const float* samples, size_t count);
  // Flushes buffered audio and emits the final result for the segment.
  void Finish();
  // Drops all state and starts a new stream.
  void Reset();
  // Takes over `previous`'s stream, as libsoda does when language
  // identification switches packs (rewind_when_switching_language): its
  // language identifier, audio history and resampler, then recognizes the
  // audio since its last final again. Set the callbacks first.
  void ContinueFrom(Recognizer* previous);
  // Replaces the biasing phrases, from the next label on.
  void SetPhrases(const std::vector<BiasPhrase>& phrases);

  const SymbolTable& symbols() const { return symbols_; }
  std::string LabelsToText(const std::vector<int>& labels) const;

 private:
  Recognizer() = default;
  bool Init(const RecognizerOptions& options, std::string* error);
  // `unrounded` (if not null) receives the 48 kHz resampler's outputs before
  // rounding to 16-bit values, at full scale 1.
  void Resample(const float* in, size_t n, std::vector<float>* out,
                std::vector<float>* unrounded = nullptr);
  // Runs 16 kHz audio through the pipeline, restarting it after each final.
  void Process(const float* samples, size_t count);
  // libsoda restarts recognition at each final's end: the front end, VAD and
  // segmenter start afresh and the audio since then is processed again.
  void Restart();
  void ResetPipeline();
  // Moves frontend output frames, with their end times, to the encoder queue.
  void Enqueue(std::vector<std::vector<float>>* frames);
  void RunVad(std::vector<std::vector<float>>* frames, bool end_of_audio = false);
  // Applies the endpointer's decision for the oldest undecided frame.
  void Segment(bool speech, std::vector<std::vector<float>>* frames);
  // Ends a segmenter segment: emits the frames held for right context.
  void EndSegment(std::vector<std::vector<float>>* frames);
  // Queues a marker that finalizes the result.
  void MarkFinal(std::vector<std::vector<float>>* frames);
  void RunEncoder(bool flush);
  void Decode(const std::vector<std::vector<float>>& encoded, const std::vector<long>& ends);
  void EmitPartial();
  // `segment_end`: the decoder reached the end of a segment, after which the
  // pipeline restarts.
  void EmitFinal(bool end_of_audio, bool segment_end = false);
  // Rewrites the recognized words with the denormalizers (`final`: for a
  // final result).
  void Denormalize(bool final, const std::vector<std::unique_ptr<TextNormalizer>>& normalizers,
                   std::vector<WordTiming>* words) const;
  // The grammars among them, on the words joined by spaces.
  void DenormalizeText(bool final, const std::vector<const TextNormalizer*>& normalizers,
                       std::vector<WordTiming>* words) const;
  // pQRNN spoken punctuation, which keeps the words' times as libsoda does.
  void PunctuateSpoken(const PqrnnSpokenPunctuationNormalizer& normalizer,
                       std::vector<WordTiming>* words) const;
  // Fills in the words' confidence buckets.
  void SetWordConfidence(RecognitionEvent* e) const;
  // Marks the words where the speaker changes, and the formatted text
  // before them. The turns are placed among `all`'s words if given (the
  // words a final is sent from, which start with `e`'s).
  void MarkSpeakerChanges(RecognitionEvent* e, const RecognitionEvent* all = nullptr) const;
  // Fills `e` (text forms and words) from a hypothesis, whose label frames
  // end at `frame_ends_ms` (by default, the first pass's).
  void Describe(const Hypothesis& h, bool final, RecognitionEvent* e) const;
  void Describe(const Hypothesis& h, bool final, const std::vector<int64_t>& frame_ends_ms,
                RecognitionEvent* e) const;
  // Describe's two steps: the words, normalized but not yet punctuated, and
  // the raw text; then the punctuated words and text forms (`trailing_mark`:
  // punctuating after the last word too, as for a final), of the first
  // `count` words only if given and from word `first` (punctuated in the
  // context of all of them). `sentence_ends` receives which words the
  // punctuation model ended a sentence after.
  void DescribeWords(const Hypothesis& h, bool final, const std::vector<int64_t>& frame_ends_ms,
                     RecognitionEvent* e) const;
  void Format(bool final, bool trailing_mark, RecognitionEvent* e,
              size_t count = std::string::npos, std::vector<bool>* sentence_ends = nullptr,
              size_t first = 0) const;
  // The raw text of words, spaced as Describe's.
  std::string JoinRaw(const std::vector<WordTiming>& words, bool final) const;
  // The start of the frame where the hypothesis's last word ended.
  int64_t LastWordEnd(const Hypothesis& h, const std::vector<int64_t>& frame_ends_ms) const;
  // hold_unfinished_sentences: emits a final from the held words and `e`'s
  // (when `force`, all of them), and holds back the rest.
  void EmitHolding(RecognitionEvent* e, int64_t new_words_end_ms, bool force);
  // Sends a final's event with the fields every final gets (`all` as for
  // MarkSpeakerChanges).
  void SendFinal(RecognitionEvent* e, const RecognitionEvent* all = nullptr);
  // Sends the held words if a final left them more than 3 s ago.
  void CheckHeld();
  int64_t AudioMs() const;
  // Feeds first-pass frames through the cascaded encoder to the second pass
  // (`flush`: the segment is closed).
  void RunCascade(bool flush);
  // At a segment end: encodes the rest, with the right context padding.
  void FlushCascade();
  void ResetCascade();

  RecognizerOptions options_;
  PipelineConfig config_;
  AutoGainControl agc_;  // Inactive unless the pipeline has one.
  Frontend frontend_;
  StreamingEncoder encoder_;
  PredictionNetwork decoder_;
  JointNetwork joint_;
  SymbolTable symbols_;
  std::unique_ptr<BeamSearch> search_;
  // The cascaded second pass (en-US): a non-causal encoder over the first
  // pass's encoder output, with its own decoder, which gives the finals.
  bool cascade_ = false;
  StreamingEncoder cascade_encoder_;
  PredictionNetwork decoder2_;
  JointNetwork joint2_;
  std::unique_ptr<BeamSearch> search2_;
  std::vector<std::vector<float>> cascade_pending_;  // Awaiting the cascaded encoder.
  std::vector<int64_t> cascade_input_ends_ms_;       // Ends of the segment's first-pass frames.
  std::vector<float> last_first_pass_frame_;
  size_t cascade_outputs_ = 0;           // Cascaded encoder outputs this segment.
  std::vector<int64_t> frame_ends2_ms_;  // Ends of the frames search2_ has seen.
  WordPieceModel wpm_;
  bool has_wpm_ = false;
  ContextBiaser biaser_;  // Shared by both passes.
  std::unique_ptr<Punctuator> punctuator_;  // Null when disabled or absent.
  std::unique_ptr<Truecaser> truecaser_;    // Null when the pack has none.
  std::unique_ptr<RoadrunnerCapitalizer> capitalizer_;  // Likewise.
  std::unique_ptr<FstNormalizer> masker_;   // Null unless masking is on.
  // ja-JP spoken punctuation commands; null outside dictation.
  std::unique_ptr<SpokenPunctuationNormalizer> spoken_punctuation_;
  // The pack's denormalizers that apply in this recognition mode, in order.
  std::vector<std::unique_ptr<TextNormalizer>> denormalizers_;
  std::vector<std::unique_ptr<TextNormalizer>> precase_denormalizers_;  // Before truecasing.
  // A ClassNormalizer is among them: class tags the model emits ("<money>")
  // stay in the words until it rewrites their spans.
  bool keep_class_tags_ = false;
  std::unique_ptr<VoiceActivityDetector> vad_;  // Null when disabled or absent.
  std::unique_ptr<LanguageIdentifier> langid_;  // Null unless enabled.
  std::unique_ptr<SpeakerTurnDetector> speaker_turns_;  // Likewise.
  // The end of the last final: older speaker turns are no longer placed.
  int64_t speaker_turns_from_ms_ = 0;
  Endpointer endpointer_;
  std::vector<std::vector<float>> vad_input_;  // Frames tapped from frontend_.
  // With config_.vad.spectral_noise, the VAD's input comes from the FFT
  // energies through vad_frontend_ instead.
  std::vector<std::vector<float>> vad_power_;
  VadFrontend vad_frontend_;
  int64_t vad_frames_ = 0;
  float vad_frame_ms_ = 30.0f;
  long decided_frames_ = 0;       // Frames the endpointer has labeled.
  bool in_segment_ = false;       // The segmenter is passing speech on.
  long segment_first_frame_ = 0;  // First frame of the segment in progress.
  Callback callback_;
  std::function<void(const std::vector<float>&)> feature_tap_;
  std::function<void(int64_t, float, int)> vad_tap_;
  std::function<void(const EndpointEvent&)> endpoint_callback_;
  void SendEndpoint(EndpointEvent::Type type, int64_t audio_ms, bool force = false);
  // Reports audio levels for model-rate samples about to be processed,
  // whose full scale is 1 / `scale`.
  void ReportLevels(const float* samples, size_t count, double scale);
  void ResetLevels();
  std::function<void(const AudioLevelEvent&)> audio_level_callback_;
  double level_power_ = 0.0;      // Smoothed mean square.
  double level_block_sum_ = 0.0;  // Sum of squares in the current 20 ms.
  int level_block_samples_ = 0;
  bool level_pending_ = false;  // A posterior not reported yet.
  float pending_level_ = 0.0f;
  int64_t pending_level_ms_ = 0;
  bool speech_started_ = false;  // The last endpoint event was a start.
  int64_t last_endpoint_ms_ = -1;  // And its time.
  bool session_started_ = false;

  std::vector<std::vector<float>> pending_frames_;  // An empty frame ends a segment.
  std::vector<long> pending_ends_;  // Their ends, in stacked filterbank frames.
  std::vector<long> frame_ends_;    // Frontend output ends, until enqueued.
  int queued_finals_ = 0;  // Segment ends not yet decoded.
  bool first_batch_ = true;  // The next encoder batch starts a segment.
  size_t backlog_ = 0;       // Frames left of the backlog a segment start released.
  float stacked_frame_ms_ = 30.0f;
  std::vector<int64_t> frame_ends_ms_;  // Ends of the frames the search has seen.
  int64_t decoded_end_ms_ = 0;          // End of the audio decoded so far.
  int64_t segment_start_ms_ = 0;
  int64_t last_final_end_ms_ = 0;  // Where the last final ended.
  std::string last_partial_;
  // hold_unfinished_sentences: the words held back, not punctuated, the end
  // of the last one, the audio time of the final that left them (until a
  // partial, -1), and the end of the last final with words.
  bool holding_ = false;
  std::vector<WordTiming> held_;
  int64_t held_end_ms_ = 0;
  int64_t held_since_ms_ = -1;
  int64_t formatter_end_ms_ = 0;
  float frame_ms_ = 60.0f;
  // Whether the language separates words by spaces (the punctuation
  // config's separate_words_by_space): its partials list words rather than
  // characters, and results after the first start with a space.
  bool spaced_words_ = false;
  bool character_partials_ = false;  // Partials list characters, not words.
  std::vector<std::string> punctuation_marks_;  // The punctuation settings' marks (",", "，").
  bool had_final_ = false;

  // 16 kHz audio kept for restarts, like libsoda's 30 s audio history.
  std::vector<float> history_;
  int64_t history_start_ = 0;  // Sample index of history_[0].
  int64_t origin_ = 0;         // Sample index where the pipeline (re)started.
  int64_t restart_at_ = -1;    // Sample index to restart at, or -1.
  int64_t skip_to_ = 0;        // Audio before this sample index is not recognized.
  std::function<int64_t()> queued_audio_;

  // Resampler state (input rate -> 16 kHz).
  double resample_ratio_ = 1.0;
  double resample_pos_ = 0.0;
  std::vector<float> resample_history_;
  std::vector<float> resample_kernel_;
  int resample_taps_ = 0;
  int resample_phases_ = 0;
};

}  // namespace soda

#endif  // SODA_RECOGNIZER_H_
