// Recognizer configuration extracted from a SODA language pack.
//
// A language pack ships its recognizer as a "greco" pipeline: a binary
// greco.PipelineDef proto (configs/*.config) listing streams such as Framer,
// Window, FilterBank, FrameStacker, CausalEncoderStream and the decoder, plus
// resources pointing at the model files. Only the parameters this
// reimplementation uses are extracted; defaults mirror the proto defaults.

#ifndef SODA_CONFIG_H_
#define SODA_CONFIG_H_

#include <map>
#include <string>
#include <vector>

namespace soda {

struct FrameStackConfig {
  int input_size = 0;
  int left_context = 0;
  int right_context = 0;
  bool zero_padding = false;
  int subsample = 1;  // From the SubsampleStream following the stacker.
};

struct FrontendConfig {
  // Framer.
  float frame_size_ms = 25.0f;
  float frame_step_ms = 10.0f;
  // Window.
  float preemphasis = 0.97f;
  int window_type = 1;  // greco.WindowParams.WindowType; 1 = HANNING.
  float noise_scale = 0.0f;
  float window_size_ms = 25.0f;
  // Fft.
  int fft_input_length_ms = 25;
  int min_fft_size = 512;
  bool fft_overdrive = false;
  // FilterBank.
  int channel_count = 80;
  float upper_band_limit = 7500.0f;
  float lower_band_limit = 125.0f;
  float fbank_output_floor = 1.0f;
  // Stacking before normalization, and the stacking layer after it.
  FrameStackConfig stack1{80, 2, 0, false, 3};
  // Packs whose SegmenterStream precedes the first SubsampleStream (ko-KR,
  // hi-IN) gate every stacked frame and subsample only the speech frames.
  bool subsample_after_gate = false;
  FrameStackConfig stack2{241, 0, 1, true, 2};
  // AppendClusterIdStream on the encoder path (-1 disables it).
  int cluster_id = 2;
  std::string mean_stddev_file;  // Relative to the SODAModels directory.
};

struct DecoderConfig {
  int sos_id = 1;
  int eos_id = 2;
  // Multiplies P(</S>) (proto default 1): CONTINUOUS sets 0, never ending an
  // utterance on </S>; SHORT sets 1.4.
  float eos_penalty = 1.0f;
  // FstDecoderParams.enable_resume_search: a new segment continues from the
  // previous one's decoder context instead of from <s> (CONTINUOUS sets it).
  bool resume_search = false;
  float smoothing_temperature = 1.0f;
  float internal_lm_smoothing_temperature = 1.0f;
  float internal_lm_cost_scale = 0.0f;
  bool subtract_internal_lm = false;
  bool is_hat_model = true;
  int decoder_initial_value = 1;
  // FstSearchParams.
  int max_arcs = 3;  // Hypotheses kept by histogram pruning.
  int min_arcs = 0;
  float local_beam = 5.0f;
  float histogram_scale = 5.0f;
  int histogram_buckets = 100;
  float word_pen = 0.0f;  // Added to every label's cost.
  // DualSearchParams: labels costlier than this (-log p) are not expanded;
  // 0 disables the limit. max_expansions caps the expansion rounds a frame.
  float max_model_cost = 0.0f;
  int max_expansions = 100;
  // Labels costlier than this get no biasing (DualSearchParams
  // max_model_cost_to_influence), and whether a label's cost limit and the
  // pruning look at the cheaper of its biased and unbiased costs.
  float max_model_cost_to_influence = 1e20f;
  bool prune_on_min_cost = false;
};

// greco.EndpointerParams: the state machine of Chromium's energy endpointer,
// driven by per-frame speech posteriors. Durations are in seconds.
struct EndpointerConfig {
  int posterior_dimension = 0;
  float frame_period = 0.01f;
  float endpoint_margin = 0.2f;
  float onset_window = 0.15f;
  float speech_on_window = 0.4f;
  float offset_window = 0.15f;
  float onset_detect_dur = 0.09f;
  float onset_confirm_dur = 0.075f;
  float on_maintain_dur = 0.1f;
  float offset_confirm_dur = 0.12f;
  float decision_threshold = 0.96f;
  // Speech output is padded by these on each side of a detected utterance.
  float additional_margin_before_speech = 0.0f;
  float additional_margin_after_speech = 0.0f;
  bool never_transition_to_silence = false;
};

// greco.SpectralNoiseParams: per-bin noise levels from percentiles of the
// FFT energies seen so far (a DataHistoryCDF).
struct SpectralNoiseConfig {
  int initial_buffer_num_frames = 20;
  int num_frames_start_downsampling = 200;
  int sample_every_n_frames = 2;
  // In output order: peak, median, background, floor (where enabled).
  std::vector<float> percentiles{0.95f, 0.25f};
  bool add_input_to_output = true;
};

// greco.SpectralSubtractorParams.
struct SpectralSubtractorConfig {
  int number_spectral_vectors = 3;
  float oversubtraction_factor = 0.7f;
  float noise_low_subtraction_threshold = 16.0f;
  float noise_high_subtraction_threshold = 70.0f;
  float beta_floor_scale = 0.2f;
  float snr_low_subtraction_threshold = 10.0f;
  float snr_high_subtraction_threshold = 44.0f;
  float snr_low_scale = 1.0f;
  float snr_high_scale = 0.0f;
};

// Voice activity branch: stacked filterbanks -> one-hot cluster id -> LSTM
// speech posterior -> endpointer ("vad.decision_for_segmenter"). Some packs
// (ko-KR, hi-IN) instead feed the LSTM 10 ms log mel energies of the
// noise-subtracted spectrum: FFT energies -> SpectralNoise ->
// SpectralSubtractor -> FilterBank.
struct VadConfig {
  bool spectral_noise = false;
  SpectralNoiseConfig noise;
  SpectralSubtractorConfig subtractor;
  // A Window of the branch's own (pl-PL) replaces the main path's dither
  // with this noise scale; negative: the branch takes the main FFT energies.
  float noise_scale = -1.0f;
  // The branch's own FilterBank (greco.FilterBankParams defaults).
  int channel_count = 20;
  float upper_band_limit = 3800.0f;
  float lower_band_limit = 125.0f;
  float fbank_output_floor = 1.0f;
  std::string model_file;        // Relative to SODAModels; may be obfuscated.
  std::string mean_stddev_file;
  int cluster_id = -1;
  int num_clusters = 0;          // >0: the cluster id is one-hot encoded.
  // LstmComputeStreamParams: once min_frames_before_reset frames have passed,
  // a run of reset_frames_below posteriors under reset_threshold clears the
  // LSTM state (0 frames: never).
  int min_frames_before_reset = 0;
  int reset_frames_below = 1;
  float reset_threshold = 0.1f;
  EndpointerConfig endpointer;
  // greco.SegmenterParams: segments longer than these are cut (0 = never).
  int impatience_start_ms = 0;
  int impatience_limit_ms = 0;
  // The pipeline reports audio levels (an AudioLevelEventStream; pt-BR's
  // has none, and libsoda then sends no level events).
  bool audio_level_events = false;
};

// A CausalEncoderStream: the encoder model and how frames are batched.
struct EncoderConfig {
  std::string file;  // Relative to the SODAModels directory.
  // Batches the encoder takes while streaming (the largest one ready is
  // used) and when flushing at a segment end.
  std::vector<int> time_batch_sizes{2, 4, 8};
  std::vector<int> flush_batch_sizes{1};
  // EncoderStreamParams.domain_id_mapping_ref: encoders with a domain_id
  // input take the recognition domain's id from this map (default_domain_id
  // for other domains; -1 if the encoder takes none).
  std::map<std::string, int> domain_ids;
  int default_domain_id = -1;
};

// A second, non-causal pass that some packs (en-US) run for final results:
// the first encoder's outputs, padded with right context, go through a
// cascaded encoder and a second transducer.
struct CascadeConfig {
  EncoderConfig encoder;  // file is empty if the pipeline has no cascade.
  // EncoderInsertRightContextStream: frames of lookahead the cascaded
  // encoder sees; at a segment end the last frame is repeated.
  int right_context = 0;
  bool use_last_frame = false;
  DecoderConfig decoder;
  std::string decoder_file;
  std::string joint_file;
  std::string joint_prior_file;
};

// The class_substitution_twiddler context model biasing recognition toward
// caller phrases (greco.DynamicClassTwiddlerModelParams); wpm_file is empty
// if the pack has none.
struct BiasingConfig {
  std::string wpm_file;  // Relative to the SODAModels directory.
  // The model's name; it only takes phrases from a context of that name.
  std::string context_name;
  float token_weight = 0.0f;   // grammar_token_base_weight.
  bool respect_boost = false;  // respect_grammar_phrase_boost.
  float retwiddle_penalty = 0.0f;
};

// greco.AutoGainControlParams (only the SAMPLE_LEVEL_TARGET_RANGE estimator
// is implemented).
struct AgcConfig {
  bool activated = false;  // False when the pipeline has no AutoGainControl.
  int estimator = 3;
  int delay_ms = 8;
  int attack_window_ms = 0;  // 0: the level follows over delay_ms instead.
  float half_life_sec = 2.0f;
  float floor = 0.004f;
  float curvature = 0.03f;
  float level_target = 0.6f;
  float lower_level_target = 0.075f;
  float gain_scale = 1.2f;
  float lower_target_transition = 20.0f;
};

struct PipelineConfig {
  AgcConfig agc;  // Applied to the audio before the front end.
  FrontendConfig frontend;
  DecoderConfig decoder;
  EncoderConfig encoder;
  CascadeConfig cascade;
  VadConfig vad;  // model_file is empty if the pipeline has no VAD.
  BiasingConfig biasing;
  // AppendClusterIdStream.domain_name_to_internal_id of the VAD's and the
  // encoder path's streams: the recognition domain ("CAPTION", "IME", ...)
  // selects the cluster id each feeds its model.
  std::map<std::string, int> vad_cluster_id_by_domain;
  std::map<std::string, int> frontend_cluster_id_by_domain;
  // Model files, relative to the SODAModels directory.
  std::string decoder_file;
  std::string joint_file;
  std::string joint_prior_file;
  std::string symbols_file;
  // GeneralCapitalizationNormalizer: case variants per word and an n-gram LM
  // choosing among them (empty if the pack has none).
  std::string capitalization_transducer_file;
  std::string capitalization_lm_file;
  // RuntimeRoadrunnerCapitalizationNormalizer: a TFLite model casing words
  // in windows of `window`, `left` and `right` of them context only.
  struct CapitalizationModel {
    std::string file;  // Empty if the pack has none.
    int window = 5;
    int left = 1;
    int right = 1;
  } capitalization_model;
  // The denormalizer list's PORN_DENORMALIZER: an FstNormalizer .mfar that
  // masks offensive words (empty if the pack has none).
  std::string offensive_word_masker_file;
  // The list's other byte-level FstNormalizer (and ClassNormalizer) .mfar
  // denormalizers, in order: untyped ones (type 0) always apply, spoken
  // punctuation (3) and spoken emoji (4) only in dictation.
  struct Denormalizer {
    std::string file;
    int type = 0;
    // Listed before the capitalization normalizer, so it sees the words
    // before truecasing (ja-JP, de-DE).
    bool before_capitalization = false;
    bool class_normalizer = false;  // Rewrites class-tagged spans.
    // A PortablePqrnnSpokenPunctuationNormalizer: `file` is its model, and
    // this its SpokenPunctuationInferenceSettings.
    std::string spoken_punctuation_settings;
  };
  std::vector<Denormalizer> denormalizers;
  // The list's PunctuationNormalizer (ja-JP: spoken punctuation, type 3),
  // after the others: files relative to the pack (empty if none).
  struct SpokenPunctuation {
    std::string converter_file;
    std::string symbol_file;
    std::string model_file;
    int type = 0;
  };
  SpokenPunctuation spoken_punctuation;
};


// greco.LangIdParams.ConfidenceIntervalParams: a frame's confidence score is
// a weighted sum of features of the model's output distribution.
struct LangIdConfidenceConfig {
  float prob_weight = 0.0f;
  float prob_delta_weight = 0.0f;
  float negentropy_weight = 0.0f;
  float continuity_weight = 0.0f;
  float continuity_upper_bound = -1.0f;  // Negative: unbounded.
  float confident_threshold = 0.0f;
  float highly_confident_threshold = 0.0f;
  bool has_thresholds = false;
};

// The LANGUAGE_IDENTIFICATION pipeline: its own front end and VAD, then a
// streaming classifier over the speech frames.
struct LangIdConfig {
  AgcConfig agc;
  FrontendConfig frontend;  // stack1 gives the classifier's input frames.
  VadConfig vad;
  std::string model_file;               // Relative to SODAModels.
  std::string application_params_file;  // greco.ApplicationSpecificParams.
  int frame_batch_size = 1;             // Frames per classifier invocation.
  int reset_every_n_frame = 0;
  bool reset_segment_state = false;
  // SegmenterParams.reset_interval_ms: silence that ends a segment.
  int reset_interval_ms = 1000;
  LangIdConfidenceConfig confidence;
};

bool LoadLangIdConfig(const std::string& path, LangIdConfig* config,
                      std::string* error);

// The ONDEVICE_SPEAKER_TURN_DETECTION pipeline: langid's front end, then a
// streaming encoder and a character transducer whose output marks speaker
// changes with a symbol of their own.
struct SpeakerTurnConfig {
  AgcConfig agc;
  FrontendConfig frontend;  // stack1 gives the encoder's input frames.
  VadConfig vad;            // With the segmenter's impatience.
  // SegmenterParams.reset_interval_ms: silence that ends a segment.
  int reset_interval_ms = 1000;
  // TFliteRunStreamParams.
  std::string encoder_file;      // Relative to SODAModels.
  std::string mean_stddev_file;  // Normalizes the encoder's input.
  int frame_batch_size = 1;
  int initial_activations_ignored = 0;
  int last_frame_repetitions = 0;  // Copies of the last frame at the end.
  DecoderConfig decoder;
  std::string decoder_file;
  std::string joint_file;
  std::string symbols_file;
  // SpeakerTurnStreamParams.
  int speaker_change_id = 0;
  int time_offset_ms = 0;
  int max_speaker_segment_ms = 0;
  int context_labels = 0;
  std::vector<std::string> short_responses_to_ignore;
};

bool LoadSpeakerTurnConfig(const std::string& path, SpeakerTurnConfig* config,
                           std::string* error);

// Parses a greco.PipelineDef config. Returns false (with *error set) if the
// file is unreadable or lacks the end-to-end model pieces.
bool LoadPipelineConfig(const std::string& path, PipelineConfig* config,
                        std::string* error);

// Reads an lstm.MeanStdDev proto (repeated float mean = 1, std_dev = 2).
bool LoadMeanStdDev(const std::string& path, std::vector<float>* mean,
                    std::vector<float>* stddev, std::string* error);

// Reads a whole file into *out.
bool ReadFile(const std::string& path, std::string* out);

}  // namespace soda

#endif  // SODA_CONFIG_H_
