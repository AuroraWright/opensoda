#include "recognizer.h"

#include <algorithm>
#include <cmath>
#include <sys/stat.h>

#include "pqrnn_spoken_punctuation.h"
#include "proto_wire.h"

namespace soda {
namespace {

constexpr int kModelSampleRate = 16000;
constexpr char kWordBoundary[] = "\xE2\x96\x81";  // U+2581 "▁".

bool IsDir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Decodes one UTF-8 code point starting at s[i]; advances i.
uint32_t NextCodePoint(const std::string& s, size_t* i) {
  const unsigned char c = s[*i];
  int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
  uint32_t cp = len == 1 ? c : c & (0x3F >> (len - 1));
  for (int k = 1; k < len && *i + k < s.size(); ++k) cp = (cp << 6) | (s[*i + k] & 0x3F);
  *i += len;
  return cp;
}

// Scripts written without spaces between words (CJK, kana, fullwidth forms).
bool IsSpacelessScript(uint32_t cp) {
  return (cp >= 0x2E80 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0xFF00 && cp <= 0xFFEF) || (cp >= 0x20000 && cp <= 0x2FFFF);
}

// Removes spaces adjacent to characters of spaceless scripts.
std::string GlueSpacelessScripts(const std::string& in) {
  std::vector<std::pair<uint32_t, std::string>> cps;
  for (size_t i = 0; i < in.size();) {
    size_t start = i;
    uint32_t cp = NextCodePoint(in, &i);
    cps.emplace_back(cp, in.substr(start, i - start));
  }
  std::string out;
  for (size_t k = 0; k < cps.size(); ++k) {
    if (cps[k].first == ' ') {
      const bool prev = k > 0 && IsSpacelessScript(cps[k - 1].first);
      const bool next = k + 1 < cps.size() && IsSpacelessScript(cps[k + 1].first);
      if (prev || next || out.empty() || k + 1 == cps.size()) continue;
    }
    out += cps[k].second;
  }
  return out;
}

// Whether two adjacent words are written without a space between them.
bool GlueWords(const std::string& left, const std::string& right) {
  if (left.empty() || right.empty()) return true;
  size_t i = left.size() - 1;
  while (i > 0 && (static_cast<unsigned char>(left[i]) & 0xC0) == 0x80) --i;
  size_t j = 0;
  return IsSpacelessScript(NextCodePoint(left, &i)) ||
         IsSpacelessScript(NextCodePoint(right, &j));
}

// The frames a CausalEncoderStream encodes next out of `ready`: the smallest
// of its batch sizes that fits, or once the segment is closed, the largest
// flush size (0: wait for more frames).
size_t BatchSize(const EncoderConfig& e, size_t ready, bool closed) {
  size_t n = 0;
  for (int b : e.time_batch_sizes) {
    if (static_cast<size_t>(b) <= ready && (n == 0 || static_cast<size_t>(b) < n)) n = b;
  }
  if (n == 0 && closed) {
    for (int b : e.flush_batch_sizes) {
      if (static_cast<size_t>(b) <= ready) n = std::max<size_t>(n, b);
    }
  }
  return n;
}

}  // namespace

std::unique_ptr<Recognizer> Recognizer::Create(const RecognizerOptions& options,
                                               std::string* error) {
  std::unique_ptr<Recognizer> r(new Recognizer());
  if (!r->Init(options, error)) return nullptr;
  return r;
}

bool Recognizer::Init(const RecognizerOptions& options, std::string* error) {
  options_ = options;
  std::string models = options.language_pack_dir;
  if (IsDir(models + "/SODAModels")) models += "/SODAModels";
  if (options_.config_name.empty()) {
    options_.config_name = "ONDEVICE_LARGE_CONTINUOUS";
    for (const char* name : {"ONDEVICE_LARGE_CONTINUOUS", "ONDEVICE_MEDIUM_CONTINUOUS",
                             "ONDEVICE_SMALL_CONTINUOUS"}) {
      struct stat st;
      if (stat((models + "/configs/" + name + ".config").c_str(), &st) == 0) {
        options_.config_name = name;
        break;
      }
    }
  }
  const std::string config_path = models + "/configs/" + options_.config_name + ".config";
  if (!LoadPipelineConfig(config_path, &config_, error)) return false;

  // Select the acoustic cluster ids for this recognition domain. libsoda
  // feeds them to the encoder and the VAD so the models know the audio's
  // provenance. Each AppendClusterIdStream looks the domain up in its own map
  // and otherwise uses the domain's enum value (FARFIELD 1, IME 2, TELEPHONY 3,
  // CAPTION 4, VOICE_SEARCH 7, HOTQUERY 10); the configured id is only for an
  // unspecified domain.
  auto cluster_id = [&](const std::map<std::string, int>& map, int* id) {
    static const std::map<std::string, int> kDomains = {
        {"FARFIELD", 1}, {"IME", 2},  {"TELEPHONY", 3}, {"CAPTION", 4},
        {"VOICE_SEARCH", 7}, {"HOTQUERY", 10}};
    if (*id < 0) return;
    if (auto it = map.find(options.domain); it != map.end()) {
      *id = it->second;
    } else if (auto d = kDomains.find(options.domain); d != kDomains.end()) {
      *id = d->second;
    }
  };
  cluster_id(config_.frontend_cluster_id_by_domain, &config_.frontend.cluster_id);
  cluster_id(config_.vad_cluster_id_by_domain, &config_.vad.cluster_id);

  if (config_.agc.activated && config_.agc.estimator != 3) {
    *error = "unsupported AGC gain estimator " + std::to_string(config_.agc.estimator);
    return false;
  }
  agc_.Init(config_.agc, kModelSampleRate);

  std::vector<float> mean, stddev;
  if (!LoadMeanStdDev(models + "/" + config_.frontend.mean_stddev_file, &mean, &stddev,
                      error) ||
      !frontend_.Init(config_.frontend, kModelSampleRate, mean, stddev, error) ||
      !encoder_.Init(models + "/" + config_.encoder.file, options.num_threads, error) ||
      !decoder_.Init(models + "/" + config_.decoder_file, 1, error) ||
      !joint_.Init(models + "/" + config_.joint_file, 1, error) ||
      !symbols_.Load(models + "/" + config_.symbols_file, error)) {
    return false;
  }
  cascade_ = !config_.cascade.encoder.file.empty();
  if (cascade_ &&
      (!cascade_encoder_.Init(models + "/" + config_.cascade.encoder.file, options.num_threads, error) ||
       !decoder2_.Init(models + "/" + config_.cascade.decoder_file, 1, error) ||
       !joint2_.Init(models + "/" + config_.cascade.joint_file, 1, error))) {
    return false;
  }
  if (config_.encoder.default_domain_id >= 0) {
    auto it = config_.encoder.domain_ids.find(options.domain);
    encoder_.set_domain_id(it != config_.encoder.domain_ids.end() ? it->second
                                                                   : config_.encoder.default_domain_id);
  }
  if (frontend_.output_dim() != encoder_.input_dim()) {
    *error = "frontend produces " + std::to_string(frontend_.output_dim()) +
             "-dim frames but the encoder expects " + std::to_string(encoder_.input_dim());
    return false;
  }
  frame_ms_ = frontend_.output_frame_ms();
  stacked_frame_ms_ = frontend_.stacked_frame_ms();
  frontend_.set_output_end_sink(&frame_ends_);
  if (options.use_vad && !config_.vad.model_file.empty()) {
    vad_ = std::make_unique<VoiceActivityDetector>();
    std::string vad_error;
    if (vad_->Init(models, config_.vad, &vad_error)) {
      endpointer_.Init(config_.vad.endpointer);
      if (config_.vad.spectral_noise) {
        vad_frontend_.Init(config_.vad, frontend_.num_fft_bins(), kModelSampleRate);
        frontend_.set_power_tap(&vad_power_, config_.vad.noise_scale);
        vad_frame_ms_ = config_.frontend.frame_step_ms;
      } else {
        frontend_.set_stacked_tap(&vad_input_);
        vad_frame_ms_ = stacked_frame_ms_;
      }
      frontend_.set_gated(true);
    } else {
      vad_.reset();
    }
  }
  if (options.enable_language_id) {
    langid_ = LanguageIdentifier::Create(models, error);
    if (!langid_) return false;
  }
  // Words are written with spaces when the punctuation settings segment text
  // by whitespace (BasicTextSegmenterConfig field 3 = SIMPLE_WHITESPACE) rather
  // than by character (ja-JP).
  std::string punctuation_settings;
  if (ReadFile(models + "/SODA_punctuation_config.pb", &punctuation_settings)) {
    ProtoReader r(punctuation_settings);
    while (r.Next()) {
      if (r.field() != 7) continue;
      ProtoReader seg(r.bytes());
      while (seg.Next()) {
        if (seg.field() == 1) punctuation_marks_.push_back(seg.str());
        if (seg.field() == 3) spaced_words_ = seg.int64() == 1;
      }
    }
    // Partials list characters unless the settings segment by whitespace;
    // a pack without settings (th-TH) lists words.
    character_partials_ = !spaced_words_;
  }
  if (options.enable_punctuation) {
    punctuator_ = std::make_unique<Punctuator>();
    std::string ignored;
    if (!punctuator_->Init(models, &ignored)) punctuator_.reset();
  }
  if (!config_.capitalization_transducer_file.empty() &&
      !config_.capitalization_lm_file.empty()) {
    truecaser_ = std::make_unique<Truecaser>();
    std::string ignored;
    if (!truecaser_->Init(models + "/" + config_.capitalization_transducer_file,
                          models + "/" + config_.capitalization_lm_file, &ignored)) {
      truecaser_.reset();
    }
  }
  if (const auto& c = config_.capitalization_model; !c.file.empty()) {
    capitalizer_ = std::make_unique<RoadrunnerCapitalizer>();
    std::string ignored;
    if (!capitalizer_->Init(models + "/" + c.file, c.window, c.left, c.right, &ignored)) capitalizer_.reset();
  }

  // libsoda turns spoken punctuation ("{comma}") and emoji ("{smiley-face}")
  // into symbols only in dictation; in captions they stay as written.
  for (const auto& d : config_.denormalizers) {
    constexpr int kSpokenPunctuation = 3, kSpokenEmoji = 4;
    if ((d.type == kSpokenPunctuation || d.type == kSpokenEmoji) && options.domain != "IME") continue;
    if (d.type != 0 && d.type != kSpokenPunctuation && d.type != kSpokenEmoji) continue;
    std::string ignored;
    std::unique_ptr<TextNormalizer> stage;
    if (!d.spoken_punctuation_settings.empty()) {
      auto n = std::make_unique<PqrnnSpokenPunctuationNormalizer>();
      if (!n->Init(models + "/" + d.file, d.spoken_punctuation_settings, &ignored)) continue;
      stage = std::move(n);
    } else {
      auto n = std::make_unique<FstNormalizer>();
      if (!n->Init(models + "/" + d.file, &ignored)) continue;
      n->set_class_spans(d.class_normalizer);
      keep_class_tags_ |= d.class_normalizer;
      stage = std::move(n);
    }
    (d.before_capitalization ? precase_denormalizers_ : denormalizers_).push_back(std::move(stage));
  }
  const PipelineConfig::SpokenPunctuation& sp = config_.spoken_punctuation;
  if (!sp.model_file.empty() && (sp.type != 3 || options.domain == "IME")) {
    spoken_punctuation_ = std::make_unique<SpokenPunctuationNormalizer>();
    std::string ignored;
    if (!spoken_punctuation_->Init(models + "/" + sp.converter_file, models + "/" + sp.symbol_file,
                                   models + "/" + sp.model_file, &ignored)) {
      spoken_punctuation_.reset();
    }
  }
  if (options.mask_offensive_words && !config_.offensive_word_masker_file.empty()) {
    masker_ = std::make_unique<FstNormalizer>();
    std::string ignored;
    if (!masker_->Init(models + "/" + config_.offensive_word_masker_file, &ignored)) {
      masker_.reset();
    }
  }

  auto search_options = [&](const DecoderConfig& d) {
    BeamSearchOptions bo;
    bo.max_arcs = options.beam > 0 ? options.beam : d.max_arcs;
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
  };
  search_ = std::make_unique<BeamSearch>(&decoder_, &joint_, search_options(config_.decoder),
                                         config_.decoder);
  if (cascade_) {
    search2_ = std::make_unique<BeamSearch>(&decoder2_, &joint2_, search_options(config_.cascade.decoder),
                                            config_.cascade.decoder);
  }

  if (!config_.biasing.wpm_file.empty()) {
    std::string ignored;
    if (wpm_.Load(models + "/" + config_.biasing.wpm_file, &ignored)) {
      has_wpm_ = true;
      SetPhrases(options.phrases);
    }
  }

  if (options.sample_rate == 3 * kModelSampleRate) {
    // libsoda's 48 kHz -> 16 kHz resampler: a fixed 17-tap symmetric FIR
    // centred on every third input sample (its DC gain is 1.13), whose
    // output is truncated back to 16-bit values.
    resample_ratio_ = 3.0;
    resample_taps_ = 16;
    resample_phases_ = 0;
  } else if (options.sample_rate != kModelSampleRate) {
    resample_ratio_ = static_cast<double>(options.sample_rate) / kModelSampleRate;
    // A Blackman-windowed sinc over 32 samples of the lower rate (more input
    // taps when downsampling), keeping the passband flat up to the
    // filterbank's upper band limit.
    resample_taps_ = 2 * static_cast<int>(std::ceil(16 * std::max(1.0, resample_ratio_)));
    resample_phases_ = 256;
    const double cutoff = 0.95 * std::min(1.0, 1.0 / resample_ratio_);
    resample_kernel_.resize((resample_phases_ + 1) * resample_taps_);
    for (int ph = 0; ph <= resample_phases_; ++ph) {
      const double frac = static_cast<double>(ph) / resample_phases_;
      for (int j = 0; j < resample_taps_; ++j) {
        const double t = (j - resample_taps_ / 2 + 1) - frac;
        const double x = M_PI * cutoff * t;
        const double sinc = std::fabs(t) < 1e-9 ? 1.0 : std::sin(x) / x;
        const double w = 0.42 + 0.5 * std::cos(M_PI * t / (resample_taps_ / 2)) +
                         0.08 * std::cos(2 * M_PI * t / (resample_taps_ / 2));
        resample_kernel_[ph * resample_taps_ + j] =
            static_cast<float>(cutoff * sinc * (std::fabs(t) < resample_taps_ / 2 ? w : 0));
      }
    }
  }
  Reset();
  return true;
}

void Recognizer::ResetPipeline() {
  agc_.Reset();
  frontend_.Reset();
  if (vad_) vad_->Reset();
  endpointer_.Reset();
  vad_input_.clear();
  vad_power_.clear();
  vad_frontend_.Reset();
  first_batch_ = true;
  backlog_ = 0;
  vad_frames_ = 0;
  decided_frames_ = 0;
  in_segment_ = false;
  segment_first_frame_ = 0;
  pending_frames_.clear();
  pending_ends_.clear();
  frame_ends_.clear();
  queued_finals_ = 0;
}

void Recognizer::SetPhrases(const std::vector<BiasPhrase>& phrases) {
  if (!has_wpm_) return;
  ContextBiaser::Params p;
  // A lowercased copy of each phrase (the model's TO_LOWER text
  // transformer) would make the results differ from libsoda's.
  p.token_weight = config_.biasing.token_weight;
  p.respect_boost = config_.biasing.respect_boost;
  p.retwiddle_penalty = config_.biasing.retwiddle_penalty;
  // libsoda ignores phrases sent under any other context name.
  std::vector<BiasPhrase> used;
  for (const BiasPhrase& phrase : phrases) {
    if (phrase.context == config_.biasing.context_name) used.push_back(phrase);
  }
  biaser_.Build(used, wpm_, symbols_, p);
  search_->set_biaser(&biaser_);
  if (search2_) search2_->set_biaser(&biaser_);
}

void Recognizer::Reset() {
  ResetPipeline();
  encoder_.Reset();
  search_->Reset();
  ResetCascade();
  if (langid_) langid_->Reset();
  history_.clear();
  history_start_ = origin_ = 0;
  restart_at_ = -1;
  frame_ends_ms_.clear();
  decoded_end_ms_ = 0;
  segment_start_ms_ = last_final_end_ms_ = 0;
  last_partial_.clear();
  had_final_ = false;
  speech_started_ = session_started_ = false;
  last_endpoint_ms_ = -1;
  ResetLevels();
  resample_history_.assign(resample_taps_ / 2, 0.0f);
  resample_pos_ = resample_taps_ / 2;
}

void Recognizer::Resample(const float* in, size_t n, std::vector<float>* out,
                          std::vector<float>* unrounded) {
  if (resample_phases_ == 0) {
    // It works on samples scaled to [-1, 1] by 1/32767.
    for (size_t i = 0; i < n; ++i) resample_history_.push_back(in[i] * (1.0f / 32767.0f));
  } else {
    resample_history_.insert(resample_history_.end(), in, in + n);
  }
  const int half = resample_taps_ / 2;
  while (true) {
    const long i0 = static_cast<long>(std::floor(resample_pos_));
    if (i0 + half >= static_cast<long>(resample_history_.size())) break;
    if (resample_phases_ == 0) {
      // Coefficients for offsets 0..8 (bit patterns as in libsoda).
      static constexpr float kTaps[9] = {0.358154296875f, 0.29522705078125f, 0.14068603515625f,
                                         0.0f, -0.06640625f, -0.0458373986f, 0.0f,
                                         0.03125f, 0.03125f};
      const float* x = &resample_history_[i0];
      float acc = x[0] * kTaps[0];
      for (int k : {1, 2, 4, 5}) acc += (x[k] + x[-k]) * kTaps[k];
      acc += (x[7] + x[-7] + x[8] + x[-8]) * kTaps[7];
      acc = std::clamp(acc, -1.0f, 1.0f);
      out->push_back(std::trunc(acc * 32767.0f));
      if (unrounded) unrounded->push_back(acc);
      resample_pos_ += resample_ratio_;
      continue;
    }
    const double frac = resample_pos_ - i0;
    const float* k = &resample_kernel_[static_cast<int>(std::lround(frac * resample_phases_)) *
                                       resample_taps_];
    const float* x = &resample_history_[i0 - half + 1];
    float acc = 0.0f;
    for (int j = 0; j < resample_taps_; ++j) acc += k[j] * x[j];
    out->push_back(acc);
    resample_pos_ += resample_ratio_;
  }
  // The next output reads back `half` samples (the FIR one more).
  const long keep_from = static_cast<long>(std::floor(resample_pos_)) - half;
  if (keep_from > 0) {
    resample_history_.erase(resample_history_.begin(), resample_history_.begin() + keep_from);
    resample_pos_ -= keep_from;
  }
}

void Recognizer::AddAudio(const int16_t* samples, size_t count) {
  const int ch = std::max(1, options_.channel_count);
  const size_t frames = count / ch;
  std::vector<float> mono(frames);
  for (size_t i = 0; i < frames; ++i) {
    float acc = 0.0f;
    for (int c = 0; c < ch; ++c) acc += samples[i * ch + c];
    mono[i] = acc / ch;
  }
  AddAudioFloat(mono.data(), mono.size());
}

void Recognizer::set_language_callback(LanguageIdentifier::Callback cb) {
  if (langid_) langid_->set_callback(std::move(cb));
}

void Recognizer::AddAudioFloat(const float* samples, size_t count) {
  std::vector<std::vector<float>> frames;
  std::vector<float> resampled, unrounded;
  if (options_.sample_rate != kModelSampleRate) {
    Resample(samples, count, &resampled, &unrounded);
    samples = resampled.data();
    count = resampled.size();
  }
  if (langid_) langid_->Accept(samples, count);
  if (vad_) {
    // libsoda measures the 48 kHz resampler's output before rounding.
    if (unrounded.empty()) {
      ReportLevels(samples, count, 1.0 / 32767.0);
    } else {
      ReportLevels(unrounded.data(), unrounded.size(), 1.0);
    }
    history_.insert(history_.end(), samples, samples + count);
    const size_t keep = 30 * kModelSampleRate;
    if (history_.size() > 2 * keep) {
      const size_t drop = history_.size() - keep;
      history_.erase(history_.begin(), history_.begin() + drop);
      history_start_ += drop;
    }
  }
  Process(samples, count);
}

void Recognizer::ReportLevels(const float* samples, size_t count, double scale) {
  // libsoda smooths the mean square of each 20 ms block (computed in double
  // precision, samples at full scale 1) with exp(-20 ms / 100 ms).
  constexpr int kBlock = kModelSampleRate / 50;
  const double decay = std::exp(-0.2);
  for (size_t i = 0; i < count; ++i) {
    const double x = samples[i] * scale;
    level_block_sum_ += x * x;
    if (++level_block_samples_ < kBlock) continue;
    level_power_ = decay * level_power_ + (1.0 - decay) * (level_block_sum_ / kBlock);
    level_block_sum_ = 0.0;
    level_block_samples_ = 0;
    if (!audio_level_callback_ || !config_.vad.audio_level_events) continue;
    AudioLevelEvent e;
    e.rms = static_cast<float>(std::sqrt(level_power_));
    // The posterior goes with the first report after the VAD produced it
    // (libsoda's level stream runs on its own thread).
    e.has_speech_level = level_pending_;
    e.speech_level = pending_level_;
    e.speech_level_ms = pending_level_ms_;
    level_pending_ = false;
    audio_level_callback_(e);
  }
}

void Recognizer::ResetLevels() {
  level_power_ = level_block_sum_ = 0.0;
  // The stream libsoda measures from 48 kHz audio starts two samples
  // earlier than the one it recognizes.
  level_block_samples_ = resample_phases_ == 0 && resample_ratio_ == 3.0 ? 2 : 0;
  level_pending_ = false;
}

void Recognizer::Process(const float* samples, size_t count) {
  session_started_ = true;
  std::vector<std::vector<float>> frames;
  std::vector<float> audio(samples, samples + count);
  agc_.Process(audio.data(), audio.size());
  frontend_.Accept(audio.data(), audio.size(), &frames);
  RunVad(&frames);
  Enqueue(&frames);
  RunEncoder(/*flush=*/false);
  while (restart_at_ >= 0) Restart();
}

void Recognizer::Restart() {
  const int64_t from = std::max(restart_at_, history_start_);
  restart_at_ = -1;
  ResetPipeline();
  encoder_.Reset();
  origin_ = from;
  const int64_t end = history_start_ + static_cast<int64_t>(history_.size());
  if (from >= end) return;
  std::vector<std::vector<float>> frames;
  std::vector<float> audio(history_.begin() + (from - history_start_), history_.end());
  agc_.Process(audio.data(), audio.size());
  frontend_.Accept(audio.data(), audio.size(), &frames);
  RunVad(&frames);
  Enqueue(&frames);
  RunEncoder(/*flush=*/false);
}

void Recognizer::Finish() {
  bool open = false;  // The audio ends inside a segment.
  while (true) {
    while (restart_at_ >= 0) Restart();
    std::vector<std::vector<float>> frames;
    frontend_.Flush(&frames);
    RunVad(&frames, /*end_of_audio=*/true);
    if (vad_) {
      // Frames still awaiting a decision belong to the segment in progress.
      const bool speech = in_segment_ || endpointer_.state() != Endpointer::kPreSpeech;
      while (frontend_.held() > 1) Segment(speech, &frames);
      if (speech && !in_segment_ && frontend_.held() > 0) Segment(true, &frames);
      open = in_segment_;
      if (in_segment_) {
        // libsoda closes a segment the audio ends in as if speech ended at
        // the end-of-audio event, on the last VAD frame.
        in_segment_ = false;
        frontend_.Release(false, &frames);
        ++decided_frames_;
        EndSegment(&frames);
      } else {
        while (frontend_.held() > 0) Segment(false, &frames);
      }
    }
    Enqueue(&frames);
    RunEncoder(/*flush=*/true);
    if (restart_at_ < 0) break;
  }
  if (langid_) langid_->Finish();
  // libsoda ends the stream a VAD frame before the end of the last one, and
  // sends that before the final of a segment the audio ends in.
  if (vad_) {
    SendEndpoint(EndpointEvent::kEndOfAudio,
                 origin_ * 1000 / kModelSampleRate +
                     static_cast<int64_t>(std::max<int64_t>(vad_frames_ - 1, 0) * vad_frame_ms_),
                 /*force=*/true);
  }
  EmitFinal(/*end_of_audio=*/true, /*segment_end=*/open);
  restart_at_ = -1;
  if (vad_) {
    // The audio's last, partial block counts as a whole one.
    if (level_block_samples_ > 0) {
      ReportLevels(std::vector<float>(kModelSampleRate / 50 - level_block_samples_, 0.0f).data(),
                   kModelSampleRate / 50 - level_block_samples_, 1.0);
    }
    if (level_pending_ && audio_level_callback_ && config_.vad.audio_level_events) {
      AudioLevelEvent e;
      e.has_rms = false;
      e.has_speech_level = true;
      e.speech_level = pending_level_;
      e.speech_level_ms = pending_level_ms_;
      audio_level_callback_(e);
    }
  }
  ResetLevels();
  // The next audio starts a fresh stream.
  frontend_.Reset();
  encoder_.Reset();
  if (vad_) vad_->Reset();
  endpointer_.Reset();
  vad_frontend_.Reset();
  if (langid_) langid_->Reset();
  decoder_.ClearCache();
  if (cascade_) decoder2_.ClearCache();
  vad_frames_ = decided_frames_ = segment_first_frame_ = 0;
  first_batch_ = true;
  backlog_ = 0;
  decoded_end_ms_ = segment_start_ms_ = last_final_end_ms_ = 0;
  history_.clear();
  history_start_ = origin_ = 0;
}

void Recognizer::ContinueFrom(Recognizer* previous) {
  if (previous->langid_) {
    langid_ = std::move(previous->langid_);
  }
  history_ = std::move(previous->history_);
  history_start_ = previous->history_start_;
  resample_pos_ = previous->resample_pos_;
  resample_history_ = previous->resample_history_;
  level_power_ = previous->level_power_;
  level_block_sum_ = previous->level_block_sum_;
  level_block_samples_ = previous->level_block_samples_;
  const int64_t from_ms = previous->last_final_end_ms_;
  decoded_end_ms_ = segment_start_ms_ = last_final_end_ms_ = from_ms;
  session_started_ = true;
  restart_at_ = from_ms * kModelSampleRate / 1000;
  while (restart_at_ >= 0) Restart();
}

void Recognizer::Enqueue(std::vector<std::vector<float>>* frames) {
  for (size_t i = 0; i < frames->size(); ++i) {
    std::vector<float>& f = (*frames)[i];
    if (feature_tap_ && !f.empty()) feature_tap_(f);
    pending_frames_.push_back(std::move(f));
    pending_ends_.push_back(frame_ends_[i]);
  }
  frames->clear();
  frame_ends_.clear();
}

void Recognizer::RunVad(std::vector<std::vector<float>>* frames, bool end_of_audio) {
  if (!vad_) return;
  if (config_.vad.spectral_noise) {
    for (const auto& power : vad_power_) vad_frontend_.Push(power, &vad_input_);
    vad_power_.clear();
    if (end_of_audio) vad_frontend_.Flush(&vad_input_);
  }
  std::vector<bool> decisions;
  for (const auto& frame : vad_input_) {
    ++vad_frames_;
    const float p = vad_->Process(frame);
    endpointer_.Process(p);
    level_pending_ = true;
    pending_level_ = p;
    pending_level_ms_ = origin_ * 1000 / kModelSampleRate + static_cast<int64_t>((vad_frames_ - 1) * vad_frame_ms_);
    // greco's SegmenterStream passes the frames the endpointer labels speech
    // on to the encoder and ends a segment at the first silent one.
    endpointer_.TakeDecisions(&decisions);
    for (bool speech : decisions) Segment(speech, frames);
    decisions.clear();
    // It also cuts segments at impatience_limit. (impatience_start only
    // shortens the trailing silence a segment needs to end, which is already
    // zero: any non-speech decision ends one.)
    const VadConfig& vc = config_.vad;
    const int64_t segment_ms = static_cast<int64_t>((decided_frames_ - segment_first_frame_) * vad_frame_ms_);
    if (in_segment_ && vc.impatience_limit_ms > 0 && segment_ms >= vc.impatience_limit_ms) {
      EndSegment(frames);
      MarkFinal(frames);
      segment_first_frame_ = decided_frames_;
    }
    if (vad_tap_) {
      vad_tap_(origin_ * 1000 / kModelSampleRate + static_cast<int64_t>(vad_frames_ * vad_frame_ms_), p,
               endpointer_.state());
    }
  }
  vad_input_.clear();
}

void Recognizer::Segment(bool speech, std::vector<std::vector<float>>* frames) {
  const int64_t origin_ms = origin_ * 1000 / kModelSampleRate;
  if (speech && !in_segment_) {
    in_segment_ = true;
    segment_first_frame_ = decided_frames_;
    const int64_t onset_ms = origin_ms + static_cast<int64_t>(decided_frames_ * vad_frame_ms_);
    // The VAD runs ahead of the decoder: an onset found while the last
    // segment's end still waits to be decoded is decided again after the
    // restart that end brings, and the final keeps its own start.
    if (queued_finals_ == 0) segment_start_ms_ = onset_ms;
    SendEndpoint(EndpointEvent::kStartOfSpeech, onset_ms);
  }
  frontend_.Release(speech, frames);
  ++decided_frames_;
  if (!speech && in_segment_) {
    in_segment_ = false;
    SendEndpoint(EndpointEvent::kEndOfSpeech,
                 origin_ms + static_cast<int64_t>((decided_frames_ - 1) * vad_frame_ms_));
    EndSegment(frames);
    MarkFinal(frames);
  }
}

void Recognizer::SendEndpoint(EndpointEvent::Type type, int64_t audio_ms, bool force) {
  // A restart decides the frames since the last final again; report each
  // onset and end once. (A VAD that reads speech from its first frames, as
  // ja-JP's and de-DE's do with the caption cluster id, starts speech where
  // each restart begins.) An onset the restart moves is reported again:
  // libsoda's VAD runs ahead of its decoder, so it may report the next
  // onset before the end of an empty segment restarts recognition earlier.
  const bool start = type == EndpointEvent::kStartOfSpeech;
  if (!force && start == speech_started_ && audio_ms == last_endpoint_ms_) return;
  speech_started_ = start;
  last_endpoint_ms_ = audio_ms;
  if (endpoint_callback_) endpoint_callback_({type, audio_ms});
}

void Recognizer::EndSegment(std::vector<std::vector<float>>* frames) {
  frontend_.EndSegment(frames);
}

void Recognizer::MarkFinal(std::vector<std::vector<float>>* frames) {
  frames->emplace_back();  // Finalizes the result when the decoder reaches it.
  frame_ends_.push_back(-1);
  ++queued_finals_;
}

void Recognizer::RunEncoder(bool flush) {
  // libsoda's CausalEncoderStream (time_batch_size, time_batch_size_on_flush)
  // encodes the largest batch of ready frames it can, and the encoder's
  // dynamic int8 quantization makes its outputs depend on the batching. When
  // it keeps up with live audio, frames reach it a few at a time and it
  // mostly encodes the smallest batch; encoding that whatever is queued
  // reproduces Chrome's results best (and does not depend on how the caller
  // chunks the audio).
  std::string error;
  while (!pending_frames_.empty()) {
    if (pending_frames_.front().empty()) {
      pending_frames_.erase(pending_frames_.begin());
      pending_ends_.erase(pending_ends_.begin());
      --queued_finals_;
      first_batch_ = true;
      EmitFinal(/*end_of_audio=*/false, /*segment_end=*/vad_ != nullptr);
      if (restart_at_ >= 0) return;
      continue;
    }
    size_t ready = 0;
    while (ready < pending_frames_.size() && !pending_frames_[ready].empty()) ++ready;
    const bool closed = flush || ready < pending_frames_.size();
    size_t n = BatchSize(config_.encoder, ready, closed);
    // Where the segmenter gates 10 ms frames (ko-KR, hi-IN, pt-BR,
    // cmn-Hans-CN), it releases a backlog at a segment's start. libsoda's
    // encoder thread wakes on its first frames and encodes the smallest
    // batch, then the largest batch that fits the rest, which has arrived
    // meanwhile (as logged from libsoda at real-time speed: 2, then 8 of 8,
    // or 4 of 6 and 2). ja-JP, en-US and de-DE match best without.
    if (first_batch_ && n > 0 && config_.frontend.subsample_after_gate) {
      backlog_ = ready - n;
    } else if (backlog_ > 0 && n > 0) {
      for (int b : config_.encoder.time_batch_sizes) {
        if (static_cast<size_t>(b) <= backlog_) n = std::max<size_t>(n, b);
      }
      backlog_ -= std::min(backlog_, n);
    }
    if (n == 0) break;
    first_batch_ = false;
    std::vector<std::vector<float>> chunk(pending_frames_.begin(), pending_frames_.begin() + n);
    std::vector<long> ends(pending_ends_.begin(), pending_ends_.begin() + n);
    pending_frames_.erase(pending_frames_.begin(), pending_frames_.begin() + n);
    pending_ends_.erase(pending_ends_.begin(), pending_ends_.begin() + n);
    std::vector<std::vector<float>> encoded;
    if (!encoder_.Encode(chunk, &encoded, &error)) return;
    Decode(encoded, ends);
  }
}

void Recognizer::Decode(const std::vector<std::vector<float>>& encoded,
                        const std::vector<long>& ends) {
  const int endpoint_frames =
      options_.endpoint_silence_ms > 0
          ? std::max(1, static_cast<int>(options_.endpoint_silence_ms / frame_ms_))
          : 0;
  for (size_t i = 0; i < encoded.size(); ++i) {
    search_->Advance(encoded[i]);
    decoded_end_ms_ = origin_ * 1000 / kModelSampleRate +
                      static_cast<int64_t>(ends[std::min(i, ends.size() - 1)] * stacked_frame_ms_);
    frame_ends_ms_.push_back(decoded_end_ms_);
    if (cascade_) {
      last_first_pass_frame_ = encoded[i];
      cascade_pending_.push_back(encoded[i]);
      cascade_input_ends_ms_.push_back(decoded_end_ms_);
      RunCascade(/*flush=*/false);
    }
    const Hypothesis& best = search_->Best();
    if (!best.labels.empty() && best.labels.back() == config_.decoder.eos_id) {
      // The model ended the utterance with </S>.
      EmitFinal(/*end_of_audio=*/false);
      continue;
    }
    if (!vad_ && endpoint_frames > 0 && !best.labels.empty() &&
        search_->frames() - 1 - best.label_frames.back() >= endpoint_frames) {
      EmitFinal(/*end_of_audio=*/false);
      continue;
    }
    if (!best.labels.empty()) EmitPartial();
  }
}

std::string Recognizer::LabelsToText(const std::vector<int>& labels) const {
  std::string text;
  for (int id : labels) {
    if (id <= 0 || id >= symbols_.size()) continue;
    const std::string& s = symbols_.Symbol(id);
    if (s.size() > 2 && s.front() == '<' && s.back() == '>') continue;
    if (s.compare(0, 3, kWordBoundary) == 0) {
      text += ' ';
      text += s.substr(3);
    } else {
      text += s;
    }
  }
  return GlueSpacelessScripts(text);
}

void Recognizer::RunCascade(bool flush) {
  const int right_context = config_.cascade.right_context;
  std::string error;
  while (!cascade_pending_.empty()) {
    const size_t n = BatchSize(config_.cascade.encoder, cascade_pending_.size(), flush);
    if (n == 0) break;
    std::vector<std::vector<float>> chunk(cascade_pending_.begin(), cascade_pending_.begin() + n);
    cascade_pending_.erase(cascade_pending_.begin(), cascade_pending_.begin() + n);
    std::vector<std::vector<float>> encoded;
    if (!cascade_encoder_.Encode(chunk, &encoded, &error)) return;
    // libsoda's EncoderUnshiftStream: the cascaded encoder's output for a
    // frame comes right_context frames later; it takes that frame's time.
    for (const auto& frame : encoded) {
      const size_t k = cascade_outputs_++;
      if (k < static_cast<size_t>(right_context)) continue;
      search2_->Advance(frame);
      frame_ends2_ms_.push_back(cascade_input_ends_ms_[k - right_context]);
    }
  }
}

void Recognizer::FlushCascade() {
  RunCascade(/*flush=*/true);
  // EncoderInsertRightContextStream: at a segment end, right_context copies
  // of the last frame (or zeros) let the last frames through.
  if (cascade_input_ends_ms_.empty()) return;
  const std::vector<float> pad = config_.cascade.use_last_frame
                                     ? last_first_pass_frame_
                                     : std::vector<float>(last_first_pass_frame_.size(), 0.0f);
  for (int i = 0; i < config_.cascade.right_context; ++i) cascade_pending_.push_back(pad);
  RunCascade(/*flush=*/true);
}

void Recognizer::ResetCascade() {
  if (!cascade_) return;
  cascade_encoder_.Reset();
  search2_->Reset();
  cascade_pending_.clear();
  cascade_input_ends_ms_.clear();
  frame_ends2_ms_.clear();
  cascade_outputs_ = 0;
}

void Recognizer::Describe(const Hypothesis& h, bool final, RecognitionEvent* e) const {
  Describe(h, final, frame_ends_ms_, e);
}

void Recognizer::Describe(const Hypothesis& h, bool final, const std::vector<int64_t>& frame_ends_ms,
                          RecognitionEvent* e) const {
  e->spaced_words = spaced_words_;
  e->raw_text = LabelsToText(h.labels);
  // Words start at pieces carrying the boundary marker.
  WordTiming* current = nullptr;
  for (size_t i = 0; i < h.labels.size(); ++i) {
    const int id = h.labels[i];
    if (id <= 0 || id >= symbols_.size()) continue;
    const std::string& s = symbols_.Symbol(id);
    if (s.size() > 2 && s.front() == '<' && s.back() == '>') {
      std::string name;
      bool closing;
      if (keep_class_tags_ && FstNormalizer::ClassTag(s, &name, &closing)) {
        e->words.push_back({});
        e->words.back().raw_text = s;
        e->words.back().start_ms = -1;  // The next word's start.
        current = nullptr;
      }
      continue;
    }
    const bool boundary = s.compare(0, 3, kWordBoundary) == 0;
    if (boundary || current == nullptr) {
      e->words.push_back({});
      current = &e->words.back();
      current->start_ms = -1;
      current->joined = s == kWordBoundary;
    }
    // libsoda starts a word at the start of the frame where the label before
    // its first piece of text was emitted (a bare word boundary is a label
    // of its own), and the first label of a segment at the segment start.
    if (current->start_ms < 0 && s != kWordBoundary) {
      current->start_ms = segment_start_ms_;
      if (i > 0 && !frame_ends_ms.empty()) {
        const size_t f = std::min<size_t>(h.label_frames[i - 1], frame_ends_ms.size() - 1);
        current->start_ms = std::max<int64_t>(segment_start_ms_, frame_ends_ms[f] - static_cast<int64_t>(frame_ms_));
      }
    }
    current->raw_text += boundary ? s.substr(3) : s;
  }
  // A trailing bare boundary or class tag.
  for (size_t i = e->words.size(); i-- > 0;) {
    if (e->words[i].start_ms < 0) {
      e->words[i].start_ms = i + 1 < e->words.size() ? e->words[i + 1].start_ms : decoded_end_ms_;
    }
  }
  // In packs punctuated by character (ja-JP, cmn-*), libsoda's raw text has
  // no spaces in partials (`テスト@GmailCO`), and in finals none before a
  // word that began at a bare word boundary; elsewhere the scripts decide.
  auto rejoin = [this, e, final] {
    e->raw_text.clear();
    for (size_t i = 0; i < e->words.size(); ++i) {
      const WordTiming& w = e->words[i];
      const bool glue = character_partials_ && (!final || w.joined);
      if (i > 0 && !glue && !GlueWords(e->words[i - 1].raw_text, w.raw_text)) e->raw_text += ' ';
      e->raw_text += w.raw_text;
    }
  };
  if (character_partials_ && !e->words.empty()) rejoin();
  // The config's denormalizer list fixes the order: grammars listed before
  // the capitalization normalizer see the words before truecasing.
  if (!precase_denormalizers_.empty() && !e->words.empty()) {
    Denormalize(final, precase_denormalizers_, &e->words);
    rejoin();
  }
  if ((truecaser_ || capitalizer_) && !e->words.empty()) {
    std::vector<std::string> cased;
    for (const auto& w : e->words) cased.push_back(w.raw_text);
    if (truecaser_) {
      truecaser_->Apply(&cased);
    } else {
      capitalizer_->Apply(&cased);
    }
    for (size_t i = 0; i < cased.size(); ++i) e->words[i].raw_text = cased[i];
    rejoin();
  }
  if (!denormalizers_.empty() && !e->words.empty()) {
    Denormalize(final, denormalizers_, &e->words);
    rejoin();
  }
  if (spoken_punctuation_ && !e->words.empty()) {
    std::vector<SpokenPunctuationNormalizer::Word> words;
    for (size_t i = 0; i < e->words.size(); ++i) {
      const int64_t end = i + 1 < e->words.size() ? e->words[i + 1].start_ms : decoded_end_ms_;
      words.push_back({e->words[i].raw_text, e->words[i].start_ms, std::max(end, e->words[i].start_ms)});
    }
    if (spoken_punctuation_->Apply(&words)) {
      e->words.clear();
      e->raw_text.clear();
      for (const auto& w : words) {
        if (!e->words.empty() && !GlueWords(e->words.back().raw_text, w.text)) e->raw_text += ' ';
        e->raw_text += w.text;
        WordTiming t;
        t.raw_text = w.text;
        t.start_ms = w.start_ms;
        e->words.push_back(t);
      }
    }
  }
  std::vector<std::string> words;
  for (const auto& w : e->words) words.push_back(w.raw_text);
  if (punctuator_) punctuator_->Punctuate(&words, final);
  // libsoda masks last, over the punctuated words joined by spaces; the
  // grammar only matches whole words.
  if (masker_ && !words.empty()) {
    std::string joined;
    for (const auto& w : words) joined += (joined.empty() ? "" : " ") + w;
    std::vector<std::string> masked;
    const std::string out = masker_->Apply(joined);
    for (size_t i = 0, j; i <= out.size(); i = j + 1) {
      j = std::min(out.find(' ', i), out.size());
      masked.push_back(out.substr(i, j - i));
    }
    if (masked.size() == words.size()) {
      words = std::move(masked);
    } else {
      for (auto& w : words) w = masker_->Apply(w);
    }
  }
  for (size_t i = 0; i < words.size(); ++i) {
    e->words[i].text = words[i];
    if (i > 0 && !GlueWords(words[i - 1], words[i])) e->text += ' ';
    e->text += words[i];
    if (final || spaced_words_) {
      if (i > 0) e->spaced_text += ' ';
      e->spaced_text += words[i];
      continue;
    }
    // Partials list characters, each with any mark that follows it, where
    // the punctuation settings segment text by character; elsewhere (th-TH,
    // which has none) words.
    if (!character_partials_) {
      if (!e->spaced_text.empty()) e->spaced_text += ' ';
      e->spaced_text += words[i];
      continue;
    }
    for (size_t j = 0; j < words[i].size();) {
      size_t n = 1;
      while (j + n < words[i].size() && (static_cast<unsigned char>(words[i][j + n]) & 0xC0) == 0x80) ++n;
      const std::string ch = words[i].substr(j, n);
      const bool mark = ch == "\u3001" || ch == "\u3002" || ch == "\uff1f" || ch == "\uff01" ||
                        ch == "," || ch == "." || ch == "?" || ch == "!" ||
                        std::find(punctuation_marks_.begin(), punctuation_marks_.end(), ch) !=
                            punctuation_marks_.end();
      if (!e->spaced_text.empty() && !mark) e->spaced_text += ' ';
      e->spaced_text += ch;
      j += n;
    }
  }
}

void Recognizer::Denormalize(bool final, const std::vector<std::unique_ptr<TextNormalizer>>& normalizers,
                             std::vector<WordTiming>* words) const {
  std::vector<const TextNormalizer*> grammars;
  for (const auto& d : normalizers) {
    if (const auto* p = dynamic_cast<const PqrnnSpokenPunctuationNormalizer*>(d.get())) {
      DenormalizeText(final, grammars, words);
      grammars.clear();
      PunctuateSpoken(*p, words);
    } else {
      grammars.push_back(d.get());
    }
  }
  DenormalizeText(final, grammars, words);
}

void Recognizer::PunctuateSpoken(const PqrnnSpokenPunctuationNormalizer& normalizer,
                                 std::vector<WordTiming>* words) const {
  std::vector<PqrnnSpokenPunctuationNormalizer::Word> timed;
  for (size_t i = 0; i < words->size(); ++i) {
    const int64_t start = (*words)[i].start_ms;
    const int64_t end = i + 1 < words->size() ? (*words)[i + 1].start_ms : decoded_end_ms_;
    timed.push_back({(*words)[i].raw_text, start, std::max(start, end)});
  }
  if (!normalizer.ApplyWords(&timed)) return;
  // libsoda's results list each part of a word between spaces as a word of
  // its own ("liste :" in fr-FR), except a part of marks the punctuation
  // model writes, which stays with the part before ("soir ?"). A part starts
  // as far into the word's frames as it is into its text, in bytes, rounded
  // to a frame ("dit « bonjour" over frames 9-28: "«" at 14, "bonjour" at
  // 19). Line breaks are dropped; the word after one is capitalized already.
  auto attached = [this](const std::string& part) {
    if (!punctuator_ || part.empty()) return false;
    for (size_t i = 0; i < part.size();) {
      bool found = false;
      for (const auto& m : punctuator_->supported_marks()) {
        if (!m.empty() && part.compare(i, m.size(), m) == 0) {
          i += m.size();
          found = true;
          break;
        }
      }
      if (!found) return false;
    }
    return true;
  };
  const double frame = static_cast<double>(frame_ms_);
  std::vector<WordTiming> out;
  for (const auto& w : timed) {
    bool first = true;
    for (size_t i = 0, j; i < w.text.size(); i = j + 1) {
      j = std::min(w.text.find_first_of(" \n", i), w.text.size());
      if (j == i) continue;
      const std::string part = w.text.substr(i, j - i);
      if (!first && attached(part)) {
        out.back().raw_text += " " + part;
        continue;
      }
      first = false;
      const double span = (w.end_ms - w.start_ms) / frame;
      const double at = w.start_ms / frame + static_cast<double>(i) * span / static_cast<double>(w.text.size());
      out.push_back({});
      out.back().raw_text = part;
      out.back().start_ms = static_cast<int64_t>(std::floor(at + 0.5) * frame);
    }
  }
  *words = std::move(out);
}

void Recognizer::DenormalizeText(bool final, const std::vector<const TextNormalizer*>& normalizers,
                                 std::vector<WordTiming>* words) const {
  if (normalizers.empty() || words->empty()) return;
  std::string text;
  for (const auto& w : *words) text += (text.empty() ? "" : " ") + w.raw_text;
  const std::string before = text;
  for (const auto* d : normalizers) text = d->Apply(text);
  if (text == before) return;
  // libsoda drops a paragraph or line break ("{new-paragraph}" in
  // dictation) and capitalizes the next word. A final drops the spaces
  // around it too, joining the words; a partial keeps one.
  for (size_t i = text.find('\n'); i != std::string::npos; i = text.find('\n', i)) {
    size_t a = i, b = i;
    while (a > 0 && (text[a - 1] == ' ' || text[a - 1] == '\n')) --a;
    while (b < text.size() && (text[b] == ' ' || text[b] == '\n')) ++b;
    if (b < text.size() && text[b] >= 'a' && text[b] <= 'z') text[b] = static_cast<char>(text[b] - 'a' + 'A');
    const bool space = !final && a > 0 && b < text.size();
    text.replace(a, b - a, space ? " " : "");
    i = a + space;
  }
  std::vector<std::string> tokens;
  for (size_t i = 0, j; i <= text.size(); i = j + 1) {
    j = std::min(text.find(' ', i), text.size());
    if (j > i) tokens.push_back(text.substr(i, j - i));
  }
  // Word starts carry over through the longest common subsequence of the
  // words. Between kept words, as many new words as replaced ones take
  // theirs one for one (hi-IN's transliteration of its romanized words);
  // otherwise a new word starts where the first word it replaced did.
  const size_t n = words->size(), m = tokens.size();
  std::vector<std::vector<int>> lcs(n + 1, std::vector<int>(m + 1, 0));
  for (size_t i = n; i-- > 0;) {
    for (size_t j = m; j-- > 0;) {
      lcs[i][j] = (*words)[i].raw_text == tokens[j] ? lcs[i + 1][j + 1] + 1
                                                     : std::max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  std::vector<int64_t> starts(m);
  std::vector<bool> joined(m);  // Only the first word replacing several stays joined.
  size_t gap_word = 0, gap_token = 0;  // Where the current replaced stretch begins.
  auto fill_gap = [&](size_t word_end, size_t token_end) {
    const bool one_for_one = word_end - gap_word == token_end - gap_token;
    for (size_t j = gap_token; j < token_end; ++j) {
      const size_t i = one_for_one ? gap_word + (j - gap_token) : gap_word;
      starts[j] = (*words)[std::min(i, n - 1)].start_ms;
      joined[j] = (one_for_one || j == gap_token) && (*words)[std::min(i, n - 1)].joined;
    }
  };
  for (size_t i = 0, j = 0; j < m || i < n;) {
    if (i < n && j < m && (*words)[i].raw_text == tokens[j]) {
      fill_gap(i, j);
      joined[j] = (*words)[i].joined;
      starts[j++] = (*words)[i++].start_ms;
      gap_word = i;
      gap_token = j;
    } else if (i < n && (j == m || lcs[i + 1][j] >= lcs[i][j + 1])) {
      ++i;  // A replaced word.
    } else {
      ++j;  // A new word.
    }
  }
  fill_gap(n, m);
  // As libsoda formats them, closing marks join the word before and opening
  // brackets the word after.
  auto only = [](const std::string& t, const char* set) {
    return !t.empty() && t.find_first_not_of(set) == std::string::npos;
  };
  std::vector<WordTiming> out;
  std::string prefix;
  for (size_t j = 0; j < m; ++j) {
    const std::string& t = tokens[j];
    if (only(t, "([{")) {
      prefix += t;
      continue;
    }
    if (!out.empty() && only(t, ",.?!;:%)]}")) {
      out.back().raw_text += prefix + t;
    } else {
      out.push_back({});
      out.back().raw_text = prefix + t;
      out.back().start_ms = starts[j];
      out.back().joined = joined[j];
    }
    prefix.clear();
  }
  if (!prefix.empty()) {
    if (out.empty()) out.push_back({prefix, prefix, words->front().start_ms});
    else out.back().raw_text += prefix;
  }
  *words = std::move(out);
}

// libsoda sends no result whose text is only line breaks (a spoken
// "new paragraph" on its own).
static bool Blank(const std::string& text) {
  return text.find_first_not_of(" \n") == std::string::npos;
}

void Recognizer::EmitPartial() {
  if (!callback_) return;
  RecognitionEvent e;
  e.type = RecognitionEvent::kPartial;
  Describe(search_->Best(), /*final=*/false, &e);
  if (Blank(e.raw_text) || e.spaced_text == last_partial_) return;
  last_partial_ = e.spaced_text;
  e.leading_space = spaced_words_ && had_final_;
  SetWordConfidence(&e);
  e.start_ms = segment_start_ms_;
  // libsoda's partials end a frame before ours.
  e.end_ms = std::max<int64_t>(segment_start_ms_, decoded_end_ms_ - static_cast<int64_t>(frame_ms_));
  callback_(e);
}

void Recognizer::SetWordConfidence(RecognitionEvent* e) const {
  // As observed from libsoda (en-US, ja-JP, cmn-*): a word that a space
  // precedes in the raw text is LOW, as from a confidence of 0 (these models
  // compute none), counting the space before each result after the
  // recognizer's first in packs that write words with spaces; other words
  // stay UNKNOWN (ja-JP: `テスト@ Gmailcomです` makes only Gmail LOW). In a
  // pack without punctuation (th-TH) every word stays UNKNOWN.
  if (!punctuator_) return;
  size_t pos = 0;
  for (WordTiming& w : e->words) {
    const size_t at = e->raw_text.find(w.raw_text, pos);
    if (at == std::string::npos) break;
    w.confidence = (at > 0 ? e->raw_text[at - 1] == ' ' : e->leading_space) ? 1 : 0;
    pos = at + w.raw_text.size();
  }
}

void Recognizer::EmitFinal(bool end_of_audio, bool segment_end) {
  if (segment_end) {
    // The result ends at the last encoder frame boundary before the segment
    // end, where recognition restarts. Encoder frames start at the
    // segment's onset.
    const int64_t base_ms = segment_start_ms_;
    const int64_t frame = static_cast<int64_t>(frame_ms_);
    decoded_end_ms_ = base_ms + std::max<int64_t>(0, (decoded_end_ms_ - base_ms - 1) / frame * frame);
    restart_at_ = decoded_end_ms_ * kModelSampleRate / 1000;
  }
  RecognitionEvent e;
  e.type = RecognitionEvent::kFinal;
  if (cascade_) {
    // Finals come from the second pass.
    FlushCascade();
    Describe(search2_->Best(), /*final=*/true, frame_ends2_ms_, &e);
  } else {
    Describe(search_->Best(), /*final=*/true, &e);
  }
  if (callback_ && (!Blank(e.raw_text) || end_of_audio)) {
    e.end_of_audio = end_of_audio;
    // Like libsoda, a result starts at its first word (usually where the
    // previous one ended) and ends at its segment's end.
    e.start_ms = segment_start_ms_;
    e.end_ms = decoded_end_ms_;
    if (!e.words.empty()) e.start_ms = std::max(e.start_ms, e.words.front().start_ms);
    e.leading_space = spaced_words_ && had_final_;
    SetWordConfidence(&e);
    had_final_ |= !Blank(e.raw_text);
    callback_(e);
  }
  // A restart at a segment end starts the search afresh too (measured
  // against libsoda on the lesson recording: 19% vs 27% with the decoder
  // context kept).
  if (config_.decoder.resume_search && !end_of_audio && !segment_end) {
    search_->Resume();
  } else {
    search_->Reset();
  }
  ResetCascade();
  segment_start_ms_ = last_final_end_ms_ = decoded_end_ms_;
  frame_ends_ms_.clear();
  last_partial_.clear();
}

}  // namespace soda
