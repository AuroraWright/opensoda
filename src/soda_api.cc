// libsoda-compatible C API on top of soda::Recognizer.

#include "soda_api.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "proto_wire.h"
#include "recognizer.h"

namespace {

// speech.soda.chrome.SodaResponse.SodaMessageType
enum SodaMessageType {
  kRecognition = 1, kStop = 2, kShutdown = 3, kStart = 4, kEndpoint = 5, kAudioLevel = 6, kLangId = 7
};
// SodaRecognitionResult.ResultType
enum ResultType { kPartial = 1, kFinal = 2 };
// SodaRecognitionResult.FinalResultEndpointReason
enum EndpointReason { kEndpointUnknown = 0 };
// SodaEndpointEvent.EndpointType
enum EndpointType { kStartOfSpeechEvent = 0, kEndOfSpeechEvent = 1, kEndOfAudioEvent = 2 };
// SodaLangIdEvent.AsrSwitchResult
enum AsrSwitchResult { kDefaultNoSwitch = 0, kSwitchSucceeded = 1, kSwitchFailed = 2, kSwitchSkippedNoLp = 3 };
// ExtendedSodaConfigMsg.RecognitionMode
enum RecognitionMode { kIme = 1, kCaption = 2 };

// libsoda's table of locales the language identifier reports as another
// (es-ES as es-us, cmn-Hant-TW as cmn-hans-cn): an identified language
// stands for its own locale and these ("alternate top locales").
constexpr std::pair<const char*, const char*> kLangIdLocales[] = {
    {"ar-x-gulf", "ar-eg"},       {"cmn-hant-tw", "cmn-hans-cn"}, {"zh-cn", "cmn-hans-cn"},
    {"zh-hant-tw", "cmn-hans-cn"}, {"zh-tw", "cmn-hans-cn"},       {"de-at", "de-de"},
    {"de-be", "de-de"},           {"de-ch", "de-de"},             {"en-au", "en-us"},
    {"en-ca", "en-us"},           {"en-gb", "en-us"},             {"en-ie", "en-us"},
    {"en-in", "en-us"},           {"en-ph", "en-us"},             {"en-sg", "en-us"},
    {"es-es", "es-us"},           {"fr-be", "fr-fr"},             {"fr-ca", "fr-fr"},
    {"fr-ch", "fr-fr"},           {"it-ch", "it-it"},
};

// The locales an identified language stands for, its own first.
std::vector<std::string> LangIdLocales(const std::string& language) {
  std::vector<std::string> locales = {language};
  for (const auto& [locale, reported] : kLangIdLocales) {
    if (language == reported) locales.push_back(locale);
  }
  return locales;
}

int64_t NowUsec() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

class SodaSession {
 public:
  SodaSession(SerializedSodaEventHandler cb, void* handle) : cb_(cb), handle_(handle) {}
  ~SodaSession() {
    {
      std::lock_guard<std::mutex> l(mu_);
      quit_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    SendType(kShutdown);
  }

  bool Init(const std::string& config) {
    soda::RecognizerOptions o;
    int mode = kIme;
    bool simulate_realtime = false;
    bool speaker_change = false;
    soda::ProtoReader r(config);
    while (r.Next()) {
      switch (r.field()) {
        case 1: o.channel_count = static_cast<int>(r.varint()); break;
        case 2: o.sample_rate = static_cast<int>(r.varint()); break;
        case 5: simulate_realtime = r.varint() != 0; break;  // simulate_realtime_testonly
        case 7: o.language_pack_dir = r.str(); break;
        case 8: mode = static_cast<int>(r.varint()); break;
        case 11: o.enable_language_id |= r.varint() != 0; break;  // enable_lang_id
        case 12: o.enable_punctuation = r.varint() != 0; break;  // enable_formatting
        // multilang_config: Chrome sets it for Live Caption, and libsoda then
        // reports language identification even with enable_lang_id off. Its
        // field 1 maps each installed language to its pack.
        case 15: {
          o.enable_language_id = multilang_ = true;
          soda::ProtoReader m(r.bytes());
          while (m.Next()) {
            if (m.field() != 1) continue;
            soda::ProtoReader entry(m.bytes());
            std::string locale, dir;
            while (entry.Next()) {
              if (entry.field() == 1) locale = entry.str();
              if (entry.field() == 2) dir = entry.str();
            }
            for (char& ch : locale) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (!locale.empty() && !dir.empty()) packs_[locale] = dir;
          }
          break;
        }
        case 13: speaker_change |= r.varint() != 0; break;  // enable_speaker_change_detection
        case 16: o.mask_offensive_words = r.varint() != 0; break;
        // speaker_diarization_mode: SPEAKER_CHANGE_DETECTION (2) is the same.
        case 17: speaker_change |= r.varint() == 2; break;
        case 19: o.phrases = ParseContext(r.bytes()); break;  // recognition_context
      }
    }
    if (o.language_pack_dir.empty()) return false;
    // libsoda refuses speaker change detection with multilang languages
    // (unless limited to one switch per session, which Chrome's
    // MultilangConfig cannot ask for) and recognizes without it.
    o.speaker_change_detection = speaker_change && packs_.empty();
    // Without multilang languages, libsoda's text formatter holds back each
    // final's unfinished sentence.
    o.hold_unfinished_sentences = packs_.empty();
    // libsoda runs the pack's CONTINUOUS config in both modes (its log names
    // the config it loads), which RecognizerOptions picks by default; the
    // mode only selects the recognition domain.
    o.domain = mode == kCaption ? "CAPTION" : "IME";
    // Fed faster than it recognizes, libsoda skips audio to keep up (see
    // max_backlog_ms), keeping more with multilang_config, where a language
    // switch rewinds (max_restream_duration_for_multilang_ms). With
    // simulate_realtime_testonly it reads the audio at real-time speed
    // instead, never falling behind; opensoda then recognizes all of it as
    // fast as it can.
    o.max_backlog_ms = simulate_realtime ? 0 : multilang_ ? 5000 : 1000;
    std::string error;
    recognizer_ = soda::Recognizer::Create(o, &error);
    if (!recognizer_) return false;
    options_ = o;
    channels_ = std::max(1, o.channel_count);
    Wire(recognizer_.get());
    recognizer_->set_language_callback([this](const soda::LanguageEvent& e) { SendLanguage(e); });
    language_ = PackLanguage(o.language_pack_dir);
    worker_ = std::thread([this] { Run(); });
    return true;
  }

  void AddAudio(const char* data, int size) {
    if (size <= 0) return;
    Command c{Command::kAudio, {}};
    c.audio.resize(size / 2);
    std::memcpy(c.audio.data(), data, c.audio.size() * 2);
    queued_frames_ += static_cast<int64_t>(c.audio.size()) / channels_;
    Post(std::move(c));
  }
  void Start() { Post({Command::kStart, {}}); }
  void Stop() { Post({Command::kStop, {}}); }
  void MarkDone() { Post({Command::kMarkDone, {}}); }
  void UpdateContext(std::string_view context) {
    Command c{Command::kContext, {}};
    c.phrases = ParseContext(context);
    Post(std::move(c));
  }

 private:
  struct Command {
    enum Type { kAudio, kStart, kStop, kMarkDone, kContext } type;
    std::vector<int16_t> audio;
    std::vector<soda::BiasPhrase> phrases;
  };

  // RecognitionContext {1: {1: context name, 2: {1: {1: phrase, 2: boost}}}},
  // as Chrome's Web Speech API sends it.
  static std::vector<soda::BiasPhrase> ParseContext(std::string_view context) {
    std::vector<soda::BiasPhrase> phrases;
    soda::ProtoReader r(context);
    while (r.Next()) {
      if (r.field() != 1 || r.wire_type() != soda::kLengthDelimited) continue;
      soda::ProtoReader input(r.bytes());
      std::string name;
      while (input.Next()) {
        if (input.field() == 1) name = input.str();
        if (input.field() != 2 || input.wire_type() != soda::kLengthDelimited) continue;
        soda::ProtoReader list(input.bytes());
        while (list.Next()) {
          if (list.field() != 1 || list.wire_type() != soda::kLengthDelimited) continue;
          soda::ProtoReader phrase(list.bytes());
          soda::BiasPhrase p;
          p.context = name;
          while (phrase.Next()) {
            if (phrase.field() == 1) p.text = phrase.str();
            if (phrase.field() == 2) p.boost = phrase.fixed_float();
          }
          if (!p.text.empty()) phrases.push_back(std::move(p));
        }
      }
    }
    return phrases;
  }

  void Post(Command c) {
    {
      std::lock_guard<std::mutex> l(mu_);
      queue_.push_back(std::move(c));
    }
    cv_.notify_one();
  }

  void Run() {
    while (true) {
      Command c;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return quit_ || !queue_.empty(); });
        if (quit_ && queue_.empty()) return;
        c = std::move(queue_.front());
        queue_.pop_front();
      }
      switch (c.type) {
        case Command::kAudio:
          queued_frames_ -= static_cast<int64_t>(c.audio.size()) / channels_;
          if (audio_start_usec_ == 0) audio_start_usec_ = NowUsec();
          recognizer_->AddAudio(c.audio.data(), c.audio.size());
          FinishSwitch();
          break;
        case Command::kStart:
          recognizer_->Reset();
          audio_start_usec_ = 0;
          SendType(kStart);
          break;
        case Command::kStop:
          recognizer_->Finish();
          SendType(kStop);
          break;
        case Command::kMarkDone:
          // Chrome treats STOP as the end-of-session signal.
          recognizer_->Finish();
          SendType(kStop);
          break;
        case Command::kContext:
          options_.phrases = c.phrases;
          recognizer_->SetPhrases(c.phrases);
          break;
      }
    }
  }

  void Wire(soda::Recognizer* r) {
    r->set_callback([this](const soda::RecognitionEvent& e) { SendRecognition(e); });
    r->set_queued_audio_callback([this] { return queued_frames_.load(); });
    r->set_audio_level_callback([this](const soda::AudioLevelEvent& e) { SendAudioLevel(e); });
    r->set_endpoint_callback([this](const soda::EndpointEvent& e) {
      SendEndpoint(e.type == soda::EndpointEvent::kStartOfSpeech ? kStartOfSpeechEvent
                   : e.type == soda::EndpointEvent::kEndOfSpeech ? kEndOfSpeechEvent
                                                                 : kEndOfAudioEvent,
                   e.audio_ms);
    });
  }

  // Loads the pack for a newly identified language (or for a locale it
  // stands for); the switch happens once the current audio chunk is
  // processed.
  int PrepareSwitch(const std::string& language) {
    auto it = packs_.end();
    for (const std::string& locale : LangIdLocales(language)) {
      it = packs_.find(locale);
      if (it != packs_.end()) break;
    }
    if (it == packs_.end()) return kSwitchSkippedNoLp;
    soda::RecognizerOptions o = options_;
    o.language_pack_dir = it->second;
    o.enable_language_id = false;  // The current recognizer's carries over.
    std::string error;
    auto next = soda::Recognizer::Create(o, &error);
    if (!next) return kSwitchFailed;
    next_recognizer_ = std::move(next);
    next_language_ = it->first;
    return kSwitchSucceeded;
  }

  void FinishSwitch() {
    if (!next_recognizer_) return;
    Wire(next_recognizer_.get());
    next_recognizer_->ContinueFrom(recognizer_.get());
    recognizer_ = std::move(next_recognizer_);
    language_ = next_language_;
  }

  void SendType(int type) {
    soda::ProtoWriter w;
    w.Varint(1, type);
    Send(w);
  }

  void SendEndpoint(int type, int64_t audio_ms) {
    soda::ProtoWriter timing;  // TimingMetrics, laid out as libsoda fills it.
    const int64_t elapsed = NowUsec() - audio_start_usec_;
    timing.Varint(1, elapsed);
    timing.Varint(2, audio_ms * 1000);
    timing.Varint(3, elapsed);
    timing.Varint(4, audio_ms * 1000);
    soda::ProtoWriter event;  // SodaEndpointEvent
    event.Varint(1, type);
    event.Message(2, timing);
    soda::ProtoWriter response;
    response.Varint(1, kEndpoint);
    response.Message(3, event);
    Send(response);
  }

  void SendAudioLevel(const soda::AudioLevelEvent& e) {
    soda::ProtoWriter info;  // AudioLevelInfo
    if (e.has_rms) info.Float(1, e.rms);
    if (e.has_speech_level) {
      info.Float(2, e.speech_level);
      info.Varint(3, e.speech_level_ms * 1000);
    }
    soda::ProtoWriter response;
    response.Varint(1, kAudioLevel);
    response.Message(4, info);
    Send(response);
  }

  // The pack's locale, lowercase ("ja-jp"), from .../SODALanguagePacks/<locale>/.
  static std::string PackLanguage(const std::string& dir) {
    const std::string marker = "SODALanguagePacks/";
    const size_t at = dir.find(marker);
    if (at == std::string::npos) return "";
    std::string locale = dir.substr(at + marker.size());
    locale = locale.substr(0, locale.find('/'));
    for (char& c : locale) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return locale;
  }

  void SendLanguage(const soda::LanguageEvent& e) {
    soda::ProtoWriter event;  // SodaLangIdEvent
    event.Bytes(1, e.language);
    event.Varint(2, e.confidence_level);
    // With multilang on, libsoda switches to the pack of a highly confident
    // other language, or reports that it has none. The new pack recognizes
    // again from the last final. A language that stands for the current
    // locale is not another (es-us in es-ES).
    const std::vector<std::string> locales = LangIdLocales(e.language);
    const bool other = multilang_ && !language_.empty() &&
                       std::find(locales.begin(), locales.end(), language_) == locales.end() &&
                       e.confidence_level == 3 && !e.superseded && !next_recognizer_;
    if (other) event.Varint(3, PrepareSwitch(e.language));
    soda::ProtoWriter response;
    response.Varint(1, kLangId);
    response.Message(5, event);
    Send(response);
  }

  void SendRecognition(const soda::RecognitionEvent& e) {
    // Mirrors what Chrome receives from libsoda in caption mode: hypothesis 0
    // is the formatted text (words separated by spaces; in partials of
    // languages written without spaces, characters), hypothesis 1 the raw
    // text; each part carries both word forms.
    // libsoda sends no empty result; the stream ends with an end of audio
    // endpoint event instead.
    if (e.raw_text.empty()) return;
    soda::ProtoWriter result;  // SodaRecognitionResult
    const std::string space = e.leading_space ? " " : "";
    result.Bytes(1, space + e.spaced_text);
    result.Bytes(1, space + e.raw_text);
    const bool final = e.type == soda::RecognitionEvent::kFinal;
    result.Varint(2, final ? kFinal : kPartial);
    // libsoda reports every final as ENDPOINT_UNKNOWN, even one that end of
    // audio cut short, in caption and IME mode alike.
    if (final) result.Varint(3, kEndpointUnknown);
    soda::ProtoWriter timing;  // TimingMetrics; libsoda sets only these two.
    timing.Varint(2, e.start_ms * 1000);
    timing.Varint(4, e.end_ms * 1000);
    result.Message(4, timing);
    auto add_part = [&](const std::string& text, const std::string& raw, int64_t offset_ms,
                        int confidence, bool speaker_change = false) {
      soda::ProtoWriter part;  // HypothesisPart
      part.Bytes(1, text);
      part.Bytes(1, raw);
      part.Varint(2, offset_ms);
      if (speaker_change) part.Varint(3, 1);  // speaker_change
      part.Varint(5, confidence);  // word_confidence, written even when 0
      result.Message(5, part);
    };
    if (final) {
      for (const auto& word : e.words) {
        add_part(word.text, word.raw_text, word.start_ms - e.start_ms, word.confidence,
                 word.speaker_change);
      }
    } else if (e.spaced_words) {
      // In languages written with spaces, one part per word, at offset 0.
      for (const auto& word : e.words) {
        add_part(word.text, word.raw_text, 0, word.confidence, word.speaker_change);
      }
    } else {
      // Otherwise partials list one part per character, all at offset 0.
      for (size_t i = 0; i < e.raw_text.size();) {
        size_t n = 1;
        while (i + n < e.raw_text.size() && (static_cast<unsigned char>(e.raw_text[i + n]) & 0xC0) == 0x80) ++n;
        const std::string ch = e.raw_text.substr(i, n);
        i += n;
        if (ch != " ") add_part(ch, ch, 0, 0);
      }
    }
    soda::ProtoWriter response;
    response.Varint(1, kRecognition);
    response.Message(2, result);
    Send(response);
  }

  void Send(const soda::ProtoWriter& w) {
    if (cb_) cb_(w.data().data(), static_cast<int>(w.data().size()), handle_);
  }

  SerializedSodaEventHandler cb_;
  void* handle_;
  std::unique_ptr<soda::Recognizer> recognizer_;
  int channels_ = 1;
  bool multilang_ = false;
  std::map<std::string, std::string> packs_;  // multilang_config's packs by locale.
  soda::RecognizerOptions options_;
  std::string language_;  // The current pack's locale, lowercase.
  std::unique_ptr<soda::Recognizer> next_recognizer_;  // A pending switch.
  std::string next_language_;
  int64_t audio_start_usec_ = 0;
  std::atomic<int64_t> queued_frames_{0};  // Audio posted but not yet recognized.
  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Command> queue_;
  bool quit_ = false;
};

}  // namespace

extern "C" {

void* CreateExtendedSodaAsync(SerializedSodaConfig config) {
  if (!config.soda_config || config.soda_config_size <= 0) return nullptr;
  auto session = std::make_unique<SodaSession>(config.callback, config.callback_handle);
  if (!session->Init(std::string(config.soda_config, config.soda_config_size))) {
    return nullptr;
  }
  return session.release();
}

void DeleteExtendedSodaAsync(void* handle) { delete static_cast<SodaSession*>(handle); }

void ExtendedAddAudio(void* handle, const char* audio, int size) {
  if (handle) static_cast<SodaSession*>(handle)->AddAudio(audio, size);
}

void ExtendedSodaStart(void* handle) {
  if (handle) static_cast<SodaSession*>(handle)->Start();
}

void ExtendedSodaStop(void* handle) {
  if (handle) static_cast<SodaSession*>(handle)->Stop();
}

void ExtendedSodaMarkDone(void* handle) {
  if (handle) static_cast<SodaSession*>(handle)->MarkDone();
}

void UpdateRecognitionContext(void* handle, RecognitionContext context) {
  if (!handle || !context.recognition_context || context.recognition_context_size <= 0) return;
  static_cast<SodaSession*>(handle)->UpdateContext(
      std::string_view(context.recognition_context, context.recognition_context_size));
}

}  // extern "C"
