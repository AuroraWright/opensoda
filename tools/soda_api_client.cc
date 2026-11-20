// Drives any libsoda-compatible library through its C API.
//
// Usage: soda_api_client LIBRARY PACK_DIR file.wav... [--clips LIST] [--jsonl F]
//                        [--dump-dir D] [--ime] [--realtime]
//                        [--speed X] [--simulate-realtime] [--chunk-ms N] [--multilang LOCALE=DIR]...
//                        [--no-format] [--mask-offensive] [--api-key KEY] [--phrase TEXT[:BOOST]]...
//                        [--config-file F] [--update-context] [--context-name NAME]
//                        [--no-multilang] [--speaker-change]
//
// --realtime paces the audio like a live microphone; --speed X paces it X
// times faster. Unpaced, the libraries skip audio to keep up as with a live
// microphone: when recognition starts after a final with more than 1 s (5 s
// with the multilang config Chrome sends) of audio waiting, all but that
// much is dropped. --simulate-realtime sets the config's
// simulate_realtime_testonly: Google's libsoda then reads the audio at
// real-time speed itself, and opensoda recognizes all of it as fast as it
// can. --chunk-ms sets how much audio each AddAudio call carries.
//
// Several clips (as arguments, or listed one per line in --clips LIST, where
// anything after a tab is ignored) are recognized the way Chrome captions
// one media stream after another: the library is loaded once, and each clip
// gets its own session, created with the clip's sample rate and channel
// count, started, fed, marked done and deleted once it has sent STOP.
// --jsonl F writes one JSON line per clip: its finals (formatted and raw text,
// start and end), their concatenations, the number of partials, and the
// session's wall-clock time. Results are printed only for a single clip
// unless --verbose.
//
// A clip that has not finished --clip-timeout S seconds (default 30) after
// all its audio was fed is recorded as an error, and the client exits with
// status 3, since a session stuck inside the library cannot be deleted
// (Google's libsoda can loop forever after ExtendedSodaMarkDone when its
// search ends a segment without a result). tests/corpus.py then goes on
// with the next clip in a new process.
//
// --dump-responses F writes every response to F as {f64 time, u64 handle,
// u32 size, bytes} (read by tests/soda_dump.py); with several clips,
// --dump-dir D writes D/<clip number>.bin for each (from 00000.bin).
//
// --no-multilang leaves out the multilang config Chrome sends (libsoda then
// holds back each final's unfinished sentence); --speaker-change sets
// enable_speaker_change_detection, which libsoda only honors without one.
//
// --update-context sends the --phrase list with UpdateRecognitionContext
// after ExtendedSodaStart instead of in the config. --context-name names the
// context the phrases go in (default "android-speech-api-generic-phrases").
//
// --config-file replays a serialized ExtendedSodaConfigMsg (e.g. one captured
// from Chrome), replacing only its channel count, sample rate and API key.
//
// LIBRARY may be Chrome's libsoda.so (as a reference) or this project's
// libsoda.dylib; both are loaded with dlopen and exercised identically.

#include <dlfcn.h>

#include <cstdlib>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "proto_wire.h"

namespace {

typedef void (*EventHandler)(const char*, int, void*);
struct SerializedSodaConfig {
  const char* soda_config;
  int soda_config_size;
  EventHandler callback;
  void* callback_handle;
};
typedef void* (*CreateFn)(SerializedSodaConfig);
typedef void (*HandleFn)(void*);
typedef void (*AddAudioFn)(void*, const char*, int);
typedef struct {
  const char* recognition_context;
  int recognition_context_size;
} RecognitionContext;
typedef void (*UpdateContextFn)(void*, RecognitionContext);

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Final {
  std::string text, raw;  // hypotheses 0 (formatted) and 1
  int64_t start_us = 0, end_us = 0;
};

// One session's state; its address is the callback handle.
struct Session {
  bool print = true;
  FILE* dump = nullptr;
  std::atomic<bool> done{false};
  std::atomic<int64_t> last_event_ms{0};
  std::mutex mu;  // Guards the fields below.
  std::vector<Final> finals;
  int partials = 0;
};

void OnEvent(const char* data, int size, void* handle) {
  Session* s = static_cast<Session*>(handle);
  s->last_event_ms = NowMs();
  if (s->dump) {
    const double now = NowMs() / 1000.0;
    const uint64_t dump_handle = 0;
    const uint32_t n = static_cast<uint32_t>(size);
    std::fwrite(&now, sizeof(now), 1, s->dump);
    std::fwrite(&dump_handle, sizeof(dump_handle), 1, s->dump);
    std::fwrite(&n, sizeof(n), 1, s->dump);
    std::fwrite(data, 1, n, s->dump);
  }
  soda::ProtoReader r(std::string_view(data, size));
  int type = 0;
  std::string_view result, langid;
  while (r.Next()) {
    if (r.field() == 1) type = static_cast<int>(r.varint());
    if (r.field() == 2) result = r.bytes();
    if (r.field() == 5) langid = r.bytes();
  }
  if (type == 7) {  // LANGID: language, confidence level, switch result.
    soda::ProtoReader e(langid);
    std::string language;
    int level = 0, switch_result = 0;
    while (e.Next()) {
      if (e.field() == 1) language = e.str();
      if (e.field() == 2) level = static_cast<int>(e.varint());
      if (e.field() == 3) switch_result = static_cast<int>(e.varint());
    }
    if (s->print) std::printf("<langid %s confidence=%d switch=%d>\n", language.c_str(), level, switch_result);
    return;
  }
  if (type != 1) {
    if (type != 6 && s->print) std::printf("<event type=%d>\n", type);
    // STOP follows the last result of audio marked done.
    if (type == 2) s->done = true;
    return;
  }
  Final f;
  int result_type = 0, reason = 0, hypotheses = 0;
  soda::ProtoReader rr(result);
  while (rr.Next()) {
    if (rr.field() == 1) (hypotheses++ == 0 ? f.text : f.raw) = rr.str();
    if (rr.field() == 2) result_type = static_cast<int>(rr.varint());
    if (rr.field() == 3) reason = static_cast<int>(rr.varint());
    if (rr.field() == 4) {
      soda::ProtoReader t(rr.bytes());
      while (t.Next()) {
        if (t.field() == 2) f.start_us = static_cast<int64_t>(t.varint());
        if (t.field() == 4) f.end_us = static_cast<int64_t>(t.varint());
      }
    }
  }
  // Like Chrome, ignore results without hypotheses.
  if (hypotheses == 0) return;
  if (result_type == 2) {
    if (s->print) std::printf("FINAL(reason=%d): %s\n", reason, f.text.c_str());
    std::lock_guard<std::mutex> lock(s->mu);
    s->finals.push_back(std::move(f));
  } else {
    if (s->print) std::printf("  partial: %s\n", f.text.c_str());
    std::lock_guard<std::mutex> lock(s->mu);
    ++s->partials;
  }
  if (s->print) std::fflush(stdout);
}

// Reads a 16-bit PCM WAV file (plain or WAVE_FORMAT_EXTENSIBLE).
bool ReadPcm16Wav(const std::string& path, std::vector<char>* pcm, int* rate, int* channels,
                  std::string* error) {
  std::ifstream f(path, std::ios::binary);
  char hdr[12];
  if (!f.read(hdr, 12) || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
    *error = "not a WAV file";
    return false;
  }
  bool have_format = false;
  while (f) {
    char id[4];
    uint32_t size;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&size), 4);
    if (!f) break;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      std::vector<char> fmt(std::max<uint32_t>(size, 16));
      f.read(fmt.data(), size);
      if (size & 1) f.seekg(1, std::ios::cur);
      uint16_t tag, bits;
      std::memcpy(&tag, &fmt[0], 2);
      std::memcpy(&bits, &fmt[14], 2);
      if ((tag != 1 && tag != 0xFFFE) || bits != 16) {
        *error = "not 16-bit PCM (format " + std::to_string(tag) + ", " + std::to_string(bits) + " bits)";
        return false;
      }
      uint16_t ch;
      uint32_t sr;
      std::memcpy(&ch, &fmt[2], 2);
      std::memcpy(&sr, &fmt[4], 4);
      *channels = ch;
      *rate = static_cast<int>(sr);
      have_format = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (!have_format) break;
      pcm->resize(size);
      f.read(pcm->data(), size);
      pcm->resize(static_cast<size_t>(f.gcount()) / (2 * *channels) * (2 * *channels));
      return true;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);
    }
  }
  *error = have_format ? "no data chunk" : "no fmt chunk";
  return false;
}

void AppendJsonString(std::string* out, const std::string& s) {
  out->push_back('"');
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      out->push_back('\\');
      out->push_back(static_cast<char>(c));
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", c);
      out->append(b);
    } else {
      out->push_back(static_cast<char>(c));
    }
  }
  out->push_back('"');
}

struct Options {
  std::string pack;
  bool ime = false, format = true, mask = false, update_context = false, verbose = false;
  bool simulate_realtime = false;
  bool multilang = true, speaker_change = false;
  double speed = 0;  // 0: as fast as possible.
  double clip_timeout_s = 30;  // After the audio is fed; 0: none.
  int chunk_ms = 100;
  std::string api_key = "dummy_api_key", config_file, saved_config;
  std::string context_name = "android-speech-api-generic-phrases";
  std::vector<std::pair<std::string, float>> phrases;
  std::vector<std::pair<std::string, std::string>> other_packs;  // --multilang LOCALE=DIR
};

// The serialized ExtendedSodaConfigMsg for a clip, as Chrome builds it.
std::string BuildConfig(const Options& o, int rate, int channels, std::string* context_msg) {
  soda::ProtoWriter cfg;  // speech.soda.chrome.ExtendedSodaConfigMsg
  cfg.Varint(1, channels);
  cfg.Varint(2, rate);
  cfg.Bytes(6, o.api_key);
  if (o.simulate_realtime) cfg.Bool(5, true);  // simulate_realtime_testonly
  if (!o.saved_config.empty()) {
    soda::ProtoReader r(o.saved_config);  // All fields but 1, 2 and 6 (and 5, set above).
    while (r.Next()) {
      const uint32_t f = r.field();
      if (f == 1 || f == 2 || f == 6 || (f == 5 && o.simulate_realtime)) continue;
      if (r.wire_type() == 0) {
        cfg.Varint(f, r.varint());
      } else {
        cfg.Bytes(f, r.bytes());
      }
    }
  } else {
    cfg.Bytes(7, o.pack);
    cfg.Varint(8, o.ime ? 1 : 2);
    cfg.Bool(12, o.format);  // enable_formatting
    if (o.mask) cfg.Bool(16, true);  // mask_offensive_words
    if (o.speaker_change) cfg.Bool(13, true);  // enable_speaker_change_detection
    // Chrome also sends a multi-language config mapping the language to its
    // pack ({1: map<language, pack dir>, 2: 1}), listing every installed pack;
    // --multilang adds more.
    std::string language = "ja-jp";
    const size_t slash = o.pack.find("/SODALanguagePacks/");
    if (slash != std::string::npos) {
      const size_t begin = slash + std::strlen("/SODALanguagePacks/");
      language = o.pack.substr(begin, o.pack.find('/', begin) - begin);
      for (char& c : language) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    std::vector<std::pair<std::string, std::string>> packs = o.other_packs;
    packs.insert(packs.begin(), {language, o.pack});
    soda::ProtoWriter multi;
    for (const auto& [locale, dir] : packs) {
      soda::ProtoWriter entry;
      entry.Bytes(1, locale);
      entry.Bytes(2, dir);
      multi.Bytes(1, entry.data());
    }
    multi.Varint(2, 1);  // rewind_when_switching_language
    if (o.multilang) cfg.Bytes(15, multi.data());
  }
  if (!o.phrases.empty()) {
    // recognition_context (19), as Chrome's Web Speech API sends it.
    soda::ProtoWriter list;
    for (const auto& [text, boost] : o.phrases) {
      soda::ProtoWriter phrase;
      phrase.Bytes(1, text);
      phrase.Float(2, boost);
      list.Message(1, phrase);
    }
    soda::ProtoWriter input;
    input.Bytes(1, o.context_name);
    input.Message(2, list);
    soda::ProtoWriter context;
    context.Message(1, input);
    if (o.update_context) {
      *context_msg = context.data();
    } else {
      cfg.Message(19, context);
    }
  }
  return cfg.data();
}

struct Library {
  void* handle = nullptr;
  CreateFn create = nullptr;
  HandleFn destroy = nullptr, start = nullptr, mark_done = nullptr;
  AddAudioFn add_audio = nullptr;
  UpdateContextFn update_context = nullptr;
};

// Recognizes one clip in its own session. Returns false if it could not run.
bool RunClip(const Library& lib, const Options& o, const std::vector<char>& pcm, int rate,
             int channels, Session* s, std::string* error) {
  std::string context_msg;
  const std::string cfg = BuildConfig(o, rate, channels, &context_msg);
  SerializedSodaConfig config{cfg.data(), static_cast<int>(cfg.size()), OnEvent, s};
  void* soda = lib.create(config);
  if (!soda) {
    *error = "CreateExtendedSodaAsync failed";
    return false;
  }
  lib.start(soda);
  if (!context_msg.empty() && lib.update_context) {
    lib.update_context(soda, {context_msg.data(), static_cast<int>(context_msg.size())});
  }
  // Chunks of --chunk-ms (default 100 ms), optionally paced like a live
  // microphone (on a fixed schedule, so sleeping late does not add up).
  const size_t chunk = static_cast<size_t>(std::max(1, rate * o.chunk_ms / 1000)) * channels * 2;
  const auto begin = std::chrono::steady_clock::now();
  const double bytes_per_us = rate * channels * 2 / 1e6;
  for (size_t off = 0; off < pcm.size(); off += chunk) {
    const size_t n = std::min(chunk, pcm.size() - off);
    lib.add_audio(soda, pcm.data() + off, static_cast<int>(n));
    if (o.speed > 0) {
      std::this_thread::sleep_until(
          begin + std::chrono::microseconds(static_cast<int64_t>((off + n) / bytes_per_us / o.speed)));
    }
  }
  lib.mark_done(soda);
  // Wait for STOP, or for the event stream to go quiet. (libsoda can send
  // the end-of-audio endpoint after the end-of-audio final.)
  s->last_event_ms = NowMs();
  while (!s->done && NowMs() - s->last_event_ms < 3000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  lib.destroy(soda);
  return true;
}

// Calls on_timeout and exits the process (status 3) unless destroyed within
// `seconds`.
class Watchdog {
 public:
  Watchdog(double seconds, std::function<void()> on_timeout) {
    if (seconds <= 0) return;
    thread_ = std::thread([this, seconds, on_timeout = std::move(on_timeout)] {
      std::unique_lock<std::mutex> lock(mu_);
      if (!cv_.wait_for(lock, std::chrono::duration<double>(seconds), [this] { return done_; })) {
        on_timeout();
        std::fflush(nullptr);
        std::_Exit(3);
      }
    });
  }
  ~Watchdog() {
    if (!thread_.joinable()) return;
    {
      std::lock_guard<std::mutex> lock(mu_);
      done_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  std::thread thread_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s LIBRARY PACK_DIR file.wav... [--clips LIST] [--jsonl F] [--dump-dir D] [--verbose] [--ime] [--realtime] [--speed X] [--simulate-realtime] [--chunk-ms N] [--no-format] [--mask-offensive] [--api-key KEY] [--phrase TEXT[:BOOST]]... [--update-context] [--context-name NAME] [--multilang LOCALE=DIR]... [--no-multilang] [--speaker-change] [--dump-responses F] [--config-file F]\n",
                 argv[0]);
    return 2;
  }
  Options o;
  o.pack = argv[2];
  std::vector<std::string> clips;
  std::string jsonl_path, dump_dir, dump_path;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a.rfind("--", 0) != 0) {
      clips.push_back(a);
    } else if (a == "--multilang" && has_value) {
      const std::string v = argv[++i];
      const size_t eq = v.find('=');
      if (eq != std::string::npos) o.other_packs.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else if (a == "--ime") {
      o.ime = true;
    } else if (a == "--realtime") {
      o.speed = 1;
    } else if (a == "--simulate-realtime") {
      o.simulate_realtime = true;
    } else if (a == "--verbose") {
      o.verbose = true;
    } else if (a == "--chunk-ms" && has_value) {
      o.chunk_ms = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--speed" && has_value) {
      o.speed = std::atof(argv[++i]);
    } else if (a == "--clip-timeout" && has_value) {
      o.clip_timeout_s = std::atof(argv[++i]);
    } else if (a == "--no-format") {
      o.format = false;
    } else if (a == "--update-context") {
      o.update_context = true;
    } else if (a == "--mask-offensive") {
      o.mask = true;
    } else if (a == "--no-multilang") {
      o.multilang = false;
    } else if (a == "--speaker-change") {
      o.speaker_change = true;
    } else if (a == "--api-key" && has_value) {
      o.api_key = argv[++i];
    } else if (a == "--config-file" && has_value) {
      o.config_file = argv[++i];
    } else if (a == "--context-name" && has_value) {
      o.context_name = argv[++i];
    } else if (a == "--dump-responses" && has_value) {
      dump_path = argv[++i];
    } else if (a == "--dump-dir" && has_value) {
      dump_dir = argv[++i];
    } else if (a == "--jsonl" && has_value) {
      jsonl_path = argv[++i];
    } else if (a == "--clips" && has_value) {
      std::ifstream list(argv[++i]);
      if (!list) {
        std::fprintf(stderr, "cannot read %s\n", argv[i]);
        return 1;
      }
      for (std::string line; std::getline(list, line);) {
        line = line.substr(0, line.find('\t'));
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty()) clips.push_back(line);
      }
    } else if (a == "--phrase" && has_value) {
      // TEXT or TEXT:BOOST.
      const std::string v = argv[++i];
      const size_t colon = v.rfind(':');
      o.phrases.emplace_back(v.substr(0, colon),
                             colon == std::string::npos ? 1.0f : std::stof(v.substr(colon + 1)));
    } else {
      std::fprintf(stderr, "unknown or incomplete option %s\n", a.c_str());
      return 2;
    }
  }
  if (clips.empty()) {
    std::fprintf(stderr, "no clips\n");
    return 2;
  }
  if (!dump_path.empty() && clips.size() > 1) {
    std::fprintf(stderr, "--dump-responses takes one clip; use --dump-dir\n");
    return 2;
  }
  if (!o.config_file.empty()) {
    std::ifstream in(o.config_file, std::ios::binary);
    o.saved_config.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (o.saved_config.empty()) {
      std::fprintf(stderr, "cannot read %s\n", o.config_file.c_str());
      return 1;
    }
  }

  Library lib;
  lib.handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib.handle) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  lib.create = reinterpret_cast<CreateFn>(dlsym(lib.handle, "CreateExtendedSodaAsync"));
  lib.destroy = reinterpret_cast<HandleFn>(dlsym(lib.handle, "DeleteExtendedSodaAsync"));
  lib.add_audio = reinterpret_cast<AddAudioFn>(dlsym(lib.handle, "ExtendedAddAudio"));
  lib.start = reinterpret_cast<HandleFn>(dlsym(lib.handle, "ExtendedSodaStart"));
  lib.mark_done = reinterpret_cast<HandleFn>(dlsym(lib.handle, "ExtendedSodaMarkDone"));
  lib.update_context = reinterpret_cast<UpdateContextFn>(dlsym(lib.handle, "UpdateRecognitionContext"));
  if (!lib.create || !lib.destroy || !lib.add_audio || !lib.start || !lib.mark_done) {
    std::fprintf(stderr, "missing SODA entry points\n");
    return 1;
  }

  FILE* jsonl = nullptr;
  if (!jsonl_path.empty() && !(jsonl = std::fopen(jsonl_path.c_str(), "w"))) {
    std::fprintf(stderr, "cannot write %s\n", jsonl_path.c_str());
    return 1;
  }
  const bool batch = clips.size() > 1;
  int failures = 0;
  for (size_t i = 0; i < clips.size(); ++i) {
    std::vector<char> pcm;
    int rate = 0, channels = 0;
    std::string error;
    Session s;
    s.print = !batch || o.verbose;
    if (!dump_path.empty()) s.dump = std::fopen(dump_path.c_str(), "wb");
    if (!dump_dir.empty()) {
      char name[32];
      std::snprintf(name, sizeof(name), "/%05zu.bin", i);
      s.dump = std::fopen((dump_dir + name).c_str(), "wb");
    }
    const int64_t t0 = NowMs();
    bool ok = ReadPcm16Wav(clips[i], &pcm, &rate, &channels, &error);
    if (ok) {
      const double duration = pcm.size() / 2.0 / channels / rate;
      const double timeout = o.clip_timeout_s > 0
                                 ? o.clip_timeout_s + (o.speed > 0 ? duration / o.speed : 0)
                                 : 0;
      Watchdog watchdog(timeout, [&] {
        char msg[64];
        std::snprintf(msg, sizeof(msg), "timeout: not finished after %.0f s", timeout);
        if (jsonl) {
          std::string record = "{\"clip\": ";
          AppendJsonString(&record, clips[i]);
          record += ", \"error\": ";
          AppendJsonString(&record, msg);
          record += ", \"sample_rate\": " + std::to_string(rate) + ", \"channels\": " +
                    std::to_string(channels) + ", \"duration_ms\": " +
                    std::to_string(static_cast<int64_t>(duration * 1000 + 0.5)) + "}\n";
          std::fputs(record.c_str(), jsonl);
        }
        std::fprintf(stderr, "%s: %s\n", clips[i].c_str(), msg);
      });
      ok = RunClip(lib, o, pcm, rate, channels, &s, &error);
    }
    const int64_t wall_ms = NowMs() - t0;
    if (s.dump) std::fclose(s.dump);
    if (!ok) {
      ++failures;
      std::fprintf(stderr, "%s: %s\n", clips[i].c_str(), error.c_str());
    }
    if (batch) {
      std::fprintf(stderr, "[%zu/%zu] %s: %zu finals, %.1f s\n", i + 1, clips.size(), clips[i].c_str(),
                   s.finals.size(), wall_ms / 1000.0);
    }
    if (jsonl) {
      std::lock_guard<std::mutex> lock(s.mu);
      std::string line = "{\"clip\": ";
      AppendJsonString(&line, clips[i]);
      if (!ok) {
        line += ", \"error\": ";
        AppendJsonString(&line, error);
      } else {
        const double duration = pcm.size() / 2.0 / channels / rate;
        std::string text, raw;
        line += ", \"sample_rate\": " + std::to_string(rate) + ", \"channels\": " + std::to_string(channels) +
                ", \"duration_ms\": " + std::to_string(static_cast<int64_t>(duration * 1000 + 0.5)) +
                ", \"finals\": [";
        for (size_t k = 0; k < s.finals.size(); ++k) {
          const Final& f = s.finals[k];
          line += k ? ", {\"text\": " : "{\"text\": ";
          AppendJsonString(&line, f.text);
          line += ", \"raw\": ";
          AppendJsonString(&line, f.raw);
          line += ", \"start_ms\": " + std::to_string(f.start_us / 1000) +
                  ", \"end_ms\": " + std::to_string(f.end_us / 1000) + "}";
          text += f.text;
          raw += f.raw;
        }
        line += "], \"text\": ";
        AppendJsonString(&line, text);
        line += ", \"raw\": ";
        AppendJsonString(&line, raw);
        line += ", \"partials\": " + std::to_string(s.partials) +
                ", \"stopped\": " + (s.done ? "true" : "false") +
                ", \"wall_ms\": " + std::to_string(wall_ms);
      }
      line += "}\n";
      std::fputs(line.c_str(), jsonl);
      std::fflush(jsonl);
    }
  }
  if (jsonl) std::fclose(jsonl);
  return failures == static_cast<int>(clips.size()) ? 1 : 0;
}
