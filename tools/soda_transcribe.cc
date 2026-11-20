// Transcribes a WAV file with an installed SODA language pack.
//
// Usage: soda_transcribe [--pack DIR | --language ja-JP] [--config NAME]
//                        [--domain CAPTION|IME] [--partials] [--threads N]
//                        [--beam N] [--endpoint-ms N] [--no-vad] [--langid]
//                        [--mask-offensive] [--dump-features OUT]
//                        [--dump-vad OUT] file.wav
//
// Without --pack, the newest pack Chrome installed for --language (default
// ja-JP) is used.
//
// --dump-features writes every encoder input frame to OUT as raw float32.
// --dump-vad writes "ms posterior state" per VAD frame to OUT.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "language_packs.h"
#include "recognizer.h"

namespace {

struct Wav {
  int sample_rate = 0;
  int channels = 0;
  std::vector<int16_t> samples;
};

bool ReadWav(const std::string& path, Wav* wav, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *error = "cannot open " + path;
    return false;
  }
  char riff[12];
  f.read(riff, 12);
  if (!f || std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(riff + 8, "WAVE", 4) != 0) {
    *error = path + " is not a RIFF/WAVE file";
    return false;
  }
  int bits = 0, format = 0;
  while (f) {
    char id[4];
    uint32_t size;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&size), 4);
    if (!f) break;
    if (std::memcmp(id, "fmt ", 4) == 0) {
      std::vector<char> fmt(size);
      f.read(fmt.data(), size);
      format = *reinterpret_cast<uint16_t*>(&fmt[0]);
      wav->channels = *reinterpret_cast<uint16_t*>(&fmt[2]);
      wav->sample_rate = static_cast<int>(*reinterpret_cast<uint32_t*>(&fmt[4]));
      bits = *reinterpret_cast<uint16_t*>(&fmt[14]);
    } else if (std::memcmp(id, "data", 4) == 0) {
      wav->samples.resize(size / 2);
      f.read(reinterpret_cast<char*>(wav->samples.data()), size);
      break;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);
    }
  }
  if ((format != 1 && format != 0xFFFE) || bits != 16 || wav->channels < 1) {
    *error = path + ": only 16-bit PCM WAV is supported";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  soda::RecognizerOptions options;
  std::string wav_path, language = "ja-JP";
  bool partials = false;
  bool langid = false;  // Print language identification events.
  std::string features_path, vad_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--pack") options.language_pack_dir = value();
    else if (a == "--language") language = value();
    else if (a == "--config") options.config_name = value();
    else if (a == "--domain") options.domain = value();
    else if (a == "--threads") options.num_threads = std::stoi(value());
    else if (a == "--beam") options.beam = std::stoi(value());
    else if (a == "--endpoint-ms") options.endpoint_silence_ms = std::stoi(value());
    else if (a == "--partials") partials = true;
    else if (a == "--no-vad") options.use_vad = false;
    else if (a == "--langid") langid = true;
    else if (a == "--mask-offensive") options.mask_offensive_words = true;
    else if (a == "--dump-features") features_path = value();
    else if (a == "--dump-vad") vad_path = value();
    else wav_path = a;
  }
  if (wav_path.empty()) {
    std::cerr << "usage: " << argv[0]
              << " [--pack DIR | --language ja-JP] [--config NAME] [--partials]"
                 " [--threads N] [--beam N] [--endpoint-ms MS] file.wav\n";
    return 2;
  }
  if (options.language_pack_dir.empty()) {
    options.language_pack_dir = soda::FindInstalledLanguagePack(language);
    if (options.language_pack_dir.empty()) {
      std::cerr << "no installed Chrome SODA language pack for " << language << "\n";
      return 1;
    }
  }

  Wav wav;
  std::string error;
  if (!ReadWav(wav_path, &wav, &error)) {
    std::cerr << error << "\n";
    return 1;
  }
  options.enable_language_id = langid;
  options.sample_rate = wav.sample_rate;
  options.channel_count = wav.channels;
  auto recognizer = soda::Recognizer::Create(options, &error);
  if (!recognizer) {
    std::cerr << error << "\n";
    return 1;
  }
  std::ofstream features;
  if (!features_path.empty()) {
    features.open(features_path, std::ios::binary);
    recognizer->set_feature_tap([&](const std::vector<float>& f) {
      features.write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
    });
  }
  std::FILE* vad_out = vad_path.empty() ? nullptr : std::fopen(vad_path.c_str(), "w");
  if (vad_out) {
    recognizer->set_vad_tap([&](int64_t ms, float p, int state) {
      std::fprintf(vad_out, "%lld %.4f %d\n", static_cast<long long>(ms), p, state);
    });
  }
  recognizer->set_language_callback([](const soda::LanguageEvent& e) {
    std::printf("langid %.3f %s %d\n", e.audio_ms / 1000.0, e.language.c_str(),
                e.confidence_level);
  });
  recognizer->set_callback([&](const soda::RecognitionEvent& e) {
    if (e.type == soda::RecognitionEvent::kFinal) {
      if (!e.text.empty()) {
        std::printf("[%7.2f - %7.2f] %s\n", e.start_ms / 1000.0, e.end_ms / 1000.0,
                    e.text.c_str());
      }
    } else if (partials) {
      std::fprintf(stderr, "  ... [%7.2f] %s\n", e.end_ms / 1000.0, e.text.c_str());
    }
    std::fflush(stdout);
  });
  // Feed in 100 ms chunks, as a live stream would arrive.
  const size_t chunk = static_cast<size_t>(wav.sample_rate / 10) * wav.channels;
  for (size_t off = 0; off < wav.samples.size(); off += chunk) {
    recognizer->AddAudio(wav.samples.data() + off,
                         std::min(chunk, wav.samples.size() - off));
  }
  recognizer->Finish();
  if (vad_out) std::fclose(vad_out);
  return 0;
}
