# opensoda

A C++ reimplementation of **libsoda**, the on-device speech recognizer (SODA) behind Chrome's Live Caption.

Chrome ships libsoda as a closed binary, and downloads a language pack of TensorFlow Lite models for each language its
user picks. This project rebuilds the library from reverse engineering, so that other applications, such as MacPopup,
can transcribe speech with the packs Chrome has already installed. It produces a drop-in `libsoda.dylib` with the same
C API, which runs the same models and, on the test recordings, gives the same results as Google's library.

**The models are not included.** opensoda reads a language pack from your own Chrome installation:

```
~/Library/Application Support/Google/Chrome/SODALanguagePacks/<lang>/<version>/SODAModels
```

The packs and Google's libsoda belong to Google. This project never ships or modifies them. Reverse-engineering
inputs (Google's binary, extracted configs, IDA databases) stay in the git-ignored `re/` directory; keep them out of
commits, releases and bug reports.

## Status

- **Same output as the original.** Compared with libsoda on the same audio, opensoda gives the same finals (including
  punctuation, capitalization and written forms like `$25.50`), the same partials, speech events, word timings and
  confidences, and the same language identification events. The remaining differences are ones libsoda's own runs
  show: its output changes with how fast it is fed. See [docs/verification.md](docs/verification.md).
- **Same C API.** The library exports the functions Chrome calls (`CreateExtendedSodaAsync`, `ExtendedSodaStart`,
  `ExtendedAddAudio`, `ExtendedSodaMarkDone`, `ExtendedSodaStop`, `DeleteExtendedSodaAsync`,
  `UpdateRecognitionContext`), and sends the same responses.
- **Faster.** It uses about 25% less CPU and less memory than libsoda on the same audio.
- **17 languages.** Every pack below runs and has been compared with libsoda:

  | Packs | Version | Checked in |
  |---|---|---|
  | ja-JP | 1.5072.0 | captions and dictation |
  | en-US | 1.5075.0 | captions and dictation |
  | de-DE, fr-FR | 1.5072.0 | captions and dictation |
  | es-ES, it-IT | 1.5071.0 | captions |
  | ru-RU, pt-BR, cmn-Hans-CN, cmn-Hant-TW, id-ID, pl-PL, th-TH, tr-TR, vi-VN | 1.3071.0 | captions |
  | hi-IN | 1.3072.0 | captions |
  | ko-KR | 1.3073.0 | captions |

  All were checked with SODA 1.2.12.
- **Features:** captions and dictation (IME) modes, punctuation and formatting, spoken punctuation in dictation,
  contextual biasing (en-US), language identification and switching between packs, speaker change detection (en-US),
  and offensive-word masking.
- **Not implemented:** parts of libsoda likely used in Android: entity injection, result prefetching, and the
  one-shot (`SHORT`) configs used for voice queries. LAS rescoring is also left out: the packs give it no weight. See
  [Not implemented](docs/pipeline.md#not-implemented).

## Building

You need CMake 3.20 or later, Ninja, and a C++17 compiler. Check out two dependencies under `third_party/`:

| Path | Version |
|---|---|
| `third_party/tensorflow` | TensorFlow v2.18.0 |
| `third_party/marisa-trie` | marisa-trie v0.3.1 |

Then build:

```sh
cmake -S . -B build -G Ninja
ninja -C build
```

Only TensorFlow's `tensorflow/lite` is built, with XNNPACK off. On arm64 the build also compiles TFLite's NEON kernels
for 4-bit fully connected layers, which TFLite's CMake build leaves out. It replaces TFLite's CPU check with one that
also finds the dot-product instructions on macOS (`cmake/tflite_cpu_check.cc`). The checkout itself isn't modified.
These details matter for matching libsoda's numbers; see
[docs/verification.md](docs/verification.md#bit-exact-stages).

The build produces:

| Output | What it is |
|---|---|
| `libsoda.dylib` | the drop-in replacement for libsoda |
| `soda_transcribe` | transcribes a WAV file |
| `soda_api_client` | drives any libsoda-compatible library (this one or Google's) through the C API, as Chrome does |
| `soda_punctuate` | runs a pack's punctuation model on tokens |
| `soda_capitalize` | cases words with a pack's Roadrunner capitalization model |

## Using it

### From the command line

```sh
build/soda_transcribe speech.wav                       # the newest ja-JP pack Chrome installed
build/soda_transcribe --language en-US speech.wav
build/soda_transcribe --pack ".../SODALanguagePacks/ja-JP/1.5072.0" speech.wav
build/soda_transcribe --domain IME --partials speech.wav
```

Input can be 16-bit PCM or float WAV, at any sample rate, mono or interleaved. Other options:

- `--langid` prints language identification events.
- `--mask-offensive` masks offensive words.
- `--no-vad` turns off segmentation by voice activity.
- `--dump-features OUT` writes the encoder's input frames as raw float32.
- `--dump-vad OUT` writes the VAD posterior and endpointer state for each frame.

`soda_api_client` runs a whole session through the C API. It's useful for testing an integration, or for comparing
opensoda with another libsoda-compatible library:

- `--speed X` paces the audio at X times real time, and `--chunk-ms N` sets the audio per `ExtendedAddAudio` call
  (100 ms by default). `--simulate-realtime` sets `simulate_realtime_testonly`.
- `--multilang LOCALE=DIR` lists another installed pack for language switching. `--no-multilang` sends no multilang
  config, and `--speaker-change` turns on speaker change detection.
- `--phrase TEXT[:BOOST]` sends a biasing phrase, in the config or, with `--update-context`, through
  `UpdateRecognitionContext` after the start. `--context-name` changes the context it's sent under.
- `--dump-responses F` saves every raw response.
- Given several clips (or `--clips LIST`), it loads the library once and gives each clip its own session, as Chrome
  does for each media stream. `--jsonl F` writes each clip's finals.

### From an application

Load `libsoda.dylib` with `dlopen` and use it exactly as Chrome uses Google's library. The declarations are in
`include/soda_api.h`, and `tools/soda_api_client.cc` is a complete client.

1. Build a serialized `speech.soda.chrome.ExtendedSodaConfigMsg` (fields below).
2. Call `CreateExtendedSodaAsync({config, size, callback, handle})`, then `ExtendedSodaStart`.
3. Feed interleaved 16-bit PCM at the configured rate with `ExtendedAddAudio`, as it arrives. Chrome sends 10–20 ms
   chunks.
4. At the end of the stream, call `ExtendedSodaMarkDone`. An `END_OF_AUDIO` endpoint event and any last final are
   followed by `STOP`. Then call `DeleteExtendedSodaAsync`.

Start a new session when the sample rate or channel count changes, as Chrome does.

#### Config fields

| Field | Number | Meaning |
|---|---|---|
| `channel_count` | 1 | |
| `sample_rate` | 2 | |
| `simulate_realtime_testonly` | 5 | set it for audio from a file fed faster than real time (see below) |
| `api_key` | 6 | ignored by opensoda |
| `language_pack_directory` | 7 | the pack's `SODAModels` path |
| `recognition_mode` | 8 | 2 = captions, 1 = dictation (IME) |
| `enable_formatting` | 12 | punctuation and formatting on or off |
| `enable_speaker_change_detection` | 13 | marks where another speaker starts (en-US); `speaker_diarization_mode` (17) = 2 does the same |
| `multilang_config` | 15 | `{1: {1: locale, 2: SODAModels path}, ...}`, one entry per installed pack, as Chrome sends it |
| `mask_offensive_words` | 16 | masks offensive words with asterisks |
| `recognition_context` | 19 | phrases to bias toward: `{1: {1: context name, 2: {1: {1: phrase, 2: boost}, ...}}}` |

A few fields change more than their name says, as they do in libsoda:

- **`multilang_config`** turns on language identification and switching between the listed packs. Without it, each
  final holds back its unfinished last sentence, which leads the next result. Chrome always sends it.
- **Speaker change detection** only works without a `multilang_config`; with one, libsoda ignores it.
- **Biasing phrases** only work in packs with a biasing model (en-US), and only under that model's context name,
  `android-speech-api-generic-phrases`. `UpdateRecognitionContext` replaces them mid-stream.

**Feeding audio faster than real time.** libsoda keeps captions close to live audio. When recognition falls behind,
it skips all but the last second of the unrecognized audio (5 s with a `multilang_config`), and so does opensoda. To
recognize a file fed faster than real time, set `simulate_realtime_testonly`: opensoda then recognizes all of the audio
as fast as it can.

#### Responses

The callback receives serialized `SodaResponse` messages on a worker thread:

| Type | Number | Contents |
|---|---|---|
| `RECOGNITION` | 1 | a `SodaRecognitionResult`: `PARTIAL` (1) or `FINAL` (2), its hypotheses, timing and word parts |
| `ENDPOINT` | 5 | a `SodaEndpointEvent`: `START_OF_SPEECH`, `END_OF_SPEECH`, and `END_OF_AUDIO` at the end of the stream |
| `LANGID` | 7 | a `SodaLangIdEvent`: the language (`ja-jp`), a confidence level from 0 to 3, and the switch result |
| `START`, `STOP`, `SHUTDOWN` | 4, 2, 3 | no payload |

- `hypothesis(0)` is the formatted text Chrome shows. In Japanese and Chinese it separates characters (in partials)
  or words (in finals) with spaces. To show such text without spaces, use `hypothesis(1)` (the raw text), or remove
  the spaces between characters of scripts written without them.
- A result's `TimingMetrics` has only its start and end in the audio (fields 2 and 4), as in libsoda.
- `LANGID` events (about 15 per second of speech) are sent when the config has `enable_lang_id` or a
  `multilang_config`. After a successful switch, results come from the new language's pack.
- With speaker change detection, the formatted text puts `\n – ` before each word where another speaker starts, and
  that word's part has field 3 set to 1.

[docs/verification.md](docs/verification.md#responses) describes the responses in full.

## How it works

opensoda builds the recognition pipeline from the language pack's config, as libsoda does, so packs with different
architectures run from the same code. Each stage reimplements a part of libsoda:

| Stage | What it does | Code |
|---|---|---|
| Front end | Resamples, applies gain control, and computes normalized log-mel frames | `src/frontend.cc` |
| Voice activity and endpointing | An LSTM VAD and libsoda's endpointer split the audio into speech segments | `src/endpointer.cc`, `src/vad_frontend.cc` |
| Encoder | A streaming Conformer; en-US adds a non-causal second pass for the finals | `src/recognizer.cc` |
| Search | libsoda's breadth-first transducer beam search, with optional phrase biasing | `src/transducer.cc`, `src/biasing.cc` |
| Text formatting | Denormalization grammars, truecasing, punctuation, spoken punctuation in dictation, offensive-word masking | `src/fst_normalizer.cc`, `src/truecaser.cc`, `src/roadrunner_capitalizer.cc`, `src/punctuator.cc`, `src/spoken_punctuation.cc`, `src/pqrnn_spoken_punctuation.cc` |
| Language identification | A separate classifier pipeline that can switch the session to another pack | `src/langid.cc` |
| Speaker change detection | A separate transducer that marks speaker turns, placed among the words | `src/speaker_turn.cc` |
| C API | The functions, protos and threading model Chrome uses | `src/soda_api.cc`, `include/soda_api.h` |

[docs/pipeline.md](docs/pipeline.md) describes each stage in detail: the pack files it uses, and the details of
libsoda's behaviour it reproduces.

## Testing

`tests/regress.py --check` runs 40 cases across all 17 packs (captions, dictation, biasing, language switching,
speaker change detection). For each one, it compares every partial, final, word timing and confidence, speech event
and language identification event with libsoda's. `tests/corpus.py` runs a corpus of clips through opensoda or
libsoda the way Chrome captions them, one session per clip, and compares the runs. `tests/bench.py` measures CPU time
and memory.

The test recordings and libsoda's responses stay local, in `tests/data/`. See [tests/README.md](tests/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | the library |
| `include/` | the C API header |
| `tools/` | `soda_transcribe`, `soda_api_client` and the other command-line tools |
| `tests/` | regression, corpus and benchmark scripts |
| `cmake/` | build helpers for TensorFlow Lite |
| `docs/` | how the pipeline works, and how it was verified |

## License

This project is licensed under the [Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/); see `LICENSE`.

Two build helpers are based on third-party code and keep its licence:
- `cmake/eigen_ptanh_float.h`: Eigen's float `tanh`, MPL-2.0.
- `cmake/tflite_cpu_check.cc`: a modified TensorFlow Lite `cpu_check.cc`, Apache License 2.0.

The license covers only this project's code, not Google's libsoda or the language packs, which you supply yourself.
