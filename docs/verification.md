# Verification against libsoda

Everything here compares opensoda with Google's libsoda on the same audio. The reference runs come from two places:

- **Chrome itself.** Audio was played with Live Caption on, and the audio and results libsoda handled were captured
  from Chrome's speech process.
- **Offline libsoda,** for repeatable runs: a locally modified copy of Chrome's libsoda (not part of this project),
  driven by `soda_api_client`. It reproduces Chrome's result boundaries on the recordings to 0.01 s.

Most test recordings are synthesized speech. `tests/regress.py` reruns the comparisons; see
[tests/README.md](../tests/README.md).

## Responses

opensoda's responses have the same layout as the ones Chrome receives.

**Hypotheses.**

- `hypothesis(0)` is the formatted text, with marks attached. Finals list words separated by spaces. Partials list
  characters separated by spaces where the pack's punctuation settings segment text by character (ja-JP, cmn-Hans-CN,
  cmn-Hant-TW: `1 4 0 0 万`), and words otherwise (th-TH, which has no punctuation settings, and languages written
  with spaces).
- `hypothesis(1)` is the raw text.
- In languages written with spaces (en-US), every result after a session's first starts with a space, in both
  hypotheses.
- In ja-JP, a partial's raw text has no spaces (`テスト@GmailCO`). A final's raw text has no space before a word that
  began at a bare word-boundary piece (`テスト@ Gmailcomです` from `▁テ スト ▁ @ ▁g ma i l ▁ co m`).

**Word timing.** Each `HypothesisPart` carries both forms of its text and a start offset.

- As in libsoda, a word starts at the start of the frame where the label before its text was emitted (the segment
  start, for a segment's first label). A final starts at its first word.
- A label's frame is that of the earliest alignment the search merged. When a label extends a hypothesis whose labels
  another one already ended on the previous frame, the result keeps that one's frames if it survived that frame's
  pruning (its cost is merged either way). Otherwise, if a hypothesis that survived the previous frame extends the same
  labels, the result takes their frames from it. This is because libsoda keeps hypotheses in a prefix tree, where a
  surviving descendant keeps an earlier node, with its time, alive. In ja-JP, `▁g ma i l ▁ co m` takes the bare `▁`
  of a `▁g ma i l ▁ .` emitted seven frames before `co`.
- Where the segmenter gates 10 ms frames before subsampling (ko-KR, hi-IN, pt-BR, cmn-Hans-CN), a kept frame's time is
  that of the last 10 ms frame it stands for.
- Words the text normalizers rewrite keep the start of the word they replace, one for one where as many come out as
  went in. (hi-IN's models write romanized Hindi, which a denormalizer transliterates to Devanagari word by word.)

Result boundaries match libsoda's, and so do 99% of word starts on the clean Japanese recording (92% in dictation,
where the rest are a frame off).

**Word confidence.** These models compute no confidence score, and `word_confidence` follows libsoda's rule:

- A word preceded by a space in the raw text is `ASR_CONFIDENCE_LOW`. In languages written with spaces this counts the
  space before each result after a session's (or a switched-to pack's) first, whether or not the config buckets
  confidences.
- Every other word is `ASR_CONFIDENCE_UNKNOWN`.

So in en-US only the first word of the first result is `UNKNOWN`, while in ja-JP nearly every word is, except for one
like `Gmail` in `テスト@ Gmailcomです`. In th-TH, the one pack without punctuation, every word is `UNKNOWN`. This
matches every result compared, in all languages.

**Endpoint reason.** Every final's endpoint reason is `ENDPOINT_UNKNOWN`, even for one cut by the end of the audio, in
both modes and all three languages checked.

**Speech events.** Speech start and end events come from the segmenter.

- A segment, its encoder frames and its START_OF_SPEECH begin at the speech onset the endpointer reports after each
  restart.
- ja-JP's VAD reads speech from its first frames with the caption cluster id. So in Live Caption its START_OF_SPEECH
  opens the stream and follows each endpointed final at the restart point; in dictation it comes at the real onset.
  de-DE's caption VAD does the same.
- libsoda's VAD runs ahead of its decoder. After a segment with no words, it may report the next onset before the
  decoder reaches that segment's end; the restart then starts speech again earlier, at the restart point, with a
  second START_OF_SPEECH. opensoda does the same, and the final keeps the start its segment had.
- The stream ends with an END_OF_AUDIO event, timed at the start of the last VAD frame, not with an empty final. If the
  audio ends during speech, END_OF_AUDIO comes first, and the last final ends where an END_OF_SPEECH at that time
  would have ended it. This was checked on clips cut mid-sentence, in en-US and ja-JP, in both modes.

**Audio levels.** Audio level responses (type 6, `AudioLevelInfo`) are sent every 20 ms, as libsoda sends them.

- `rms` is the RMS of the audio libsoda resamples to 16 kHz (full scale 1; for 48 kHz input, taken before rounding to
  16-bit values), smoothed with a 100 ms time constant. The values are within one unit in the last place of libsoda's
  (87% identical), on 16 kHz and 48 kHz input and on Chrome's own capture.
- The first response after new output from the VAD's `AudioLevelEventStream` also carries `audio_level`, the latest
  frame's speech posterior, with that frame's start time. A last response after END_OF_AUDIO carries only the
  posterior. The posteriors are bit-identical except in the 2 s after a final: libsoda restarts its pipeline when its
  decoder catches up, 0 to 15 reports after END_OF_SPEECH depending on how far it lags, so the frames it re-reads
  there differ. Chrome doesn't use `audio_level`.

## Bit-exact stages

- Given the same audio, opensoda's encoder input is bit-identical to libsoda's. For en-US this was checked on a
  two-sentence recording, across the restart, with the gain control in place.
- Given the same frame batching, the encoder output is bit-identical too, and so are the VAD posteriors.
- Given libsoda's encoder outputs, the joint network's output is bit-identical.
- The two custom punctuation kernels, run inside our interpreter, give byte-identical output to libsoda's own kernels
  for the same model and input. Captured live from Chrome's speech process, all 195 distinct partial results got
  exactly the same punctuation from opensoda.

This takes libsoda's own arithmetic, which is why the build is set up the way it is:

- TFLite built without fused multiply-adds (`-ffp-contract=off`);
- the reference `BATCH_MATMUL` kernel;
- TFLite's NEON kernels for 4-bit fully connected layers. TFLite's CMake build leaves them out, and the scalar fallback
  rounds differently: with it, en-US partials and word starts differed from libsoda's in places;
- ruy for float matrix multiplies;
- the float `tanh` of the newer Eigen that libsoda's TFLite was built with, patched into TFLite's Eigen by the build
  (`cmake/eigen_ptanh_float.h`);
- the VAD's normalization dividing by the standard deviation.

The search's smoothed label costs (a temperature softmax in Eigen) can still differ from libsoda's by about one unit
in the last place.

## Why some results differ

libsoda isn't deterministic. Its encoder thread batches frames according to how far it lags behind the audio, and
its int8 dynamic quantization makes the encoder output depend on the batching. So libsoda's own result changes with how
fast it's fed and from run to run. Where opensoda differs from libsoda below, the difference is usually one that
libsoda's own runs show too.

## Results by language

### en-US

- **Captions** (22 synthesized sentences in three sets: plain sentences; prices, dates, percentages, phone numbers and
  addresses; spoken punctuation and times of day): every final is identical to libsoda's, including punctuation,
  written forms and capitalization, and so are the speech start and end events. All 277 partial texts libsoda shows
  appear in opensoda's partials, and every word start is identical to libsoda's.
- **Dictation (IME):** each of ten dictated sentences recognized on its own gives libsoda's final exactly. So do all 16
  finals of the three recordings that dictate them together, including `Type the word, and then the word period.`,
  where the second pass writes the command as a plain word in that context. Every partial text libsoda shows appears
  in opensoda's.
- **Contextual biasing:** four sentences with rare names, biased toward five phrases (`Siobhan Kowalczyk`, `Zorblax`,
  `Nguyen Thi Huong`, `Quillfeather`, `Xelvantra`). At boosts 1 and 3 every final matches libsoda's
  (`Siobhan Kowalczyk`, `Zorblax`, the rest unchanged), and at boost 3 so does every partial text. At boost 10 both
  over-trigger (libsoda's first pass repeats `Nguyen Thi Huong` over silence, and so does opensoda's), and one of the
  four finals matches.
- **Without a multilang config:** every final of six caption recordings (35 finals) and of one dictated recording (7)
  matches libsoda's raw and formatted text. So do the six finals of a two-voice conversation, with and without speaker
  change detection, `\n – ` marks included. On another conversation, three of four formatted finals match. There the
  speaker-turn search finds libsoda's labels, but libsoda takes their frames from its result lattice, where they land
  up to a few frames from where the search emitted them (61 of 161 labels on the same frame), so one turn falls a word
  later.

### ja-JP

- **Clean speech** (122 s, 16 synthesized sentences): the transcript is character-for-character identical to
  libsoda's, punctuation and result boundaries included. So is the transcript of the exact audio Chrome received
  during a capture (475 characters), compared with what Chrome's libsoda produced live. 261 of its 262 partial texts
  appear in opensoda's.
- **Dictation, same recording:** all 16 finals are identical to libsoda's, and so are the speech start and end events.
  273 of libsoda's 274 partials appear in opensoda's.
- **Dictation with spoken punctuation** (29 synthesized sentences with commands, alone and inside sentences): all 25
  finals and all 149 partials are identical to libsoda's. This includes `Gmail。 Com`, where the punctuation model puts
  a period inside `Gmailcom` and the capitalization after it applies inside the word too. 115 of the 121 word starts
  match. The others come from the encoder's batching: given the outputs libsoda's encoder produced in one run, all 93
  word starts of that run's 18 finals match, `com` included.
- **Noisy conversational video** (346 s lesson): 14–17% character difference against four offline libsoda runs. Both
  transcripts are poor on this audio, and libsoda itself varies 5–13% between runs. Both end 21 results, and the
  boundaries agree to the frame almost everywhere. Given libsoda's encoder outputs for this video, recorded from three
  libsoda runs, opensoda's results match libsoda's word for word in 11 to 13 of the first 12 or 13 results (about
  280 s); in each of the others, one word differs. After that, libsoda's segmentation itself varies from run to run.

### de-DE

- **Captions** (18 synthesized sentences in two recordings): the speech start and end events are identical to
  libsoda's, and so are 12 of 13 finals, raw text included. The eight sentences recognized one at a time all match.
  The other final is a first-pass decoding difference of the same kind as en-US's (`7 Euro` where libsoda's hypothesis
  has `7€` in that context; alone, both give `7 €`).
- **Dictation** (31 synthesized sentences with spoken punctuation commands: `komma`, `fragezeichen`, `neue zeile`,
  `klammer auf`, `hashtag`, `bindestrich` and others): 26 or 27 finals are identical to libsoda's, depending on
  libsoda's run. The others come from decoding. libsoda's own result for some of them changes between runs with how
  its encoder thread batches frames (`gut doppelpunktpunktpunktpunkt` or `gut! Doppelpunktpunktpunkt`,
  `max@ zeichenbeispielpunkte` or `max@zeichenbeispielpunkte`), and in one opensoda hears another word. Where the words
  agree, so do their start times, except for a few words that start one to three frames earlier.

### fr-FR

- **Captions** (8 synthesized sentences): the speech start and end events are identical to libsoda's, and so are all
  86 partials and 8 finals of libsoda fed at real-time speed. (Fed at 8×, libsoda itself hears `10 8 degrés` in one.)
  69 of 76 word starts match, the others a frame or two off. These come from the encoder's batching: given the outputs
  libsoda's encoder produced in one run, opensoda's word starts match that run's exactly, and libsoda disagrees with
  itself on 3 of 76 between runs.
- **Dictation:** 9 of 10 sentences with spoken punctuation give libsoda's final text (`Est-ce que tu viens ce soir ?`,
  `Voici la liste : du pain, du lait et des œuf.`). The other is a decoding difference (`ouvrez` for libsoda's
  `ouvrir`).

### Other packs (captions)

Five synthesized sentences each, in one recording per language.

| Pack | Partials and finals | Word starts |
|---|---|---|
| es-ES | all identical | 43 of 45 |
| it-IT | all identical | 39 of 42 |
| ru-RU | all identical | 29 of 32 |
| ko-KR | all identical | 24 of 24 |
| hi-IN | all identical | 38 of 38 |
| pt-BR | all identical | 11 of 12 |
| cmn-Hans-CN | all identical | 33 of 33 |
| id-ID | all identical (38 partials, 5 finals) | 29 of 31 |
| pl-PL | all identical (43, 5) | 34 of 34 |
| tr-TR | all identical (40, 5) | 26 of 29 |
| vi-VN | all identical (45, 5) | 42 of 44 |
| th-TH | 1 of 45 partials differs | |
| cmn-Hant-TW | 1 final differs | |

- The word starts that differ are a frame off, and every word confidence matches. The speech start and end events
  match in every pack.
- es-ES needs the CAPTION cluster id fallback (see [the pipeline](pipeline.md#recognition-domain)) for both the
  results and the events; with the configured id, its VAD and its numbers differ.
- pt-BR, cmn-Hans-CN and the last six packs are compared with libsoda fed at real-time speed. libsoda's own result in
  pt-BR and cmn-Hans-CN changes with how fast it's fed, and between runs with how its encoder thread happens to batch
  frames. cmn-Hans-CN's fourth sentence comes out as `我 花 了 15 块 钱 买 苹 果`, `火 花 了 15 块 钱 买 两 斤 苹 果` or
  `火 花 钱 买。`.
- pt-BR's libsoda drops three of the five results itself (no final for them), and opensoda does likewise.
- th-TH: another libsoda run gives opensoda's partials exactly.
- cmn-Hant-TW: the differing final is `晴又顏熱` for `晴朗又演熱`, which libsoda itself gives when its encoder thread
  batches frames differently (as it does under a debugger).
- Cased names and places match in id-ID, pl-PL, tr-TR and vi-VN.
- With the recordings at 48 kHz, which libsoda resamples, ten of these thirteen packs match every partial and final.
  In pt-BR, th-TH and cmn-Hant-TW the differences are ones libsoda's own runs show.

### Language identification and switching

Language identification events are identical to libsoda's (order, language, confidence level and switch result) on
ten recordings: 8,000 events in all, from en-US, ja-JP in both modes, a clip ending after a pause, and a recording that
switches ja-JP → en-US → ja-JP. That rests on the classifier's inputs and outputs being bit-identical to libsoda's.
The other packs' recordings match too (another 4,800 events), apart from one event each in id-ID and pt-BR at a chunk
boundary of libsoda's real-time run.

On the switching recording, the switches, the finals and their boundaries match libsoda's, to a VAD frame.

## Speed and memory

CPU time and peak memory of `soda_api_client` driving each library on an Apple silicon Mac, with the same audio, fed
in real time as Chrome does:

| Recording | opensoda CPU | libsoda CPU | opensoda memory | libsoda memory |
|---|---|---|---|---|
| en-US, 55 s | 4.9 s | 6.4 s | 179 MB | 233 MB |
| ja-JP, 122 s | 14.9 s | 20.0 s | 241 MB | 268 MB |

Fed faster (opensoda as fast as it goes with `simulate_realtime_testonly`, libsoda at 8×, at which it keeps up),
opensoda takes 3.2 s of CPU for en-US, 10.9 s for ja-JP and 1.7 s for 25 s of pl-PL, against libsoda's 4.4 s, 14.6 s
and 2.4 s. It finishes the 122 s ja-JP recording in 11 s. The results are the same either way.

Each model runs on one thread: the encoder takes a few frames at a time, and a second TFLite thread would only spin
waiting for work. `tests/bench.py` makes these measurements.
