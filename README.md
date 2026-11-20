# opensoda

An independent C++ reimplementation of **libsoda**, the on-device speech
recognizer (SODA) that Chrome uses for Live Caption. It runs the language packs
that Chrome has already downloaded, so other applications, such as MacPopup,
can transcribe audio with them. The purpose is interoperability.

## Models and binaries are not part of this project

The recognition models (TFLite files, FSTs, symbol tables and configs) and
Google's `libsoda.so` belong to Google. **This repository does not contain
them, and it must never redistribute them.** At run time, opensoda reads a
language pack from the user's own Chrome installation:

```
~/Library/Application Support/Google/Chrome/SODALanguagePacks/<lang>/<version>/SODAModels
```

Reverse-engineering inputs (Google's binary, extracted configs, IDA databases
and decoded endpointer models) live in `re/`, which is git-ignored. Keep them
out of commits, releases and bug reports.

## What is implemented

The supported packs are ja-JP 1.5072.0 and en-US 1.5075.0 with SODA 1.2.12;
de-DE and fr-FR 1.5072.0 run too (checked against libsoda on synthesized
speech, in captions and dictation), and so do es-ES and it-IT 1.5071.0, ru-RU
1.3071.0, ko-KR 1.3073.0, hi-IN 1.3072.0, pt-BR 1.3071.0 and cmn-Hans-CN
1.3071.0 (checked in captions). id-ID, pl-PL, th-TH, tr-TR, vi-VN and
cmn-Hant-TW 1.3071.0 run too (checked in captions).
libsoda loads `ONDEVICE_LARGE_CONTINUOUS` for both captions and dictation
(IME), or `ONDEVICE_MEDIUM_CONTINUOUS` in packs without it (ru-RU, ko-KR,
hi-IN, pt-BR, cmn-Hans-CN and the six above); the mode only selects the
recognition domain.
The pipeline is built from the pack's config, so the packs' different
architectures (below) run from the same code. The packs' other configs
(`ONDEVICE_LARGE_SHORT` and others) can be selected with `--config` but are
not what Chrome runs.

| Stage | Pack files | Notes |
|---|---|---|
| Audio front end | `*.mean_stddev`, config | Resampling, automatic gain control where the config has one (en-US: greco's `SAMPLE_LEVEL_TARGET_RANGE` estimator, restarted with the pipeline), log-mel filterbank, frame stacking and normalization, following libsoda's arithmetic: its 17-tap FIR resampler for 48 kHz input, its Hann table and windowing (the first 400 of each 401-sample frame, HTK-style pre-emphasis of the first sample), its dither table and phase, its split-radix FFT, double-precision log and division by the standard deviation. A stacker without `frame_zero_padding` fills its first frame's left context with copies of that frame. On the same audio the encoder input is bit-identical to libsoda's |
| Conformer encoder | `endtoendmodel/large-encoder.tflite` | Streaming, with time-batched invocation (`time_batch_size`, and `time_batch_size_on_flush` at a segment end). Frames are encoded in the smallest batch, as libsoda does when it keeps up with live audio; the encoder's int8 dynamic quantization makes its output depend on the batching. libsoda's encoder thread encodes the largest batch that fits what is queued. Where the segmenter gates 10 ms frames (ko-KR, hi-IN, pt-BR, cmn-Hans-CN) and releases a backlog at a segment start, libsoda (logged at real-time speed) encodes the smallest batch from it, then the largest that fits the rest, and opensoda does the same. en-US's encoder also takes the recognition domain's id (`domain_id_mapping`) |
| Cascaded second pass (en-US) | `large-cascaded_encoder`, `large-decoder_1`, `large-joint_posterior_1` | A non-causal encoder over the first encoder's output, with its own decoder and search settings, gives the finals; the first pass gives the partials. As in libsoda's `EncoderInsertRightContextStream` and `EncoderUnshiftStream`, its output for a frame comes 15 frames later and takes that frame's time, and at a segment end 15 copies of the last frame let the last frames through. Its internal-LM settings (`internal_lm_cost_scale` 0.26 with `subtract_internal_lm` false) leave the costs unchanged in greco's `SmoothedCosts`, so `large-joint_prior_1` is not needed. The second-pass model writes numbers, times, prices, phone numbers and addresses in written form itself (`7:00`, `$25.50`, `555-123-4567`) |
| RNN-T decoder and HAT joint | `large-decoder`, `large-joint_posterior` | greco's breadth-first transducer search over the pack's `SigmaFst` graph (which accepts any wordpiece sequence), with temperature smoothing, `</S>` handling and `eos_penalty`. From the search config: histogram pruning to about `max_arcs` hypotheses (3) and at least `min_arcs`, the `local_beam`, the word penalty, the label-cost limit (`max_model_cost`) and `max_expansions`. Given the same encoder outputs it produces libsoda's transcript. The pack's internal-LM weight is 0, so `large-joint_prior` is not needed |
| Recognition domain | config `domain_name_to_internal_id` | The acoustic cluster id fed to the encoder and the VAD. Each AppendClusterIdStream looks the domain up in its own map; a domain the map lacks gets its enum value (FARFIELD 1, IME 2, TELEPHONY 3, CAPTION 4, VOICE_SEARCH 7), as libsoda does: en-US's VAD and es-ES's both list no CAPTION, and libsoda feeds them 4. The configs' `forced_cluster_id` is only a default |
| Segment continuation | `enable_resume_search` | In CONTINUOUS, a new segment keeps the previous one's decoder context |
| Voice activity / endpointing | `acousticmodel/*.endpointer_portable_lstm_model` | The files are TFLite XOR-ed with a one-byte key; ko-KR's and hi-IN's use a `Relu1` custom op (clamp to [0, 1]). Most packs' VAD reads the stacked filterbanks with a one-hot cluster id every 30 ms. ko-KR's, hi-IN's and pl-PL's reads 40 log mel energies every 10 ms, of a noise-subtracted spectrum (pl-PL's VAD windows the audio itself, without the main path's dither): greco's SpectralNoise tracks per-bin percentiles (95th and 25th) of all FFT energies seen so far (a sorted history halved once it holds 199 values, then sampled every other frame) and holds back the first 20 frames until it has an estimate; SpectralSubtractor then subtracts the 25th-percentile noise magnitude, scaled by the noise level and the peak-to-floor ratio and floored at 0.2 of the magnitude. In those packs the segmenter also gates 10 ms frames, before they are subsampled to 30 ms (the count restarts with each segment). greco's frame-counting endpointer is ported, including its delayed output that pads each utterance by `endpoint_margin`, so pauses under about 0.6 s stay in one segment. As in libsoda's graph, the segmenter sits before the encoder: frames wait for the endpointer's decision, only speech frames are encoded, and a result ends at the last encoder frame boundary before the segment end (Chrome's END_OF_SPEECH event). Then, like libsoda, recognition restarts at that point: the front end, VAD, segmenter and encoder start afresh and re-process the audio since then from a 30 s history, and the next result starts there (or at its first word, after a silence the endpointer kept). The search starts afresh there too, which matches libsoda's output better than keeping its decoder context (`enable_resume_search`). Segments are cut at `impatience_limit_ms` (60 s); `impatience_start_ms` only shortens the trailing silence a segment needs, which is already zero in these configs |
| Punctuation | `SODA_punctuation_model.tflite`, `SODA_punctuation_config.pb` | ja-JP: pQRNN with seq_flow_lite custom ops reimplemented. Tokens are characters, the mark is the argmax per token, and partials get no trailing mark. de-DE's pQRNN segments by whitespace (the config's `segment_by`), so its tokens are words (without marks they start or end with) and a mark follows its word, unless the word already ends in one of the config's marks; its output tensors carry a `pqrnn_model/pqrnn_encoder/` prefix. en-US: greco's Roadrunner tagger, with its `CharNGramFeatures` op (hashed character n-grams, libsoda's Fingerprint2011 hash) reimplemented. It takes one lowercased string per word plus an empty one, in windows of 30 words overlapping by 2 and 5; each word's label names the mark before it (from the config's codebook). Then the first token and each token after one ending in an end-of-sentence mark is capitalized: libsoda sets every pack's capitalization behavior to `CAPITALIZE_ALL`, whatever the config says (ja-JP's has none). A word is capitalized at its first letter or digit, as libsoda's casing does (`#urlaub` → `#Urlaub`, `(1)` unchanged); where tokens are characters (ja-JP, cmn-*), a mark inside a word capitalizes the next character (`シャンTfaceb` → `シャンT。Faceb`) |
| Truecasing | `denorm/transducer*.fst`, `denorm/lm*.sorted.fst` | greco's GeneralCapitalizationNormalizer: case variants per word, chosen by Viterbi under the OpenGrm n-gram LM (for example `sns` → `SNS`). de-DE's transducer brackets the sentence with `<S>` … `</S>`; the end is scored with the LM's `</S>` arc, but libsoda scores the first word from the LM's unigram state, not after `<S>` (so `herr Schmidt` becomes `Herr Schmidt` while `ich` and `das` stay lowercase, as in libsoda's raw text). Backoff arcs count as plain epsilons: a word costs the cheapest of its arcs at every backoff level, not only the first level that has it, and the search keeps each state reached that way. A word is looked up lowercased and without the punctuation it starts or ends with (`Viele,` is scored as `viele`). As in libsoda, words the transducer's symbol table lacks, and words of punctuation only, are taken out before the search and put back unchanged, so they give no context: in fr-FR `ouvrez les guillemets bonjour` keeps `bonjour`, which after `<UNK>` would become `Bonjour` |
| Truecasing (id-ID, pl-PL, tr-TR, vi-VN) | `denorm/SODA_capitalization_model.tflite`, config `RuntimeRoadrunnerCapitalizationNormalizer` | A TFLite text-normalization model that reads and writes a serialized `speech.textnorm.Sentence`: a character n-gram tagger labels each word "keep" or "rewrite", and a character seq2seq beam search rewrites the latter (capitalizes them). Its five custom ops, which move the words between the proto and tensors (`SentenceToInputContext`, `RetokenizeSentence`, `SentenceToTokenTensors`, `AttachHypothesisToSentence`, `TokenPostProcess`, which puts back an input character the model wrote as `<unk>`), are reimplemented; on libsoda's own model inputs the outputs match it byte for byte. As in libsoda, the words are cased in windows of 5 that overlap by one word each side |
| Denormalizers | `denorm/*.mfar`, config `common_denormalizers_list` | The config's FstNormalizer and ClassNormalizer grammars (memory-mapped FST archives, as for the masker, whose FSTs apply one after another, as in fr-FR's emoji grammar: insert markers, rewrite the marked words, delete the markers; arcs that read no byte insert their output, as in `5 p.m` → `5 p.m.`), applied in list order to the words joined by spaces (those listed before the capitalization normalizer before truecasing), before punctuation. The ClassNormalizer (`embedded_class_denorm`) only rewrites spans the model tags with one of greco's 30 word classes, tags included (`<money> sieben euro </money>` → `7 €`, `<time> zwölf uhr </time>` → `12 Uhr`): a span it rejects, a tag of another class inside it or a missing closing tag drop just the tags. In ja-JP: decoration removal and pronunciation cleanup. In en-US: COVID-19, a.m./p.m. fixes, the class denormalizer and the general normalizer always, and the spoken-emoji and spoken-punctuation grammars (typed 4 and 3 in the list) only in IME mode, as in libsoda. In captions `smiley face` becomes `:-)`, and in IME `comma` becomes `,` and `new paragraph` a paragraph break. Like libsoda's formatting, closing marks join the word before, opening brackets the word after, and a line or paragraph break is dropped while the next word is capitalized, in a final with the spaces around it (`file {new-paragraph} and` → `fileAnd`), in a partial leaving one (`file And`). The punctuation model adds no mark after a word that already ends in one of its codebook's marks |
| Spoken punctuation (ja-JP dictation) | `denorm/PUNCTUATION_LSTM.model.int8.tflite`, `.syms`, `punctuation_converter_config.pb` | greco's PunctuationNormalizer, applied in dictation only, after the other denormalizers. The words are joined without the spaces Japanese does not use; the text is one token if the model's vocabulary has it, else one per character. A bidirectional LSTM, starting from a zero state for each result, flags the tokens that are commands, and a flagged run that spells one of the converter config's spoken forms becomes that form's mark and a word of its own, timed over the characters it replaces (paired brackets and quotes alternate, opening first). libsoda's segmenter at this step never matches a non-ASCII byte, so in ja-JP a command converts only when it is the whole result: `当店` alone becomes `、`, `ハイフン` becomes `-` and then `-。`, `新しい段落` a paragraph break, and a result of only line breaks is not sent at all, as in libsoda |
| Spoken punctuation (de-DE, fr-FR dictation) | `denorm/model.tflite`, config `PortablePqrnnSpokenPunctuationNormalizerParams` | greco's PortablePqrnnSpokenPunctuationNormalizer, applied in dictation only. The words are split at the punctuation they contain (`12,50` → `12` `,` `50`); a pQRNN model labels each piece as a command or not, and a run of command pieces that spells one of the settings' spoken forms (longest match first, no backtracking) becomes its mark. The mark's type places it: closing marks join the word before (`anna komma` → `anna,`), opening ones the word after (`hashtag urlaub` → `#urlaub`), joining ones both (`post bindestrich west` → `post-west`), and line breaks stand alone. A spoken sentence end capitalizes the next word and then the first one; the pieces of one recognized word are joined back, and de-DE's postnormalizers replace typographic quotes with straight ones. Each resulting word takes the times of the words it came from. fr-FR's marks `? ! : ; « »` are set apart by a space; libsoda's results then list each space-separated part as a word (`liste :` gives `liste` and `:`), except the marks the punctuation model itself writes (`soir ?` stays one word), and time a part by its byte position in the word's frames, rounded to a frame |
| Offensive-word masking | `denorm/porn_normalizer_ondevice.mfar` | Off by default; `mask_offensive_words` turns it on. The config's `PORN_DENORMALIZER` grammar is a byte-level transducer in greco's memory-mapped FST archive format, applied like libsoda's FstNormalizer: the cheapest path over the punctuated words joined by spaces (`ちんぽ` → `ち**`, `shit` → `s***`). It masks finals and partials. The raw (unformatted) text is not masked |
| Language identification | `configs/LANGUAGE_IDENTIFICATION.config`, `langid/*`, `acousticmodel/SODA_DICTATION_EP_UNIFIED_FRONTEND_LANGID.*` | A separate pipeline: greco's automatic gain control, 32 ms log-mel frames, its own VAD and segmenter, then the streaming classifier (79 outputs, 47 mapped languages). Each output gives the top language and a confidence level from greco's scoring (probability margin and negentropy against two thresholds) |
| Language switching | `multilang_config` packs | As in libsoda: the first highly confident (level 3) language identification of another language whose pack the config lists switches to that pack (`SWITCH_SUCCEEDED`), and a language without one is reported as `SWITCH_SKIPPED_NO_LP`. The language identifier has one class for several locales, and libsoda's table of them (`es-us` for es-ES, `cmn-hans-cn` for cmn-Hant-TW, `en-us` for en-GB, ...) decides both: `es-us` in an es-ES session is the current language, and `es-us` switches to an es-ES pack. The new pack's recognizer takes over the audio history and the language identifier and recognizes again from the last final (`rewind_when_switching_language`), so the sentence that triggered the switch is recognized in the new language. On a recording switching ja-JP → en-US → ja-JP, the switches, the finals and their boundaries match libsoda's, to a VAD frame |
| Contextual biasing (en-US) | `endtoendmodel/large.wpm.portable`, config `class_substitution_twiddler` | Phrases from `recognition_context` or `UpdateRecognitionContext` are split into word pieces with the pack's byte-pair word-piece model (piece ids give the merge order) and built into a prefix trie with failure links, as libsoda's FST twiddler does. Both passes consult it: each matching piece lowers the cost by the phrase's boost plus `grammar_token_base_weight` (libsoda's token generator, with `BOOST_STRATEGY_PER_TOKEN_CONSTANT` spread uniformly over the pieces), a partial match that breaks off gives its reward back, a completed phrase keeps it, and going on after one costs `retwiddle_penalty`. As in greco's `DualSearchParams`, the label-cost limit and the pruning look at the cheaper of the biased and unbiased costs (`use_min_of_original_and_influenced_cost_for_pruning`), and the second pass biases no label costlier than `max_model_cost_to_influence`. |
| C API | `include/soda_api.h` | The entry points, protos and threading model that Chrome uses |

### Verified against Google's libsoda in Chrome

The two custom punctuation kernels give the same output as libsoda's own
kernels for the same model and input. Run inside our interpreter, their outputs
are byte-identical.

Captured live from Chrome's speech process, every one of 195 distinct partial
results got exactly the same punctuation from opensoda.

Response layout follows what Chrome receives:

- `hypothesis(0)` is the formatted text. Partials list characters separated by
  spaces where the pack's punctuation settings segment text by character
  (ja-JP, cmn-Hans-CN, cmn-Hant-TW; `1 4 0 0 万`), words otherwise (th-TH,
  which has none, and languages written with spaces); finals list words
  separated by spaces. Marks are attached. In
  languages written with spaces (en-US), every result after a session's
  first starts with a space, in both hypotheses.
- `hypothesis(1)` is the raw text.
- Each `HypothesisPart` carries both forms of its text, plus a start offset.
  As in libsoda, a word starts at the start of the frame where the label
  before its text was emitted (the segment start for a segment's first
  label), and a final starts at its first word. A label's frame is that of
  the earliest alignment the search merged: when a label extends a
  hypothesis whose labels another one already ended on the previous frame,
  the result keeps that one's frames if it survived that frame's pruning
  (its cost is merged either way). Otherwise, if a hypothesis that survived
  the previous frame extends the same labels, the result takes their
  frames from it: libsoda keeps hypotheses in a prefix tree, and a
  surviving descendant keeps an earlier node, with its time, alive (in
  ja-JP `▁g ma i l ▁ co m` takes the bare `▁` of a `▁g ma i l ▁ .` emitted
  seven frames before `co`). Where the segmenter gates 10 ms
  frames before subsampling (ko-KR, hi-IN, pt-BR, cmn-Hans-CN), a kept
  frame's time is that of the last 10 ms frame it stands for. Words the
  text normalizers rewrite keep the start of the word they replace, one
  for one where as many come out as went in (hi-IN's models write
  romanized Hindi, which a denormalizer transliterates to Devanagari word
  by word). Result boundaries match libsoda's,
  and so do 99% of word starts on the clean Japanese recording (92% in
  dictation, where the rest are a frame off).
- Each part's `word_confidence` follows libsoda's: a word that a space
  precedes in the raw text is `ASR_CONFIDENCE_LOW` (these models compute
  no confidence score), counting the space before each result after a
  session's (or a switched-to pack's) first in languages written with
  spaces, whether or not the config buckets confidences; every other word
  is `ASR_CONFIDENCE_UNKNOWN`. So in en-US only the first word of the first
  result is `UNKNOWN`, and in ja-JP nearly every word is, except one like
  `Gmail` in `テスト@ Gmailcomです`. In th-TH, the one pack without
  punctuation, every word is `UNKNOWN`. This matches every result
  compared, in all languages.
- Every final's endpoint reason is `ENDPOINT_UNKNOWN`, even one cut by the
  end of the audio, in both modes and all three languages.
- Speech start and end events come from the segmenter. A segment, its
  encoder frames and its START_OF_SPEECH begin at the speech onset the
  endpointer reports after each restart. ja-JP's VAD reads speech from its
  first frames with the caption cluster id, so in Live Caption its
  START_OF_SPEECH opens the stream and follows each endpointed final at the
  restart point; in dictation it comes at the real onset. de-DE's caption
  VAD does the same. libsoda's VAD runs ahead of its decoder: after a segment
  with no words it may report the next onset before the decoder reaches that
  segment's end, and the restart then starts speech again earlier, at the
  restart point, with a second START_OF_SPEECH. opensoda does the same; the
  final keeps the start its segment had. The stream ends
  with an END_OF_AUDIO event a VAD frame before the end of the last one, not
  with an empty final. If the audio ends during speech, END_OF_AUDIO comes
  first, and the last final ends where an END_OF_SPEECH at that time would
  have ended it (checked against libsoda on clips cut mid-sentence, en-US
  and ja-JP, both modes).
- Audio level responses (type 6, `AudioLevelInfo`), as libsoda sends them
  every 20 ms: `rms` is the RMS of the audio libsoda resamples to 16 kHz
  (full scale 1, for 48 kHz input before rounding to 16-bit values),
  smoothed with a 100 ms time constant; and the first response after new
  output of the VAD's `AudioLevelEventStream` also carries `audio_level`,
  the latest frame's speech posterior, with its frame's start time. A last
  response after END_OF_AUDIO carries only the posterior. The `rms` values
  are within one unit in the last place of libsoda's (87% identical), on
  16 kHz and 48 kHz input and on Chrome's own capture. The posteriors are
  bit-identical except in the 2 s after a final: libsoda restarts its
  pipeline when its decoder catches up, 0 to 15 reports after
  END_OF_SPEECH depending on how far it lags, so the frames it re-reads
  there differ. Chrome does not use `audio_level`.

Accuracy is measured against Google's libsoda on the same audio. The
reference runs are Chrome's own (audio played with Live Caption on, with the
audio and results libsoda handled recorded) and, for repeatable runs, a
locally modified copy of Chrome's libsoda driven by `soda_api_client --speed 8`
(not part of this project). The latter reproduces Chrome's result boundaries
on the recordings to 0.01 s.

- en-US (22 synthesized sentences in three sets: plain sentences; prices,
  dates, percentages, phone numbers and addresses; spoken punctuation and
  times of day): every final is identical to libsoda's, punctuation, written
  forms and capitalization included, and so are the speech start and end
  events. All 277 partial texts libsoda shows appear in opensoda's
  partials, and every word start is identical to libsoda's.
- Clean speech (122 s, 16 synthesized sentences): opensoda's transcript is
  character-for-character identical to libsoda's, punctuation and result
  boundaries included, and so is the transcript of the exact audio Chrome
  received during a capture (475 characters), compared with what Chrome's
  libsoda produced live. 261 of its 262 partial texts appear in opensoda's.
- A noisy, conversational lesson video (346 s): 14-17% character difference
  against four offline libsoda runs. Both transcripts are poor on this audio,
  and libsoda itself varies 5-13% between runs, because its encoder batches
  frames by how far it lags behind the audio, and its dynamic int8
  quantization makes the encoder output depend on the batching. Both end 21
  results, and the boundaries agree to the frame almost everywhere.

Given the same audio, opensoda's encoder input is bit-identical to
libsoda's (for en-US checked on a two-sentence recording, across the
restart, with the gain control in place), and given the same frame
batching, so is its encoder output. Its VAD posteriors are bit-identical
too. That takes libsoda's own arithmetic: TFLite built without fused
multiply-adds (`-ffp-contract=off`), the reference `BATCH_MATMUL` kernel,
TFLite's NEON kernels for 4-bit fully connected layers (its CMake build
leaves them out, and the scalar fallback rounds differently: with it, en-US
partials and word starts differed from libsoda's in places),
ruy for float matrix multiplies, the float `tanh` of the newer Eigen that
libsoda's TFLite was built with (the build patches it into TFLite's Eigen;
`cmake/eigen_ptanh_float.h`), and the VAD's normalization dividing by the
standard deviation. Given
libsoda's encoder outputs, the joint network's output is bit-identical too.
The search's smoothed label costs (a temperature softmax in Eigen) can still
differ from libsoda's by about one unit in the last place. Given libsoda's
encoder outputs for the lesson video, recorded from three libsoda runs,
opensoda's results match libsoda's word for word in 11 to 13 of the first 12
or 13 results (about 280 s); in each of the others, one word differs. After
that, libsoda's segmentation itself varies from run to run.

Language identification events are identical to libsoda's, in order,
language, confidence level and switch result, on ten recordings (8,000
events in all: en-US, ja-JP in both modes, a clip ending after a pause, and
the ja-JP → en-US → ja-JP switching recording). That rests on the
classifier's inputs and outputs being bit-identical to libsoda's, and on
libsoda's bookkeeping around them: the classifier state is reset every
`reset_every_n_frame` calls (not frames); the first call of a stream reports
nothing; a partial batch left at a long pause is dropped when speech resumes
but classified at the end of the audio (filled with copies of its last
frame), and without one libsoda repeats its last event there; and of the
level-3 events one chunk of audio produces, only the last is considered for
a language switch (the others report 0). So are those of the other packs'
recordings (another 4,800 events), apart from one each in id-ID and pt-BR at
a chunk boundary of libsoda's real-time run.

Contextual biasing was compared with libsoda on four en-US sentences with
rare names, biased toward five phrases (`Siobhan Kowalczyk`, `Zorblax`,
`Nguyen Thi Huong`, `Quillfeather`, `Xelvantra`). At boosts 1 and 3 every
final matches libsoda's (`Siobhan Kowalczyk`, `Zorblax`, the rest unchanged),
and at boost 3 so does every partial text. At boost 10 both over-trigger
(libsoda's first pass repeats `Nguyen Thi Huong` over silence; so does
opensoda's), and one of the four finals matches. The lowercased phrase
copies that the model's `TO_LOWER` transformer suggests would change the
results (`zorblax` instead of `Zorblax`), so opensoda doesn't add them.

### Not implemented

- Entity injection (`TaggedPathProcessingStreamJdqSoda`, en-US). It
  injects contact names from an `enhanced-voice-dictation-contacts`
  context, which libsoda's Chrome API cannot carry: the model needs a
  dynamic class (`$CONTACTS`), and Chrome's `RecognitionContext` only has
  phrases. Its `ContextInput` type in libsoda has phrase fields but no
  dynamic-class field, so a dynamic class sent there is dropped with no
  change in output. With phrases under that context name, libsoda logs
  that the model "disallows missing classes" and also leaves the output
  unchanged, so the stage only passes results through.
- Speaker change detection (config field 13). libsoda then sets up a
  diarization processor, which with Chrome's multi-language config fails
  ("Diarization cannot be enabled with multilang unless multilang is limited
  to 1 switch per session"); without that config it starts, and the
  responses carry nothing new. opensoda ignores the field.
- The `ONDEVICE_LARGE_SHORT` config's extra stages (voice-command cleanup
  and a neural language-model rescorer, `large-rank_candidates.tflite`).
  libsoda does not load this config for Chrome's captions or dictation; its
  log names `ONDEVICE_LARGE_CONTINUOUS` in both modes, for ja-JP and en-US.
- Repeated partials. libsoda sends a partial each time its decoder thread
  has decoded new frames, whether or not the text changed, so it repeats a
  text with a later end time (in en-US, about half again as many partials
  as distinct texts) and skips the states between two wake-ups. How many
  frames a wake-up covers depends on thread timing (1 to 10 frames in the
  runs above). opensoda sends a partial after each frame whose text
  changed, which includes every text libsoda can show.
- LAS rescoring (`large-rank_candidates_acoustic`, in the ja-JP, en-US,
  de-DE and fr-FR packs; the others have none). The configs
  rescore the 8-best (ja-JP: of segments under 15 s; en-US: of the second
  pass, segments under 60 s) with `new_lm_factor: 0` and
  `original_lm_factor: 1`, which leaves the ranking as it was.
  In the replays above, short segments match libsoda's word for word without it.
- Contextual biasing in packs without a twiddler model. ja-JP's configs
  have none, and libsoda's ja-JP output does not change with phrases either.
- Result prefetching. libsoda's DecoderEndpointerStream ends no segments
  in these configs (`enable_decoder_endpointing` and
  `enable_acoustic_endpointing` are false); its preliminary results only
  feed prefetching (`enable_result_prefetch`, `EagerPrefetchingConfig`),
  which Chrome does not turn on. None of the 3,900 results libsoda sent in
  the runs above is a prefetch result.

Dictation (IME) in en-US: each of ten dictated sentences recognized on its
own gives libsoda's final exactly, and so do all 16 finals of the three
recordings that dictate them together (including `Type the word, and then
the word period.`, where the second pass writes the command as a plain word
in that context); every partial text libsoda shows appears in opensoda's.

Dictation in ja-JP, on the clean 122 s recording: all 16 finals are
identical to libsoda's, and so are the speech start and end events; 273 of
libsoda's 274 partials appear in opensoda's. On 29 synthesized dictation
sentences with spoken punctuation commands, alone and inside sentences, all
25 finals and all 149 partials are identical to libsoda's, including
`Gmail。 Com`, where the punctuation model puts a period inside `Gmailcom`
and the capitalization after it applies inside the word too. 115 of their
121 word starts match; the others come from the encoder's batching (given
the outputs libsoda's encoder produced in one run, all 93 word starts of
that run's 18 finals match, `com` included). As in
libsoda, a ja-JP partial's raw text has no spaces (`テスト@GmailCO`), and a
final's none before a word that began at a bare word boundary piece
(`テスト@ Gmailcomです` from `▁テ スト ▁ @ ▁g ma i l ▁ co m`).

Captions in de-DE, on 18 synthesized German sentences in two recordings: the
speech start and end events are identical to libsoda's, and 12 of 13 finals,
raw text included; the eight sentences recognized one at a time all match.
The other final is a first-pass decoding difference of the same kind as
en-US's (`7 Euro` where libsoda's hypothesis has `7€` in that context; alone,
both give `7 €`).

Dictation in de-DE, on 31 synthesized sentences with spoken punctuation
commands (`komma`, `fragezeichen`, `neue zeile`, `klammer auf`, `hashtag`,
`bindestrich` and others): 26 or 27 finals are identical to libsoda's,
depending on libsoda's run. The others come from decoding: libsoda's own
result for some changes between runs with how its encoder thread batches
frames (`gut doppelpunktpunktpunktpunkt` or `gut! Doppelpunktpunktpunkt`,
`max@ zeichenbeispielpunkte` or `max@zeichenbeispielpunkte`), and in one
opensoda hears another word. Where the words agree, so do their start times,
except for a few words in dictation that start a frame or three earlier.

es-ES, it-IT, ru-RU, ko-KR, hi-IN, pt-BR and cmn-Hans-CN captions, on five
synthesized sentences each in one recording: every partial and every final
is identical to libsoda's, as are the speech start and end events (es-ES
needs the CAPTION cluster id fallback above for both; with the configured
id its VAD and its numbers differ). Word start times: 43 of 45 (es-ES), 39
of 42 (it-IT), 29 of 32 (ru-RU), 24 of 24 (ko-KR), 38 of 38 (hi-IN), 11 of
12 (pt-BR) and 33 of 33 (cmn-Hans-CN), the others a frame off; every word
confidence matches. pt-BR's and cmn-Hans-CN's are compared with libsoda
fed at real-time speed: libsoda's own result in those packs changes with
how fast it is fed, and between runs, with how its encoder thread happens
to batch frames (cmn-Hans-CN's fourth sentence comes out as `我 花 了 15
块 钱 买 苹 果`, `火 花 了 15 块 钱 买 两 斤 苹 果` or `火 花 钱 买。`).
pt-BR's libsoda drops three of the five results itself (no final for them);
opensoda does likewise.

id-ID, pl-PL, th-TH, tr-TR, vi-VN and cmn-Hant-TW captions, five
synthesized sentences each, against libsoda fed at real-time speed: every
partial and final is identical to libsoda's in id-ID (38 partials, 5
finals), pl-PL (43, 5), tr-TR (40, 5) and vi-VN (45, 5), cased names and
places included; th-TH differs in one of 45 partials (another libsoda run
gives opensoda's partials exactly), and cmn-Hant-TW in one final,
`晴又顏熱` for `晴朗又演熱`, which libsoda itself gives when its encoder
thread batches frames differently (as it does under a debugger). The speech
start and end events match too, as they do in every pack above. With the
recordings at 48 kHz, which libsoda resamples, ten of these thirteen packs
match every partial and final; in pt-BR, th-TH and cmn-Hant-TW the
differences are ones libsoda's own runs show. Word start times: 29 of 31 (id-ID),
34 of 34 (pl-PL), 26 of 29 (tr-TR) and 42 of 44 (vi-VN), the others a frame
off; every word confidence matches.

French, on 8 synthesized caption sentences: the speech start and end events
are identical to libsoda's, and so are all 86 partials and 8 finals of
libsoda fed at real-time speed (fed at 8x, libsoda itself hears `10 8
degrés` in one); 69 of 76 word starts match, the others a frame or two off.
These come from the encoder's batching: the encoder outputs are
bit-identical to libsoda's for the same batches, and given the outputs
libsoda's encoder produced in one run, opensoda's word starts match that
run's exactly (libsoda itself disagrees with itself on 3 of 76 between
runs). In dictation, 9 of 10 sentences with spoken
punctuation give libsoda's final text (`Est-ce que tu viens ce soir ?`,
`Voici la liste : du pain, du lait et des œuf.`); the other is a decoding
difference (`ouvrez` for libsoda's `ouvrir`).

### Speed and memory

CPU time and peak memory of `soda_api_client` driving each library on an
Apple silicon Mac, with the same audio, fed at real time as Chrome does:

| Recording | opensoda CPU | libsoda CPU | opensoda memory | libsoda memory |
|---|---|---|---|---|
| en-US, 55 s | 4.9 s | 6.4 s | 179 MB | 233 MB |
| ja-JP, 122 s | 14.9 s | 20.0 s | 241 MB | 268 MB |

Fed faster (opensoda as fast as it goes, libsoda at 8x, since it skips
audio it cannot keep up with), opensoda takes 3.2 s of CPU for en-US,
10.9 s for ja-JP and 1.7 s for 25 s of pl-PL, against libsoda's 4.4 s,
14.6 s and 2.4 s, and finishes the 122 s ja-JP recording in 11 s. Each
model runs on one thread: the encoder takes a few frames at a time, and a
second TFLite thread only spins waiting for work. The results are the same
either way. `tests/bench.py` makes these measurements.

## Testing

`tests/regress.py --check` runs 36 cases in all 17 packs (captions,
dictation, biasing, language switching) and compares every partial, final,
word timing and confidence, speech event and language identification event
with libsoda's; see [tests/README.md](tests/README.md). The recordings and
libsoda's responses stay local, in `tests/data/`. `tests/corpus.py` runs a
corpus of clips through opensoda or libsoda the way Chrome captions them,
one session per clip, and compares the runs.

## Building

You need CMake 3.20+ and a C++17 compiler. Check out these dependencies under
`third_party/`:

- `third_party/tensorflow`: TensorFlow v2.18.0. Only `tensorflow/lite` is built,
  with XNNPACK off. On arm64 the build adds TFLite's NEON kernels for 4-bit
  fully connected layers, which its CMake build leaves out, and replaces its
  CPU check with one that also finds the dot product instructions on macOS
  (`cmake/tflite_cpu_check.cc`); the checkout itself is not modified.
- `third_party/marisa-trie`: v0.3.1.

```sh
cmake -S . -B build -G Ninja
ninja -C build
```

This produces:

- `libsoda.dylib`: drop-in replacement for libsoda's C API.
- `soda_transcribe`: transcribes a WAV file.
- `soda_api_client`: drives any libsoda-compatible library (this one or
  Google's) through the C API, the same way Chrome does. `--speed X` paces the
  audio at X times real time (Google's libsoda keeps 30 s of audio and skips
  ahead when fed faster than it can process), `--dump-responses F` saves
  every raw response, `--multilang LOCALE=DIR` lists another installed pack
  for language switching, and `--phrase TEXT[:BOOST]` sends a biasing
  phrase (in the config, or with `UpdateRecognitionContext` after start
  under `--update-context`; `--context-name` changes the context it is
  sent under). `--chunk-ms N` sets the audio per `ExtendedAddAudio` call
  (default 100 ms). Given several clips (or `--clips LIST`), it loads the
  library once and gives each clip its own session, as Chrome does for
  each media stream, and `--jsonl F` writes each clip's finals.
- `soda_punctuate`: runs the punctuation model on tokens; `--scores` prints
  the per-token class probabilities.
- `soda_capitalize`: cases words with a pack's Roadrunner capitalization
  model (`denorm/SODA_capitalization_model.tflite`).

## Usage

```sh
# Without --pack, uses the newest pack Chrome installed for --language
# (default ja-JP).
build/soda_transcribe speech.wav
build/soda_transcribe --language en-US speech.wav
build/soda_transcribe --pack ".../SODALanguagePacks/ja-JP/1.5072.0" speech.wav
build/soda_transcribe --domain IME --partials speech.wav
```

Diagnostic options:

- `--no-vad` turns off VAD segmentation.
- `--dump-features OUT` writes the encoder input frames as raw float32.
- `--dump-vad OUT` writes the VAD posterior and endpointer state per frame.
- `--langid` prints language identification events.
- `--mask-offensive` masks offensive words, as `mask_offensive_words` does.

WAV input can be 16-bit PCM or float, at any rate, mono or interleaved.

## Integrating with MacPopup and other apps

Load `libsoda.dylib` with `dlopen` and use it exactly as Chrome uses
`libsoda.so`. See `tools/soda_api_client.cc` for a complete client.

1. Build a serialized `speech.soda.chrome.ExtendedSodaConfigMsg` with these
   fields:
   - `channel_count` (1)
   - `sample_rate` (2)
   - `api_key` (6): ignored by opensoda
   - `language_pack_directory` (7): the `SODAModels` path
   - `recognition_mode` (8): 2 = CAPTION, 1 = IME
   - `enable_formatting` (12): punctuation on or off
   - `mask_offensive_words` (16): masks offensive words with asterisks
   - `multilang_config` (15): `{1: {1: locale, 2: SODAModels path}, ...}`,
     one entry per installed pack, as Chrome sends it. It turns on language
     identification and switching between those packs
   - `recognition_context` (19): phrases to bias toward,
     `{1: {1: context name, 2: {1: {1: phrase, 2: boost (float)}, ...}}}`, as
     Chrome's Web Speech API sends them. `UpdateRecognitionContext` replaces
     them mid-stream. Only packs with a biasing model (en-US) use them, and
     only under that model's context name,
     `android-speech-api-generic-phrases` (libsoda ignores phrases under
     any other name)
2. Call `CreateExtendedSodaAsync({config, size, callback, handle})`, then
   `ExtendedSodaStart`.
3. Feed interleaved int16 PCM at the configured rate with `ExtendedAddAudio`.
   Chrome sends 10–20 ms chunks.
4. Call `ExtendedSodaMarkDone` at end of stream. An `END_OF_AUDIO`
   endpoint event and any last final are followed by `STOP`. Then call `DeleteExtendedSodaAsync`.

Callbacks receive serialized `SodaResponse` messages on a worker thread:

- `RECOGNITION` (1) carries a `SodaRecognitionResult`: `PARTIAL` (1) or
  `FINAL` (2), its hypotheses, timing and parts. Its `TimingMetrics` has
  only the result's start and end in the audio (fields 2 and 4), as in
  every result libsoda sent in the runs above.
- `ENDPOINT` (5) carries a `SodaEndpointEvent`: `START_OF_SPEECH`,
  `END_OF_SPEECH` and, at the end of the stream, `END_OF_AUDIO`, timed as
  described above.
- `LANGID` (7) carries a `SodaLangIdEvent` per classifier output (about 15
  per second of speech): the language (`ja-jp`), a confidence level from 0 to
  3, and the switch result. As in libsoda, these are sent when the config has
  `enable_lang_id` or a `multilang_config`, which Chrome always sends. After
  a successful switch, results come from the new language's pack.
- `START` (4), `STOP` (2) and `SHUTDOWN` (3) carry no result.

To show text the way Chrome does, display `hypothesis(0)`. For Japanese text
without spaces, use `hypothesis(1)` (raw), or remove the spaces between
characters of spaceless scripts.

A new session is needed when the sample rate or channel count changes, as in
Chrome.
