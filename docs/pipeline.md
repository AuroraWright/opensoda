# How the pipeline works

opensoda builds its pipeline from the language pack's config, as libsoda does, so packs with different architectures
run from the same code. libsoda loads `ONDEVICE_LARGE_CONTINUOUS` for both captions and dictation (IME), or
`ONDEVICE_MEDIUM_CONTINUOUS` in packs that don't have it. The mode only selects the recognition domain.

This page describes each stage, the pack files it uses, and the details of libsoda's behaviour it reproduces. The
names in backticks are libsoda's own (config fields, classes in Google's `greco` speech library, or files in the
pack).

## Audio front end

Pack files: `*.mean_stddev`, the config. Code: `src/frontend.cc`.

The front end resamples the audio, applies automatic gain control where the config has one, and computes log-mel
filterbank frames, which it stacks and normalizes. It follows libsoda's arithmetic exactly:

- the 17-tap FIR resampler for 48 kHz input;
- en-US's gain control, greco's `SAMPLE_LEVEL_TARGET_RANGE` estimator, which restarts with the pipeline;
- the Hann window table, applied to the first 400 samples of each 401-sample frame, with HTK-style pre-emphasis of the
  first sample;
- the dither table and its phase;
- the split-radix FFT;
- a double-precision log, and division (not multiplication) by the standard deviation.

A frame stacker without `frame_zero_padding` fills its first frame's left context with copies of that frame.

On the same audio, the encoder input is bit-identical to libsoda's.

## Conformer encoder

Pack file: `endtoendmodel/large-encoder.tflite`. Code: `src/recognizer.cc`.

The encoder is streaming and is invoked on batches of frames (`time_batch_size`, and `time_batch_size_on_flush` at a
segment end). The encoder uses int8 dynamic quantization, so its output depends on how frames are batched:

- opensoda encodes frames in the smallest batch, as libsoda does when it keeps up with live audio.
- libsoda's encoder thread encodes the largest batch that fits what is queued, so its output varies with how far it
  lags behind the audio.
- Where the segmenter gates 10 ms frames (ko-KR, hi-IN, pt-BR, cmn-Hans-CN) and releases a backlog at a segment start,
  libsoda (logged at real-time speed) encodes the smallest batch from it, then the largest that fits the rest.
  opensoda does the same.

en-US's encoder also takes the recognition domain's id (`domain_id_mapping`).

## Cascaded second pass (en-US)

Pack files: `large-cascaded_encoder`, `large-decoder_1`, `large-joint_posterior_1`.

A non-causal encoder runs over the first encoder's output, with its own decoder and search settings. It gives the
finals; the first pass gives the partials.

- As in libsoda's `EncoderInsertRightContextStream` and `EncoderUnshiftStream`, its output for a frame comes 15 frames
  later and takes that frame's time. At a segment end, 15 copies of the last frame let the last frames through.
- Its internal-LM settings (`internal_lm_cost_scale` 0.26, `subtract_internal_lm` false) leave the costs unchanged in
  greco's `SmoothedCosts`, so `large-joint_prior_1` isn't needed.
- The second-pass model writes numbers, times, prices, phone numbers and addresses in written form itself (`7:00`,
  `$25.50`, `555-123-4567`).

## Decoder and search

Pack files: `large-decoder`, `large-joint_posterior`. Code: `src/transducer.cc`.

The search is greco's breadth-first transducer search over the pack's `SigmaFst` graph, which accepts any wordpiece
sequence. It applies temperature smoothing, `</S>` handling and `eos_penalty`, and from the search config:

- histogram pruning to about `max_arcs` hypotheses (3), and at least `min_arcs`;
- the `local_beam`;
- the word penalty;
- the label-cost limit (`max_model_cost`);
- `max_expansions`.

Given the same encoder outputs, it produces libsoda's transcript. The pack's internal-LM weight is 0, so
`large-joint_prior` isn't needed.

CONTINUOUS configs set `enable_resume_search`. A final that doesn't end a segment (one the model ends with `</S>`, or
any final when the VAD is off) then lets the search go on with the previous decoder context. If that context ends in
the config's `ignored_resume_context_suffix`, the symbol is dropped from it first. After a segment end the search
starts afresh (see the restart below).

## Recognition domain

Config: `domain_name_to_internal_id`.

The domain sets the acoustic cluster id fed to the encoder and the VAD. Each `AppendClusterIdStream` looks the domain
up in its own map. As in libsoda, a domain the map lacks gets its enum value: FARFIELD 1, IME 2, TELEPHONY 3,
CAPTION 4, VOICE_SEARCH 7. For example, en-US's and es-ES's VAD maps have no CAPTION entry, and libsoda feeds them 4.
The configs' `forced_cluster_id` is only a default.

## Voice activity detection and endpointing

Pack files: `acousticmodel/*.endpointer_portable_lstm_model`. Code: `src/endpointer.cc`, `src/vad_frontend.cc`.

The VAD models are TFLite files XOR-ed with a one-byte key. ko-KR's and hi-IN's use a `Relu1` custom op (a clamp to
[0, 1]).

**VAD input.** Most packs' VAD reads the stacked filterbanks with a one-hot cluster id every 30 ms. ko-KR's, hi-IN's
and pl-PL's reads 40 log-mel energies every 10 ms, computed from a noise-subtracted spectrum (pl-PL's VAD windows the
audio itself, without the main path's dither):

- greco's `SpectralNoise` tracks per-bin percentiles (95th and 25th) of all FFT energies so far. Its sorted history
  is halved once it holds 199 values and then sampled every other frame. It holds back the first 20 frames until it
  has an estimate.
- `SpectralSubtractor` then subtracts the 25th-percentile noise magnitude, scaled by the noise level and the
  peak-to-floor ratio, and floored at 0.2 of the magnitude.

In those packs the segmenter also gates 10 ms frames, before they are subsampled to 30 ms. The count restarts with each
segment.

**Endpointer.** greco's frame-counting endpointer is ported, including its delayed output, which pads each utterance
by `endpoint_margin`. As a result, pauses under about 0.6 s stay in one segment.

**Segmenter.** As in libsoda's graph, the segmenter sits before the encoder. Frames wait for the endpointer's decision,
and only speech frames are encoded. A result ends at the last encoder frame boundary before the segment end, which is
when Chrome gets END_OF_SPEECH.

**Restart.** Then, like libsoda, recognition restarts at that point. The front end, VAD, segmenter and encoder start
afresh and re-process the audio since then from a 30 s history. The next result starts there, or at its first word if
the endpointer kept a silence. The search also starts afresh, which matches libsoda's output better than keeping its
decoder context.

Segments are cut at `impatience_limit_ms` (60 s). `impatience_start_ms` only shortens the trailing silence a segment
needs, which is already zero in these configs.

## Punctuation

Pack files: `SODA_punctuation_model.tflite`, `SODA_punctuation_config.pb`. Code: `src/punctuator.cc`.

- **ja-JP:** a pQRNN model, with its seq_flow_lite custom ops reimplemented. Tokens are characters, the mark is the
  argmax per token, and partials get no trailing mark.
- **de-DE:** a pQRNN model that segments by whitespace (the config's `segment_by`). Its tokens are words, without any
  marks they start or end with, and a mark follows its word unless the word already ends in one of the config's
  marks. Its output tensors carry a `pqrnn_model/pqrnn_encoder/` prefix.
- **en-US:** greco's Roadrunner tagger, with its `CharNGramFeatures` op reimplemented (hashed character n-grams using
  libsoda's Fingerprint2011 hash). It takes one lowercased string per word plus an empty one, in windows of 30 words
  that overlap by 2 and 5. Each word's label names the mark before it, from the config's codebook.

**Capitalization.** The first token is capitalized, and so is each token after one that ends in an end-of-sentence
mark. libsoda sets every pack's capitalization behaviour to `CAPITALIZE_ALL`, whatever the config says (ja-JP's has
none).

- A word is capitalized at its first letter or digit, as libsoda's casing does (`#urlaub` → `#Urlaub`, `(1)`
  unchanged).
- Where tokens are characters (ja-JP, cmn-*), a mark inside a word capitalizes the next character
  (`シャンTfaceb` → `シャンT。Faceb`).

## Truecasing

Pack files: `denorm/transducer*.fst`, `denorm/lm*.sorted.fst`. Code: `src/truecaser.cc`.

This is greco's `GeneralCapitalizationNormalizer`. It generates case variants per word and picks them with a Viterbi
search under the OpenGrm n-gram LM (for example `sns` → `SNS`).

- de-DE's transducer brackets the sentence with `<S>` … `</S>`. The end is scored with the LM's `</S>` arc. libsoda
  scores the first word from the LM's unigram state, not after `<S>`, so `herr Schmidt` becomes `Herr Schmidt`
  while `ich` and `das` stay lowercase, as in libsoda's raw text.
- Backoff arcs count as plain epsilons. A word costs the cheapest of its arcs at every backoff level, not only at the
  first level that has it, and the search keeps each state reached that way.
- A word is looked up lowercased and without the punctuation it starts or ends with (`Viele,` is scored as `viele`).
- As in libsoda, words the transducer's symbol table lacks, and words that are only punctuation, are taken out before
  the search and put back unchanged. They give no context: in fr-FR, `ouvrez les guillemets bonjour` keeps `bonjour`,
  which after `<UNK>` would become `Bonjour`.

**id-ID, pl-PL, tr-TR and vi-VN** use a different truecaser: `denorm/SODA_capitalization_model.tflite`, configured as
`RuntimeRoadrunnerCapitalizationNormalizer` (code: `src/roadrunner_capitalizer.cc`). It's a TFLite
text-normalization model that reads and writes a serialized `speech.textnorm.Sentence`:

- A character n-gram tagger labels each word "keep" or "rewrite", and a character seq2seq beam search rewrites (that
  is, capitalizes) the words labelled "rewrite".
- Its five custom ops are reimplemented. They move the words between the proto and tensors:
  `SentenceToInputContext`, `RetokenizeSentence`, `SentenceToTokenTensors`, `AttachHypothesisToSentence` and
  `TokenPostProcess` (which puts back an input character the model wrote as `<unk>`). On libsoda's own model inputs,
  the outputs match byte for byte.
- As in libsoda, words are cased in windows of 5 that overlap by one word on each side.

## Denormalizers

Pack files: `denorm/*.mfar`; config `common_denormalizers_list`. Code: `src/fst_normalizer.cc`.

The config's `FstNormalizer` and `ClassNormalizer` grammars are memory-mapped FST archives. They are applied in list
order to the words joined by spaces, before punctuation. Those listed before the capitalization normalizer run before
truecasing.

- An archive's FSTs apply one after another, as in fr-FR's emoji grammar: insert markers, rewrite the marked words,
  delete the markers.
- Arcs that read no byte insert their output, as in `5 p.m` → `5 p.m.`.
- The `ClassNormalizer` (`embedded_class_denorm`) only rewrites spans the model tags with one of greco's 30 word
  classes, tags included (`<money> sieben euro </money>` → `7 €`, `<time> zwölf uhr </time>` → `12 Uhr`). If it
  rejects a span, or there's a tag of another class inside it or a missing closing tag, only the tags are dropped.

What the grammars do depends on the pack:

- **ja-JP:** decoration removal and pronunciation cleanup.
- **en-US:** COVID-19, a.m./p.m. fixes, the class denormalizer and the general normalizer always run. As in libsoda,
  the spoken-emoji and spoken-punctuation grammars (types 4 and 3 in the list) run only in IME mode. In captions
  `smiley face` becomes `:-)`; in IME `comma` becomes `,` and `new paragraph` a paragraph break.

Like libsoda's formatting:

- Closing marks join the word before, and opening brackets the word after.
- A line or paragraph break is dropped and the next word capitalized. In a final the spaces around it go too
  (`file {new-paragraph} and` → `fileAnd`); in a partial one space is left (`file And`).
- The punctuation model adds no mark after a word that already ends in one of its codebook's marks.
- The punctuation model doesn't see the dictated marks a word carries: it sees `hello` for `hello,`, but `p.m.` whole.

## Spoken punctuation in dictation

### ja-JP

Pack files: `denorm/PUNCTUATION_LSTM.model.int8.tflite`, `.syms`, `punctuation_converter_config.pb`. Code:
`src/spoken_punctuation.cc`.

This is greco's `PunctuationNormalizer`, applied in dictation only, after the other denormalizers.

- The words are joined without spaces, which Japanese doesn't use. The text becomes one token if the model's
  vocabulary has it, otherwise one token per character.
- A bidirectional LSTM, starting from a zero state for each result, flags the tokens that are commands.
- A flagged run that spells one of the converter config's spoken forms becomes that form's mark, as a word of its own,
  timed over the characters it replaces. Paired brackets and quotes alternate, opening first.

libsoda's segmenter at this step never matches a non-ASCII byte. So in ja-JP a command converts only when it is the
whole result: `当店` alone becomes `、`, `ハイフン` becomes `-` and then `-。`, and `新しい段落` becomes a paragraph
break. A result made only of line breaks isn't sent at all, as in libsoda.

### de-DE and fr-FR

Pack file: `denorm/model.tflite`; config `PortablePqrnnSpokenPunctuationNormalizerParams`. Code:
`src/pqrnn_spoken_punctuation.cc`.

This is greco's `PortablePqrnnSpokenPunctuationNormalizer`, applied in dictation only.

- The words are split at the punctuation they contain (`12,50` → `12` `,` `50`).
- A pQRNN model labels each piece as a command or not. A run of command pieces that spells one of the settings' spoken
  forms (longest match first, no backtracking) becomes its mark.
- The mark's type decides where it goes: closing marks join the word before (`anna komma` → `anna,`), opening ones the
  word after (`hashtag urlaub` → `#urlaub`), joining ones both (`post bindestrich west` → `post-west`), and line
  breaks stand alone.
- A spoken sentence end capitalizes the next word, and then the first word is capitalized.
- The pieces of one recognized word are joined back. de-DE's postnormalizers replace typographic quotes with straight
  ones. Each resulting word takes the times of the words it came from.

fr-FR's marks `? ! : ; « »` are set apart by a space. libsoda's results then list each space-separated part as a word
(`liste :` gives `liste` and `:`), except for the marks the punctuation model itself writes (`soir ?` stays one
word). It times a part by its byte position in the word's frames, rounded to a frame.

## Offensive-word masking

Pack file: `denorm/porn_normalizer_ondevice.mfar`.

Off by default; `mask_offensive_words` turns it on. The config's `PORN_DENORMALIZER` grammar is a byte-level
transducer in greco's memory-mapped FST archive format. It's applied like libsoda's `FstNormalizer`, as the cheapest
path over the punctuated words joined by spaces (`ちんぽ` → `ち**`, `shit` → `s***`). It masks finals and partials;
the raw (unformatted) text isn't masked.

## Language identification

Pack files: `configs/LANGUAGE_IDENTIFICATION.config`, `langid/*`,
`acousticmodel/SODA_DICTATION_EP_UNIFIED_FRONTEND_LANGID.*`. Code: `src/langid.cc`.

Language identification is a separate pipeline: greco's automatic gain control, 32 ms log-mel frames, its own VAD and
segmenter, and then a streaming classifier (79 outputs, 47 mapped languages). Each output gives the top language and
a confidence level from greco's scoring, which compares the probability margin and the negentropy with two thresholds.

Matching libsoda's events also takes its bookkeeping around the classifier:

- The classifier state is reset every `reset_every_n_frame` calls (calls, not frames).
- The first call of a stream reports nothing.
- A partial batch left at a long pause is dropped when speech resumes. At the end of the audio it's classified instead,
  filled with copies of its last frame; if there's no partial batch, libsoda repeats its last event there.
- Of the level-3 events that one chunk of audio produces, only the last is considered for a language switch. The
  others report 0.

## Language switching

Config: `multilang_config`.

As in libsoda, the first highly confident (level 3) identification of another language whose pack the config lists
switches to that pack (`SWITCH_SUCCEEDED`). A language without a pack is reported as `SWITCH_SKIPPED_NO_LP`.

The language identifier has one class for several locales, and libsoda's table of them (`es-us` for es-ES,
`cmn-hans-cn` for cmn-Hant-TW, `en-us` for en-GB, ...) decides both cases. For example, `es-us` in an es-ES session is
the current language, and `es-us` switches to an es-ES pack.

The new pack's recognizer takes over the audio history and the language identifier, and recognizes again from the
last final (`rewind_when_switching_language`). So the sentence that triggered the switch is recognized in the new
language.

## Contextual biasing (en-US)

Pack file: `endtoendmodel/large.wpm.portable`; config `class_substitution_twiddler`. Code: `src/biasing.cc`.

Phrases come from `recognition_context` or from `UpdateRecognitionContext`. They're split into word pieces with the
pack's byte-pair word-piece model (piece ids give the merge order) and built into a prefix trie with failure links,
as libsoda's FST twiddler does. Both passes consult it:

- Each matching piece lowers the cost by the phrase's boost plus `grammar_token_base_weight`. This is libsoda's token
  generator with `BOOST_STRATEGY_PER_TOKEN_CONSTANT`, which spreads the boost uniformly over the pieces.
- A partial match that breaks off gives its reward back; a completed phrase keeps it. Going on after a completed
  phrase costs `retwiddle_penalty`.
- As in greco's `DualSearchParams`, the label-cost limit and the pruning use the cheaper of the biased and unbiased
  costs (`use_min_of_original_and_influenced_cost_for_pruning`).
- The second pass biases no label that costs more than `max_model_cost_to_influence`.

The model's `TO_LOWER` transformer suggests adding lowercased copies of the phrases. opensoda doesn't add them: they
would change the results (`zorblax` instead of `Zorblax`).

## Keeping up with live audio

libsoda keeps captions close to live audio rather than falling further and further behind. opensoda does the same:

- Each final starts a new recognition session.
- A session that finds more than 1 s of received but unrecognized audio skips all but the last second of it. With a
  `multilang_config` the limit is 5 s (libsoda's `max_restream_duration_for_multilang_ms`).

Fed in real time, recognition is never that far behind. Fed a file as fast as possible, both libraries recognize the
first sentence and the last seconds, and skip at the same points. libsoda still sends the speech events for skipped
audio (its endpointer runs ahead of its recognizer); opensoda's restart at the skip.

With `simulate_realtime_testonly` (config field 5), libsoda reads the audio at real-time speed, and opensoda
recognizes all of it as fast as it can.

## Results without a multilang config

Chrome always sends a `multilang_config`, and then each final is sent whole. Without one (with formatting on and a
punctuation model), libsoda's text formatter holds back the unfinished last sentence of each final. opensoda does the
same:

- A final's words are punctuated together with the words held back, as for a partial (no mark after the last one).
  They're sent up to the last word the punctuation model ended a sentence after. For example, `5:00 p.m. Tomorrow`
  is sent where the model predicted a period, but not `8:00 a.m. And`.
- The rest leads the next partials and the next final.
- Punctuated together with a partial's words, held words may end a sentence after all. Those up to the last such
  sentence end are then sent as a final, which ends where the next word starts (`I did.`, once
  `we have a table for four at 7:30 great` follows).
- Held words are sent on their own, punctuated as a final, when 3 s pass after the final without a partial, and at the
  end of the audio. libsoda times this on the wall clock, opensoda in audio time.
- Such a final starts at its first word and ends where its last one does: where the next word starts, or, for the last
  word of a recognized segment, at the start of the frame of its last label.

## Speaker change detection (en-US)

Pack files: `configs/ONDEVICE_SPEAKER_TURN_DETECTION.config`, `diarization/SODA_SPEAKER_TURN.*`. Code:
`src/speaker_turn.cc`.

As in libsoda, this is a pipeline of its own:

1. Gain control, then 32 ms log-mel frames (128 channels), stacked four at a time every third frame, and normalized.
2. A streaming encoder fed three frames per call, every frame included. Its first four outputs are dropped, and at
   the end it gets four copies of the last frame.
3. A character transducer with an LSTM prediction network, searched like the ASR's. Its transcript marks a change of
   speaker with `#`.
4. Its own VAD and segmenter end the search's segments after a second of silence (less in segments over 15 s; segments
   are cut at 30 s). The search resumes with its context.

greco's `SpeakerTurnStream` makes each `#` a turn, timed at its frame, with up to 10 characters of transcript on each
side. A turn in a segment of only `mhm`, `hmm`, `hm` or `um` is dropped, together with the next one.

Turns after the last final's end are then placed among the words the formatter has (a final's, those it holds back
and a partial's), as libsoda's diarization does:

- at the word that starts closest to the turn, or at a neighbour whose surrounding text better matches the turn's
  context (by edit distance);
- then moved to the start of the sentence it's next to.

The formatted text puts `\n – ` before each marked word, and that word's part has field 3 set.

With speaker change detection on, libsoda also cuts ASR segments at 30 s instead of 60 (15 s before the trailing
silence they need shortens), and ends a result sent whole at its last word.

libsoda only runs speaker change detection when the config lists no `multilang_config` languages. Chrome always lists
them, so Chrome never gets it.

## Not implemented

### Features used outside Chrome

These features work in libsoda, but only for requests Chrome never makes. They are most likely used in Android:

- **Entity injection** (`TaggedPathProcessingStreamJdqSoda`, en-US). It injects contact names from an
  `enhanced-voice-dictation-contacts` context, which libsoda's Chrome API can't carry. The model needs a dynamic class
  (`$CONTACTS`), but Chrome's `RecognitionContext` only has phrases. libsoda's `ContextInput` type has phrase fields
  but no dynamic-class field, so a dynamic class sent there is dropped with no change in output. With phrases under
  that context name, libsoda logs that the model "disallows missing classes" and also leaves the output unchanged. So
  the stage only passes results through.
- **`ONDEVICE_LARGE_SHORT` and the other one-shot configs.** libsoda builds the config name from the request's
  application domain: the one-shot domains get `SHORT`, the others `CONTINUOUS`. Chrome sends no domain, so libsoda
  loads `ONDEVICE_LARGE_CONTINUOUS` in both modes, for ja-JP and en-US alike (its log names the config). The SHORT
  configs' extra stages (an end-of-query endpointer, voice-command cleanup, and a neural language-model rescorer,
  `large-rank_candidates.tflite`) are for voice queries. `soda_transcribe --config` can still select those configs,
  without their extra stages.
- **Result prefetching.** libsoda's `DecoderEndpointerStream` ends no segments in these configs
  (`enable_decoder_endpointing` and `enable_acoustic_endpointing` are false). Its preliminary results only feed
  prefetching (`enable_result_prefetch`, `EagerPrefetchingConfig`), which Chrome doesn't turn on. None of the 3,900
  results libsoda sent in the test runs was a prefetch result.

### Not reproduced on purpose

- **Google's DRM.** Google's libsoda only works inside Chrome. Loaded by any other application, it silences the audio
  it's given, so it never returns a result. opensoda doesn't reproduce this (congrats Google for DRMing assistive
  technologies!).

### Differences with no effect

These parts of libsoda don't change the results for any request in these packs:

- **LAS rescoring** (`large-rank_candidates_acoustic`, in the ja-JP, en-US, de-DE and fr-FR packs). The configs rescore
  the 8-best (ja-JP: of segments under 15 s; en-US: of the second pass, segments under 60 s) with `new_lm_factor: 0`
  and `original_lm_factor: 1`. libsoda then skips the rescore, and the ranking stays as it was. In the replays, short
  segments match libsoda's word for word without it.
- **Contextual biasing in packs without a biasing model.** Only en-US's configs have one. ja-JP's output in libsoda
  doesn't change with phrases either.
- **Repeated partials.** libsoda sends a partial each time its decoder thread has decoded new frames, whether or not
  the text changed. So it repeats a text with a later end time (in en-US, about half again as many partials as
  distinct texts), and skips the states between two wake-ups. How many frames a wake-up covers depends on thread
  timing (1 to 10 frames in the test runs). opensoda sends a partial after each frame whose text changed, which
  includes every text libsoda can show.
