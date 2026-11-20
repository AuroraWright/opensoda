# Tests

The tests compare opensoda's responses with Google's libsoda's on the same
audio. Neither the audio nor libsoda's responses are part of the project:
they live in `tests/data/` (git-ignored), and the language packs in
`re/packs/` (git-ignored) or Chrome's own directory.

## Running

```sh
cmake --build build
tests/regress.py --check      # all cases; fails if any score dropped
tests/regress.py en ja_ime    # some cases
tests/regress.py --update     # accept the current scores
```

Each case runs `build/soda_api_client` with `build/libsoda.dylib` on a
recording and scores the responses against libsoda's (see `cases.py`):

| Score | |
|---|---|
| `partials` | libsoda's distinct partial texts that opensoda also sends (libsoda sends partials as its decoder thread wakes up, so it skips some and repeats others) |
| `finals` | finals with libsoda's raw text, in order |
| `formatted` | the same for the formatted text |
| `starts` | word start times equal to libsoda's, in finals with the same words |
| `conf` | word confidences equal to libsoda's, likewise |
| `endpoints` | speech start and end events, in order (audio times) |
| `langid` | language identification events (language, confidence, switch result), in order |

`tests/expected.txt` holds the current scores. Where they fall short of
libsoda's totals, the difference is one libsoda's own runs show: its
encoder batches frames as its threads keep up, so its output changes with
how fast it is fed and from run to run (`fr_run2` and `de_speed8`/`fr_speed8`
compare the same recordings with other libsoda runs). `--compare A B`
scores any two dumps, e.g. two libsoda runs.

`tests/bench.py` measures CPU time and peak memory, offline or with
`--realtime`, and with `--reference LIBRARY` also another library's.

## Corpora

`tests/corpus.py` recognizes a list of clips the way Chrome captions them,
one SODA session per clip, with opensoda or another libsoda-compatible
library, and compares two runs (and each against reference transcripts):

```sh
# clips.txt: one audio file per line, optionally TAB and the transcript
tests/corpus.py run --clips clips.txt --locale ja-JP --out ours.jsonl -j 8
tests/corpus.py run --clips clips.txt --locale ja-JP --out libsoda.jsonl -j 4 \
    --library PATH/TO/libsoda --speed 4
tests/corpus.py compare libsoda.jsonl ours.jsonl --show 10
```

Each output line holds a clip's finals (formatted text, the hypothesis Chrome's
captions use, and raw text, with start and end times) and their concatenation. Chrome
feeds libsoda in real time, which `run` does by default for any library
but opensoda (whose results do not depend on the feed speed). The reference
library has to be one that works outside Chrome: Chrome's own copy
of libsoda sends no results there.

Google's libsoda sometimes never finishes a clip. When its search ends a
segment without a result (`No final state active` and `Result lattice is
not set` in its log), the loop that decodes the remaining audio after
`ExtendedSodaMarkDone` never advances and runs forever at full CPU, and the
session cannot be deleted. It depends on thread timing; fed faster than
real time it happens more often. `run` records a clip that is not finished
`--clip-timeout` seconds (default 30) after its audio as an error and goes
on with the next one; `compare` lists the failures and leaves them out.

## Data

`tests/data/<recording>/` holds `audio.wav` and one dump of libsoda's
responses per reference (`caption.bin`, `ime.bin`, ...); `sents.txt`, where
present, lists the sentences it speaks. Most recordings were synthesized
with `tests/make_recording.sh`; `ja_lesson` is a 122 s lesson recording, and
its caption reference was captured from Chrome.

To add a case: make a recording, add the case to `cases.py`, and record
libsoda's responses:

```sh
tests/make_recording.sh it2 Alice "Buongiorno a tutti." "Che ore sono?"
tests/record.py --library PATH/TO/libsoda.so it2
```

`record.py` feeds libsoda at real-time speed by default, as Chrome does.

## Packs

The tests look for `SODALanguagePacks/<locale>/<version>/SODAModels` in
`re/packs/` and in Chrome's directory, preferring the version the
references were recorded with (`cases.py`), and note when they use another
one. Chrome downloads only the packs for the languages its user picks;
`tests/fetch_pack.py LOCALE...` (or `all`) downloads others from Chrome's
update service into `re/packs/`. Packs are Google's: never commit or
redistribute them.
