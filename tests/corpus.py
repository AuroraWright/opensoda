#!/usr/bin/env python3
"""Recognizes a corpus of clips the way Chrome captions them, and compares runs.

  tests/corpus.py run --clips LIST --out RUN.jsonl (--pack DIR | --locale LOCALE)
                      [--library LIB] [--speed X] [-j N] [--ime] [--chunk-ms N]
  tests/corpus.py compare A.jsonl B.jsonl [--references LIST] [--show N]

LIST has one audio file per line, optionally followed by a tab and the
clip's reference transcript. Each clip gets its own SODA session, as each
media stream does in Chrome: created with the clip's own sample rate and
channel count, fed in 20 ms chunks, marked done at the end and deleted
after STOP; the library stays loaded between clips. Files that are not
16-bit PCM WAV are converted with afconvert first, keeping their sample
rate and channels (Chrome hands SODA the decoded audio at its own rate).

`run` writes one JSON line per clip, in LIST order: the finals (formatted
text, the hypothesis Chrome's captions use; raw text; start and end), `text` and `raw`
(the finals joined), the number of partials and the wall-clock time. The
library defaults to build/libsoda.dylib, unpaced. Google's libsoda must be
paced: Chrome feeds it in real time (the default for any other library,
--speed 1); fed faster, its encoder batches frames differently and its
results change (and it skips audio when it falls 30 s behind). On 43
Japanese clips, libsoda at --speed 2 and 4 gave its real-time results, at
8 not. -j runs N processes, each over a contiguous share of the clips.

Google's libsoda sometimes never finishes a clip: when its search ends a
segment without a result ("No final state active" in its log), it loops
forever after ExtendedSodaMarkDone, at 100% CPU, and cannot be deleted.
Whether it happens depends on thread timing (more often fed faster than
real time). A clip not finished --clip-timeout seconds (default 30) after
its audio was fed is recorded with an "error", and the run goes on with the
next clip in a new process. `compare` counts such clips separately.

`compare` reports how many clips have identical finals, the character
error rate of B against A, and with --references each run's character and
word error rates against the transcripts (case, punctuation and spaces
removed for CER; NFKC-normalized; WER only where the references are
written with spaces). Clips a run failed on are left out of its scores. --show N prints N differing clips.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regress import PACK_ROOTS, find_pack  # noqa: E402

OURS = os.path.join(ROOT, "build", "libsoda.dylib")
CLIENT = os.path.join(ROOT, "build", "soda_api_client")


def read_list(path):
    """[(audio path, reference or None)], paths relative to the list's directory."""
    base = os.path.dirname(os.path.abspath(path))
    clips = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        audio, _, ref = line.partition("\t")
        clips.append((os.path.join(base, audio.strip()), ref if _ else None))
    return clips


def is_pcm16_wav(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def playable(path, cache):
    """path, or a 16-bit PCM WAV conversion of it in cache."""
    if is_pcm16_wav(path):
        return path
    st = os.stat(path)
    key = hashlib.sha1(("%s|%d|%d" % (os.path.abspath(path), st.st_size, st.st_mtime_ns)).encode()).hexdigest()
    out = os.path.join(cache, key + ".wav")
    if not os.path.exists(out):
        tmp = out + ".tmp.wav"
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", path, tmp], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.replace(tmp, out)
    return out


def run(args):
    if not os.path.exists(CLIENT):
        sys.exit("build soda_api_client first (cmake --build build)")
    pack = args.pack
    if not pack:
        pack, note = find_pack(args.locale, PACK_ROOTS + args.packs)
        if not pack:
            sys.exit(note)
        if note:
            print("using", note, file=sys.stderr)
    library = os.path.abspath(args.library)
    speed = args.speed if args.speed is not None else (0 if os.path.samefile(library, OURS) else 1)
    clips = read_list(args.clips)
    if not clips:
        sys.exit("no clips in %s" % args.clips)
    os.makedirs(args.cache, exist_ok=True)
    played = []
    for audio, _ in clips:
        try:
            played.append(playable(audio, args.cache))
        except (subprocess.CalledProcessError, OSError) as e:
            print("%s: cannot convert (%s)" % (audio, e), file=sys.stderr)
            played.append(audio)  # soda_api_client reports the error for it.
    jobs = max(1, min(args.jobs, len(clips)))
    shares = [(len(clips) * k // jobs, len(clips) * (k + 1) // jobs) for k in range(jobs)]
    records = [None] * len(clips)
    start = time.time()
    base = [CLIENT, library, pack, "--speed", str(speed), "--chunk-ms", str(args.chunk_ms),
            "--clip-timeout", str(args.clip_timeout)]
    if args.ime:
        base.append("--ime")

    def worker(k, lo, hi, tmp):
        # Runs soda_api_client over clips lo..hi, again after the clip it
        # stopped at when it times out (status 3) or crashes.
        lst, out, log = (os.path.join(tmp, "%d.%s" % (k, ext)) for ext in ("txt", "jsonl", "log"))
        while lo < hi:
            with open(lst, "w", encoding="utf-8") as f:
                f.writelines(p + "\n" for p in played[lo:hi])
            with open(log, "w") as err:
                rc = subprocess.run(base + ["--clips", lst, "--jsonl", out], stdout=subprocess.DEVNULL,
                                    stderr=err).returncode
            got = []
            if os.path.exists(out):
                for line in open(out, encoding="utf-8"):
                    try:
                        got.append(json.loads(line))
                    except ValueError:
                        break  # A line cut short by a crash.
            records[lo:lo + len(got)] = got
            lo += len(got)
            timed_out = rc == 3 and got and got[-1].get("error", "").startswith("timeout")
            if lo < hi and not timed_out:
                if rc in (0, 3):
                    # The client stopped without a record for clip lo.
                    rc = "exited with status %d" % rc
                else:
                    rc = "crashed (status %d)" % rc
                tail = open(log, errors="replace").read().strip().splitlines()[-1:]
                records[lo] = {"clip": played[lo], "error": "soda_api_client %s%s" % (
                    rc, ": " + tail[0] if tail else "")}
                lo += 1

    with tempfile.TemporaryDirectory() as tmp:
        threads = [threading.Thread(target=worker, args=(k, lo, hi, tmp))
                   for k, (lo, hi) in enumerate(shares)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(2)
            print("\r%d/%d clips, %.0f s" % (sum(r is not None for r in records), len(clips),
                                             time.time() - start), end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as f:
        for (audio, ref), rec in zip(clips, records):
            rec["clip"] = audio
            if ref is not None:
                rec["reference"] = ref
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    errors = sum("error" in r for r in records)
    audio_s = sum(r.get("duration_ms", 0) for r in records) / 1000
    print("%s: %d clips (%.0f s of audio), %d errors, %.0f s; library %s, speed %g" % (
        args.out, len(records), audio_s, errors, time.time() - start, library, speed), file=sys.stderr)
    return 1 if errors == len(records) else 0


def normalize_chars(s):
    s = unicodedata.normalize("NFKC", s).lower()
    return [c for c in s if not c.isspace() and unicodedata.category(c)[0] not in "PS"]


def normalize_words(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return s.split()


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(pairs, norm):
    """Total edits / total reference length, over (reference, hypothesis) pairs."""
    e = n = 0
    for ref, hyp in pairs:
        r = norm(ref)
        e += edits(r, norm(hyp))
        n += len(r)
    return 100.0 * e / max(n, 1)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def compare(args):
    a, b = load(args.a), load(args.b)
    if [r["clip"] for r in a] != [r["clip"] for r in b]:
        sys.exit("the runs are not over the same clips in the same order")
    ok = [(x, y) for x, y in zip(a, b) if "error" not in x and "error" not in y]
    same_finals = sum([f["text"] for f in x["finals"]] == [f["text"] for f in y["finals"]] for x, y in ok)
    same_raw = sum(x["raw"] == y["raw"] for x, y in ok)
    print("clips: %d, compared: %d" % (len(a), len(ok)))
    for name, run in (("A", a), ("B", b)):
        failed = [r for r in run if "error" in r]
        if failed:
            timeouts = sum(r["error"].startswith("timeout") for r in failed)
            print("  %s failed on %d (%d timed out), e.g. %s: %s" % (
                name, len(failed), timeouts, failed[0]["clip"], failed[0]["error"]))
    print("identical finals (formatted): %d/%d" % (same_finals, len(ok)))
    print("identical text (raw):         %d/%d" % (same_raw, len(ok)))
    print("CER of B against A:           %.2f%%" % error_rate([(x["text"], y["text"]) for x, y in ok], normalize_chars))
    refs = {}
    if args.references:
        refs = {os.path.abspath(p): r for p, r in read_list(args.references) if r is not None}
    else:
        refs = {r["clip"]: r["reference"] for r in a if "reference" in r}
    if refs:
        # Word error rates only for references written with spaces.
        spaced = sum(" " in r.strip() for r in refs.values()) * 2 >= len(refs)
        for name, run in (("A", a), ("B", b)):
            pairs = [(refs[r["clip"]], r["text"]) for r in run if r["clip"] in refs and "error" not in r]
            if not pairs:
                print("%s against references: no clips" % name)
                continue
            line = "%s against references (%d clips): CER %.2f%%" % (
                name, len(pairs), error_rate(pairs, normalize_chars))
            if spaced:
                line += "  WER %.2f%%" % error_rate(pairs, normalize_words)
            print(line)
    shown = 0
    for x, y in ok:
        if shown >= args.show:
            break
        if [f["text"] for f in x["finals"]] != [f["text"] for f in y["finals"]]:
            shown += 1
            print("\n%s\n  A: %s\n  B: %s" % (x["clip"], x["text"], y["text"]))
            if x["clip"] in refs:
                print("  reference: %s" % refs[x["clip"]])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="recognize every clip")
    r.add_argument("--clips", required=True, help="list: audio path [TAB reference] per line")
    r.add_argument("--out", required=True, help="JSON lines to write")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--pack", help="a SODAModels directory")
    g.add_argument("--locale", help="find the pack (re/packs, Chrome's directory)")
    r.add_argument("--packs", action="append", default=[], help="another SODALanguagePacks directory")
    r.add_argument("--library", default=OURS, help="libsoda-compatible library (default: opensoda)")
    r.add_argument("--speed", type=float, help="feed speed (default: 0 for opensoda, 1 otherwise)")
    r.add_argument("-j", "--jobs", type=int, default=1, help="parallel processes")
    r.add_argument("--ime", action="store_true", help="dictation instead of captions")
    r.add_argument("--chunk-ms", type=int, default=20, help="audio per AddAudio call (Chrome: 20)")
    r.add_argument("--clip-timeout", type=float, default=30,
                   help="seconds a clip may take after its audio is fed (then it is an error)")
    r.add_argument("--cache", default=os.path.join(ROOT, "build", "corpus-cache"),
                   help="where converted clips go")
    c = sub.add_parser("compare", help="compare two runs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--references", help="list with reference transcripts (default: from the runs)")
    c.add_argument("--show", type=int, default=0, help="print this many differing clips")
    args = ap.parse_args()
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
