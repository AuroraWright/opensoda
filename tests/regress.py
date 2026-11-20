#!/usr/bin/env python3
"""Runs the regression cases and compares the responses with libsoda's.

  tests/regress.py [-j N] [--check | --update] [CASE...]

Each case runs build/soda_api_client on its recording with build/libsoda.dylib
(as fast as it goes) and scores the responses against the case's reference:

  partials   libsoda's distinct partial texts that opensoda also sends
  finals     finals with the same text, in order
  formatted  the same for the formatted text (punctuation, casing)
  starts     word start times equal to libsoda's, in finals with the same words
  conf       word confidences equal to libsoda's, likewise
  endpoints  speech start and end events, in order
  langid     language identification events, in order

--check compares the scores with tests/expected.txt and fails if any is
lower; --update rewrites it. --compare A B scores two dumps against each
other (e.g. two libsoda runs).

Packs are looked up as SODALanguagePacks/<locale>/<version>/SODAModels in
re/packs, Chrome's directory and any --packs directory.
"""

import argparse
import concurrent.futures
import difflib
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soda_dump  # noqa: E402
from cases import CASES, PACK_VERSIONS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
EXPECTED = os.path.join(ROOT, "tests", "expected.txt")
PACK_ROOTS = [
    os.path.join(ROOT, "re", "packs", "SODALanguagePacks"),
    os.path.expanduser("~/Library/Application Support/Google/Chrome/SODALanguagePacks"),
]


def find_pack(locale, roots):
    """The pack directory for locale (preferring the references' version),
    and a note if it is another version."""
    want = PACK_VERSIONS.get(locale)
    found = []
    for root in roots:
        d = os.path.join(root, locale)
        if not os.path.isdir(d):
            continue
        for v in os.listdir(d):
            models = os.path.join(d, v, "SODAModels")
            if os.path.isdir(models):
                if v == want:
                    return models, ""
                found.append((tuple(int(x) for x in re.findall(r"\d+", v)), v, models))
    if not found:
        return None, "no %s pack" % locale
    _, v, models = max(found)
    return models, "%s %s, references from %s" % (locale, v, want)


def score(ref, ours):
    """Scores two dumps; returns [(label, matching, total)]."""
    rp = {r.raw for r in ref.partials}
    sp = {r.raw for r in ours.partials}
    rf, sf = ref.finals, ours.finals
    pairs = list(zip(rf, sf))
    starts = conf = words = 0
    for a, b in pairs:
        if [w.text for w in a.words] != [w.text for w in b.words]:
            continue
        for x, y in zip(a.words, b.words):
            words += 1
            starts += x.start_ms == y.start_ms
            conf += x.confidence == y.confidence

    def in_order(a, b):
        m = difflib.SequenceMatcher(None, a, b, autojunk=False)
        return sum(blk.size for blk in m.get_matching_blocks())

    return [
        ("partials", len(rp & sp), len(rp)),
        ("finals", sum(a.raw == b.raw for a, b in pairs), len(rf)),
        ("formatted", sum(a.text == b.text for a, b in pairs), len(rf)),
        ("starts", starts, words),
        ("conf", conf, words),
        ("endpoints", in_order(ref.endpoints, ours.endpoints), len(ref.endpoints)),
        ("langid", in_order(ref.langid, ours.langid), len(ref.langid)),
    ]


def format_scores(name, scores):
    return "%-10s %s" % (name, "  ".join("%s %d/%d" % s for s in scores))


def parse_scores(line):
    name, rest = line.split(None, 1)
    return name, {k: (int(a), int(b)) for k, a, b in re.findall(r"(\w+) (\d+)/(\d+)", rest)}


def run_case(args, case, out_dir):
    name, recording, reference, locale, options = case
    roots = PACK_ROOTS + args.packs
    pack, note = find_pack(locale, roots)
    ref_path = os.path.join(DATA, recording, reference + ".bin")
    wav = os.path.join(DATA, recording, "audio.wav")
    if not pack or not os.path.exists(ref_path) or not os.path.exists(wav):
        return name, None, note or "no recording or reference in tests/data/%s" % recording
    opts = []
    for o in options:
        if "@" in o:
            prefix, other = o.split("@", 1)
            other_pack, other_note = find_pack(other, roots)
            if not other_pack:
                return name, None, other_note
            o = prefix + other_pack
        opts.append(o)
    out = os.path.join(out_dir, name + ".bin")
    cmd = [os.path.join(ROOT, "build", "soda_api_client"), args.library, pack, wav, *opts,
           "--speed", str(args.speed), "--dump-responses", out]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        return name, None, "soda_api_client failed (%d): %s" % (p.returncode, p.stderr[-500:])
    return name, score(soda_dump.read(ref_path), soda_dump.read(out)), note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--library", default=os.path.join(ROOT, "build", "libsoda.dylib"))
    ap.add_argument("--speed", type=float, default=0, help="soda_api_client --speed (0: no pacing)")
    ap.add_argument("--packs", action="append", default=[], help="another SODALanguagePacks directory")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "test-output"),
                    help="where to write opensoda's responses")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if a score is below tests/expected.txt")
    mode.add_argument("--update", action="store_true", help="rewrite tests/expected.txt")
    mode.add_argument("--compare", nargs=2, metavar=("REF", "OTHER"), help="score two dumps")
    args = ap.parse_args()

    if args.compare:
        print(format_scores("compare", score(*(soda_dump.read(p) for p in args.compare))))
        return 0

    names = {c[0] for c in CASES}
    unknown = [n for n in args.cases if n not in names]
    if unknown:
        ap.error("unknown cases: %s (known: %s)" % (", ".join(unknown), " ".join(c[0] for c in CASES)))
    cases = [c for c in CASES if not args.cases or c[0] in args.cases]
    os.makedirs(args.out, exist_ok=True)

    expected = {}
    if os.path.exists(EXPECTED):
        for line in open(EXPECTED):
            if line.strip() and not line.startswith("#"):
                n, s = parse_scores(line)
                expected[n] = s

    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_case, args, c, args.out) for c in cases]
        for f in futures:
            name, scores, note = f.result()
            results[name] = scores
            if scores is None:
                print("%-10s skipped: %s" % (name, note))
                continue
            line = format_scores(name, scores)
            changes = []
            for label, a, b in scores:
                old = expected.get(name, {}).get(label)
                if old and old != (a, b):
                    changes.append("%s was %d/%d" % (label, *old))
            if note:
                line += "  (%s)" % note
            if changes:
                line += "  [" + ", ".join(changes) + "]"
            print(line, flush=True)

    if args.update:
        for name, scores in results.items():
            if scores is not None:
                expected[name] = {label: (a, b) for label, a, b in scores}
        with open(EXPECTED, "w") as f:
            f.write("# tests/regress.py --update: opensoda's scores against libsoda's references\n")
            for name in [c[0] for c in CASES if c[0] in expected]:
                f.write(format_scores(name, [(k, *v) for k, v in expected[name].items()]) + "\n")
        print("wrote", os.path.relpath(EXPECTED))
    if args.check:
        worse = []
        for name, scores in results.items():
            for label, a, b in scores or []:
                old = expected.get(name, {}).get(label)
                # The totals come from the reference (for starts and conf,
                # from the finals whose words match), so fewer matches is
                # what counts.
                if old is not None and a < old[0]:
                    worse.append("%s %s %d/%d (expected %d/%d)" % (name, label, a, b, *old))
        skipped = [n for n, s in results.items() if s is None]
        if worse:
            print("\nWORSE:\n  " + "\n  ".join(worse))
        print("\n%d cases, %d skipped, %d scores worse" % (len(results), len(skipped), len(worse)))
        return 1 if worse else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
