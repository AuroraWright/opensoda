#!/usr/bin/env python3
"""Measures CPU time and peak memory, opensoda against another library.

  tests/bench.py [--realtime] [--reference LIBRARY [--reference-speed X]] [CASE...]

Runs soda_api_client on the en (55 s), ja (122 s) and pl (15 s) regression
recordings, one at a time, with build/libsoda.dylib and, if given, a
reference library such as Chrome's libsoda. Offline, opensoda runs
unpaced (simulating real time, so that it recognizes all the audio) and
the reference at --reference-speed (default 8; libsoda reads simulated
real-time audio at real-time speed); with --realtime both are fed at
real-time speed, as in Live Caption.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regress import DATA, PACK_ROOTS, ROOT, find_pack  # noqa: E402

BENCH = {"en": ("en_bench", "en-US"), "ja": ("ja_lesson", "ja-JP"), "pl": ("pl", "pl-PL")}


def measure(cmd):
    p = subprocess.run(["/usr/bin/time", "-l", *cmd], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True)
    t = p.stderr
    real, user, sys_ = (float(x) for x in re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", t).groups())
    rss = int(re.search(r"(\d+)\s+maximum resident set size", t).group(1)) / 2**20
    return "real %6.2f s  user %6.2f s  sys %5.2f s  peak %4.0f MB" % (real, user, sys_, rss)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", default=list(BENCH), help="en, ja, pl")
    ap.add_argument("--realtime", action="store_true", help="feed both at real-time speed")
    ap.add_argument("--reference", help="another libsoda-compatible library to measure")
    ap.add_argument("--reference-speed", type=float, default=8)
    ap.add_argument("--packs", action="append", default=[])
    args = ap.parse_args()
    client = os.path.join(ROOT, "build", "soda_api_client")
    libs = [("opensoda", os.path.join(ROOT, "build", "libsoda.dylib"), 1 if args.realtime else 0)]
    if args.reference:
        libs.append(("reference", os.path.abspath(args.reference), 1 if args.realtime else args.reference_speed))
    for name in args.cases:
        recording, locale = BENCH[name]
        pack, note = find_pack(locale, PACK_ROOTS + args.packs)
        if not pack:
            print("%s: skipped (%s)" % (name, note))
            continue
        wav = os.path.join(DATA, recording, "audio.wav")
        for label, lib, speed in libs:
            cmd = [client, lib, pack, wav, "--speed", str(speed)] + (["--simulate-realtime"] if speed == 0 else [])
            print("%s %-9s %s" % (name, label, measure(cmd)),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
