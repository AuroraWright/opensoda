#!/usr/bin/env python3
"""Records the references: a libsoda-compatible library's responses.

  tests/record.py --library LIBRARY [--speed X] [--force] CASE...

Runs each case's recording through LIBRARY (normally Google's libsoda) and
writes the responses to tests/data/<recording>/<reference>.bin, where
tests/regress.py reads them. Existing references are kept unless --force.

libsoda's output depends on how fast it is fed: its encoder batches frames
as its threads keep up, and it skips audio when it falls more than 30 s
behind. Recording at --speed 1 (the default; real time, as Chrome feeds
it) takes as long as the audio.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cases import CASES  # noqa: E402
from regress import DATA, PACK_ROOTS, ROOT, find_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--library", required=True)
    ap.add_argument("--speed", type=float, default=1)
    ap.add_argument("--force", action="store_true", help="overwrite existing references")
    ap.add_argument("--packs", action="append", default=[])
    args = ap.parse_args()
    by_name = {c[0]: c for c in CASES}
    roots = PACK_ROOTS + args.packs
    status = 0
    for name in args.cases:
        if name not in by_name:
            print("%s: unknown case" % name)
            status = 1
            continue
        _, recording, reference, locale, options = by_name[name]
        out = os.path.join(DATA, recording, reference + ".bin")
        if os.path.exists(out) and not args.force:
            print("%s: %s exists (--force to overwrite)" % (name, os.path.relpath(out)))
            continue
        pack, note = find_pack(locale, roots)
        if not pack:
            print("%s: %s" % (name, note))
            status = 1
            continue
        opts = []
        for o in options:
            if "@" in o:
                prefix, other = o.split("@", 1)
                o = prefix + (find_pack(other, roots)[0] or "")
            opts.append(o)
        if note:
            print("%s: using %s" % (name, note))
        cmd = [os.path.join(ROOT, "build", "soda_api_client"), os.path.abspath(args.library), pack,
               os.path.join(DATA, recording, "audio.wav"), *opts, "--speed", str(args.speed),
               "--dump-responses", out]
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if p.returncode:
            print("%s: failed (%d): %s" % (name, p.returncode, p.stderr[-500:]))
            status = 1
        else:
            print("%s: wrote %s" % (name, os.path.relpath(out)))
    return status


if __name__ == "__main__":
    sys.exit(main())
