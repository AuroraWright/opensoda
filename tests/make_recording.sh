#!/bin/bash
# Synthesizes a test recording with macOS speech synthesis.
#
#   tests/make_recording.sh NAME VOICE "sentence 1" "sentence 2" ...
#
# Writes tests/data/NAME/audio.wav (16 kHz mono, 2 s of silence before,
# between and after the sentences) and sents.txt. `say -v '?'` lists the
# voices. Needs sox. Then add a case to tests/cases.py and record its
# reference with tests/record.py.
set -euo pipefail
if [ $# -lt 3 ]; then sed -n '2,10p' "$0"; exit 1; fi
name=$1 voice=$2; shift 2
dir=$(cd "$(dirname "$0")" && pwd)/data/$name
mkdir -p "$dir"
tmp=$(mktemp -d)
trap 'rm -r "$tmp"' EXIT
sox -n -r 16000 -c 1 -b 16 "$tmp/sil.wav" trim 0 2
parts=("$tmp/sil.wav")
i=1
: > "$dir/sents.txt"
for text in "$@"; do
  say -v "$voice" -o "$tmp/s$i.aiff" "$text"
  sox "$tmp/s$i.aiff" -r 16000 -c 1 -b 16 "$tmp/s$i.wav"
  parts+=("$tmp/s$i.wav" "$tmp/sil.wav")
  echo "$text" >> "$dir/sents.txt"
  i=$((i + 1))
done
sox "${parts[@]}" "$dir/audio.wav"
echo "$dir/audio.wav: $(soxi -D "$dir/audio.wav") s"
