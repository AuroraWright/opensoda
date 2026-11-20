"""Reads the response dumps soda_api_client writes with --dump-responses.

Each record is {f64 time, u64 handle, u32 size, bytes}: a serialized
SodaResponse (handle 0) as libsoda or opensoda sent it. Only the fields the
tests compare are decoded.
"""

import struct
from dataclasses import dataclass, field

# SodaResponse.soda_type
RECOGNITION, STOP, SHUTDOWN, START, ENDPOINT, AUDIO_LEVEL, LANGID = 1, 2, 3, 4, 5, 6, 7
# SodaRecognitionResult.result_type
PARTIAL, FINAL = 1, 2


def _varint(b, i):
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def fields(b):
    """The (field number, value) pairs of a message: ints and bytes."""
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i)
            out.append((f, v))
        elif w == 2:
            n, i = _varint(b, i)
            out.append((f, b[i:i + n]))
            i += n
        elif w == 1:
            i += 8
        elif w == 5:
            i += 4
        else:
            raise ValueError("bad wire type %d" % w)
    return out


@dataclass
class Word:
    text: str
    start_ms: int  # from the start of the audio
    confidence: int


@dataclass
class Result:
    final: bool
    text: str  # formatted
    raw: str
    start_us: int
    end_us: int
    words: list = field(default_factory=list)


@dataclass
class Dump:
    results: list = field(default_factory=list)
    # (type, audio times): the wall-clock fields depend on how fast the
    # audio was fed.
    endpoints: list = field(default_factory=list)
    langid: list = field(default_factory=list)  # (language, level, switch result)

    @property
    def partials(self):
        return [r for r in self.results if not r.final]

    @property
    def finals(self):
        return [r for r in self.results if r.final]


def read(path):
    d = open(path, "rb").read()
    i, dump = 0, Dump()
    while i < len(d):
        _, handle, n = struct.unpack_from("<dQI", d, i)
        i += 20
        b = d[i:i + n]
        i += n
        if handle:
            continue
        top = dict(fields(b))
        kind = top.get(1)
        if kind == RECOGNITION and 2 in top:
            r = fields(top[2])
            rd = dict(r)
            if rd.get(2) not in (PARTIAL, FINAL) or 4 not in rd:
                continue
            timing = dict(fields(rd[4]))
            hyps = [v.decode() for f, v in r if f == 1]
            res = Result(rd[2] == FINAL, hyps[0] if hyps else "",
                         hyps[1] if len(hyps) > 1 else "", timing.get(2, 0),
                         timing.get(4, 0))
            for f, v in r:
                if f == 5:
                    w = dict(fields(v))
                    res.words.append(Word(w[1].decode(), timing.get(2, 0) // 1000 + w.get(2, 0),
                                          w.get(5, 0)))
            dump.results.append(res)
        elif kind == ENDPOINT and 3 in top:
            ep = dict(fields(top[3]))
            t = dict(fields(ep.get(2, b"")))
            dump.endpoints.append((ep.get(1, 0), t.get(2, 0), t.get(4, 0)))
        elif kind == LANGID and 5 in top:
            lid = dict(fields(top[5]))
            dump.langid.append((lid.get(1, b"").decode(), lid.get(2, 0), lid.get(3, 0)))
    return dump
