"""The regression cases: recordings under tests/data and the packs they need.

tests/data/<recording>/ holds audio.wav and libsoda's responses to it, one
dump per reference (caption.bin, ime.bin, ...). Neither is part of the
project: the recordings are local test audio and the references come from
Google's libsoda (see tests/README.md for how to record them).
"""

# The pack versions the references were recorded with.
PACK_VERSIONS = {
    "en-US": "1.5075.0",
    "ja-JP": "1.5072.0",
    "de-DE": "1.5072.0",
    "fr-FR": "1.5072.0",
    "es-ES": "1.5071.0",
    "it-IT": "1.5071.0",
    "ru-RU": "1.3071.0",
    "ko-KR": "1.3073.0",
    "hi-IN": "1.3072.0",
    "pt-BR": "1.3071.0",
    "cmn-Hans-CN": "1.3071.0",
    "id-ID": "1.3071.0",
    "pl-PL": "1.3071.0",
    "th-TH": "1.3071.0",
    "tr-TR": "1.3071.0",
    "vi-VN": "1.3071.0",
    "cmn-Hant-TW": "1.3071.0",
}

IME = ["--ime"]
BIAS = ["Siobhan Kowalczyk", "Zorblax", "Nguyen Thi Huong", "Quillfeather", "Xelvantra"]


def phrases(boost):
    return [a for p in BIAS for a in ("--phrase", "%s:%d" % (p, boost))]


# (name, recording, reference, pack, soda_api_client options). An option
# "@xx-YY" is replaced with that pack's directory.
CASES = [
    # en-US: synthesized sentences in captions and dictation.
    ("en", "en_bench", "caption", "en-US", []),
    ("en2", "en2", "caption", "en-US", []),
    ("en3", "en3", "caption", "en-US", []),
    ("en3_ime", "en3", "ime", "en-US", IME),
    ("en4", "en4", "caption", "en-US", []),
    ("en4_ime", "en4", "ime", "en-US", IME),
    ("en5", "en5", "caption", "en-US", []),
    ("en5_ime", "en5", "ime", "en-US", IME),
    # Contextual biasing toward five rare names, at boosts 1 and 3.
    ("b1", "bias", "boost1", "en-US", phrases(1)),
    ("b3", "bias", "boost3", "en-US", phrases(3)),
    # ja-JP -> en-US -> ja-JP language switching.
    ("mix", "mix", "caption", "ja-JP", ["--multilang", "en-us=@en-US"]),
    # ja-JP: a 122 s lesson recording (the reference captured from Chrome),
    # and synthesized dictation with spoken punctuation.
    ("ja", "ja_lesson", "caption", "ja-JP", []),
    ("ja_ime", "ja_lesson", "ime", "ja-JP", IME),
    ("jsp", "ja_punct", "ime", "ja-JP", IME),
    ("jsp2", "ja_punct2", "ime", "ja-JP", IME),
    ("jsp3", "ja_punct3", "ime", "ja-JP", IME),
    # de-DE and fr-FR. The _speed8 references come from libsoda fed 8 times
    # faster than real time, which changes how its encoder batches frames.
    ("de", "de", "caption", "de-DE", []),
    ("de_speed8", "de", "caption_speed8", "de-DE", []),
    ("deime_cap", "de_dict", "caption", "de-DE", []),
    ("deime_ime", "de_dict", "ime", "de-DE", IME),
    ("fr", "fr", "caption", "fr-FR", []),
    ("fr_run2", "fr", "caption_run2", "fr-FR", []),
    ("fr_speed8", "fr", "caption_speed8", "fr-FR", []),
    # Five synthesized sentences in each of the other packs.
    ("es", "es", "caption", "es-ES", []),
    ("it", "it", "caption", "it-IT", []),
    ("ru", "ru", "caption", "ru-RU", []),
    ("ko", "ko", "caption", "ko-KR", []),
    ("hi", "hi", "caption", "hi-IN", []),
    ("pt", "pt", "caption", "pt-BR", []),
    ("zh", "zh", "caption", "cmn-Hans-CN", []),
    ("id", "id", "caption", "id-ID", []),
    ("pl", "pl", "caption", "pl-PL", []),
    ("th", "th", "caption", "th-TH", []),
    ("tr", "tr", "caption", "tr-TR", []),
    ("vi", "vi", "caption", "vi-VN", []),
    ("tw", "tw", "caption", "cmn-Hant-TW", []),
]
