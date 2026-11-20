#!/usr/bin/env python3
"""Downloads a SODA language pack the way Chrome's component updater does.

  tests/fetch_pack.py LOCALE...    (e.g. de-DE fr-FR, or "all")

Chrome downloads only the packs for the languages its user picks; testing
the others needs them too. Each pack is unpacked into
re/packs/SODALanguagePacks/<locale>/<version>/SODAModels, where the tests
look for packs. re/ is git-ignored: the packs are Google's and must not be
committed or redistributed.

The download is checked for the CRX3 header only, not its signature.
"""

import json
import os
import shutil
import struct
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS = os.path.join(ROOT, "re", "packs", "SODALanguagePacks")
UPDATE_URL = "https://update.googleapis.com/service/update2/json"

# The component ids of Chrome's SODA language packs.
APP_IDS = {
    "en-US": "oegebmmcimckjhkhbggblnkjloogjdfg",
    "ja-JP": "onhpjgkfgajmkkeniaoflicgokpaebfa",
    "de-DE": "jclgnikdalajmocbnlgieibfmlejnhmg",
    "fr-FR": "goaoclndmgofblfopkopecdpfhljclbd",
    "es-ES": "jkcckmaejhmbhagbcebpejbihcnighdb",
    "it-IT": "jhefnhlmpagbceldaobdpcjhkknfjohi",
    "ru-RU": "njmnbpiilokhobofgcilobdmpgdkfefo",
    "ko-KR": "cogjdigcnchhklfjkhalghhfebjpdjhb",
    "hi-IN": "aolgaekiiglfhnjgagdljeilgegpfecn",
    "pt-BR": "anfcoadblmjplmkgmahlkhgnngkhoben",
    "id-ID": "kcganlnenpblkkhbnppfigiochoedgnh",
    "pl-PL": "dpkgmjeebdgklolimcncdakahbaoklpm",
    "th-TH": "olcokbjilkakbaknajbfokboieipblel",
    "tr-TR": "ifacfjnibplkincdecpkkedmekgfeoaj",
    "vi-VN": "bpnckojighgibleldbofdjmfdipfeiak",
    "cmn-Hans-CN": "plffpkkhdibnpffncjpflnolflkpgjno",
    "cmn-Hant-TW": "jblddigdjechninjlbkjjeiadfgmlpoc",
}


def update_check(app_id):
    request = {"request": {
        "protocol": "3.1", "acceptformat": "crx3,puff", "prodversion": "157.0.8103.79",
        "os": {"platform": "Mac OS X", "arch": "arm64"}, "arch": "arm64", "nacl_arch": "arm64",
        "apps": [{"appid": app_id, "version": "0.0.0.0", "updatecheck": {}}]}}
    req = urllib.request.Request(UPDATE_URL, data=json.dumps(request).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        reply = json.loads(r.read()[4:])  # after the )]}' guard
    check = reply["response"]["app"][0]["updatecheck"]
    package = check["manifest"]["packages"]["package"][0]
    url = check["urls"]["url"][-1]["codebase"] + package["name"]
    return check["manifest"]["version"], url


def fetch(locale):
    version, url = update_check(APP_IDS[locale])
    dest = os.path.join(PACKS, locale, version, "SODAModels")
    if os.path.isdir(dest):
        print("%s %s: already in %s" % (locale, version, os.path.relpath(dest, ROOT)))
        return
    os.makedirs(PACKS, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(PACKS)) as tmp:
        crx = os.path.join(tmp, "pack.crx3")
        urllib.request.urlretrieve(url, crx)
        data = open(crx, "rb").read()
        if data[:4] != b"Cr24":
            raise RuntimeError("%s: not a CRX3 file" % url)
        header = struct.unpack("<I", data[8:12])[0]
        archive = os.path.join(tmp, "pack.zip")
        open(archive, "wb").write(data[12 + header:])
        unpacked = os.path.join(tmp, "pack")
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                path = os.path.normpath(os.path.join(unpacked, name))
                if not path.startswith(unpacked + os.sep):
                    raise RuntimeError("%s: unsafe path in archive: %r" % (url, name))
            z.extractall(unpacked)
        models = os.path.join(unpacked, "SODAModels")
        if not os.path.isdir(models):
            raise RuntimeError("%s: no SODAModels in archive" % url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(models, dest)
    print("%s %s: %s" % (locale, version, os.path.relpath(dest, ROOT)))


def main():
    locales = sys.argv[1:]
    if locales == ["all"]:
        locales = list(APP_IDS)
    unknown = [l for l in locales if l not in APP_IDS]
    if not locales or unknown:
        print(__doc__.strip(), "\n\nLocales:", " ".join(APP_IDS))
        return 1
    for locale in locales:
        fetch(locale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
