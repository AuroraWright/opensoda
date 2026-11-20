// Locates SODA language packs installed by Chrome's component updater.

#ifndef SODA_LANGUAGE_PACKS_H_
#define SODA_LANGUAGE_PACKS_H_

#include <string>
#include <vector>

namespace soda {

struct InstalledLanguagePack {
  std::string language;  // e.g. "ja-JP"
  std::string version;   // e.g. "1.5072.0"
  std::string path;      // Version directory containing SODAModels/.
};

// Lists packs from Chrome's user data directories (all channels and
// Chromium), newest version first for each language.
std::vector<InstalledLanguagePack> ListInstalledLanguagePacks();

// Newest installed pack for `language`, or "" if none.
std::string FindInstalledLanguagePack(const std::string& language);

}  // namespace soda

#endif  // SODA_LANGUAGE_PACKS_H_
