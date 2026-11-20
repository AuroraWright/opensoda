#include "language_packs.h"

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <sys/stat.h>

namespace soda {
namespace {

std::vector<std::string> ChromeUserDataDirs() {
  std::vector<std::string> dirs;
  const char* home = std::getenv("HOME");
  if (!home) return dirs;
  const std::string h = home;
#if defined(__APPLE__)
  for (const char* d : {"Google/Chrome", "Google/Chrome Beta", "Google/Chrome Dev",
                        "Google/Chrome Canary", "Chromium"}) {
    dirs.push_back(h + "/Library/Application Support/" + d);
  }
#else
  for (const char* d : {"google-chrome", "google-chrome-beta", "google-chrome-unstable",
                        "chromium"}) {
    dirs.push_back(h + "/.config/" + d);
  }
#endif
  return dirs;
}

std::vector<std::string> ListDir(const std::string& path) {
  std::vector<std::string> out;
  if (DIR* d = opendir(path.c_str())) {
    while (dirent* e = readdir(d)) {
      if (e->d_name[0] != '.') out.emplace_back(e->d_name);
    }
    closedir(d);
  }
  return out;
}

std::vector<int> VersionParts(const std::string& v) {
  std::vector<int> parts;
  size_t i = 0;
  while (i < v.size()) {
    parts.push_back(std::atoi(v.c_str() + i));
    i = v.find('.', i);
    if (i == std::string::npos) break;
    ++i;
  }
  return parts;
}

bool HasModels(const std::string& version_dir) {
  struct stat st;
  return stat((version_dir + "/SODAModels/configs").c_str(), &st) == 0;
}

}  // namespace

std::vector<InstalledLanguagePack> ListInstalledLanguagePacks() {
  std::vector<InstalledLanguagePack> packs;
  for (const std::string& root : ChromeUserDataDirs()) {
    const std::string base = root + "/SODALanguagePacks";
    for (const std::string& lang : ListDir(base)) {
      for (const std::string& ver : ListDir(base + "/" + lang)) {
        const std::string path = base + "/" + lang + "/" + ver;
        if (HasModels(path)) packs.push_back({lang, ver, path});
      }
    }
  }
  std::sort(packs.begin(), packs.end(),
            [](const InstalledLanguagePack& a, const InstalledLanguagePack& b) {
              if (a.language != b.language) return a.language < b.language;
              return VersionParts(a.version) > VersionParts(b.version);
            });
  return packs;
}

std::string FindInstalledLanguagePack(const std::string& language) {
  for (const auto& p : ListInstalledLanguagePacks()) {
    if (p.language == language) return p.path;
  }
  return "";
}

}  // namespace soda
