#include "unicode_case.h"

#include <algorithm>
#include <cstdint>

namespace soda {

size_t CharLength(const std::string& s, size_t i) {
  const unsigned char c = s[i];
  const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
  return std::min(n, s.size() - i);
}

namespace {

uint32_t Decode(const std::string& s, size_t i, size_t n) {
  const unsigned char c = s[i];
  if (n == 1) return c;
  uint32_t cp = c & (0x7F >> n);
  for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  return cp;
}

void Encode(uint32_t cp, std::string* out) {
  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Simple case mapping for the scripts of the packs' languages: ASCII,
// Latin-1, Latin Extended-A, Greek and Cyrillic.
uint32_t CaseMap(uint32_t c, bool upper) {
  auto pair = [&](uint32_t u, uint32_t l) { return upper ? (c == l ? u : c) : (c == u ? l : c); };
  if (c < 0x80) {
    if (upper && c >= 'a' && c <= 'z') return c - 32;
    if (!upper && c >= 'A' && c <= 'Z') return c + 32;
    return c;
  }
  if ((c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0xE0 && c <= 0xFE && c != 0xF7)) {
    return c >= 0xE0 ? pair(c - 0x20, c) : pair(c, c + 0x20);
  }
  if (c == 0xFF || c == 0x178) return pair(0x178, 0xFF);
  if (c >= 0x100 && c <= 0x17E && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149) {
    // Upper case is even, except in 139–148 and 179–17E, where it is odd.
    const bool odd_upper = (c >= 0x139 && c <= 0x148) || c >= 0x179;
    const bool is_upper = (c % 2 == 1) == odd_upper;
    return is_upper ? pair(c, c + 1) : pair(c - 1, c);
  }
  if ((c >= 0x391 && c <= 0x3A9 && c != 0x3A2) || (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2)) {
    return c >= 0x3B1 ? pair(c - 0x20, c) : pair(c, c + 0x20);
  }
  if (c >= 0x410 && c <= 0x44F) return c >= 0x430 ? pair(c - 0x20, c) : pair(c, c + 0x20);
  if (c >= 0x400 && c <= 0x45F) return c >= 0x450 ? pair(c - 0x50, c) : pair(c, c + 0x50);
  return c;
}

}  // namespace

std::string Lower(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    const size_t n = CharLength(s, i);
    Encode(CaseMap(Decode(s, i, n), false), &out);
    i += n;
  }
  return out;
}

void Capitalize(std::string* s) {
  if (s->empty()) return;
  const size_t n = CharLength(*s, 0);
  std::string out;
  Encode(CaseMap(Decode(*s, 0, n), true), &out);
  *s = out + Lower(s->substr(n));
}

}  // namespace soda
