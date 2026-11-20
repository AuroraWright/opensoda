// Case mapping for the scripts of the packs' languages (ASCII, Latin-1,
// Latin Extended-A, Greek, Cyrillic), over UTF-8 text.

#ifndef SODA_UNICODE_CASE_H_
#define SODA_UNICODE_CASE_H_

#include <cstddef>
#include <string>

namespace soda {

// The byte length of the UTF-8 character at `i`.
size_t CharLength(const std::string& s, size_t i);
std::string Lower(const std::string& s);
// The first character upper case, the rest lower case.
void Capitalize(std::string* s);

}  // namespace soda

#endif  // SODA_UNICODE_CASE_H_
