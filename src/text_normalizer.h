// A stage of the config's denormalizer list (common_denormalizers_list):
// rewrites a result's words, joined by spaces.

#ifndef SODA_TEXT_NORMALIZER_H_
#define SODA_TEXT_NORMALIZER_H_

#include <string>

namespace soda {

class TextNormalizer {
 public:
  virtual ~TextNormalizer() = default;
  virtual std::string Apply(const std::string& text) const = 0;
};

}  // namespace soda

#endif  // SODA_TEXT_NORMALIZER_H_
