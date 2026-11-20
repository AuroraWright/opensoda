// Punctuates tokens with a language pack's punctuation model.
//
// Usage: soda_punctuate [--scores] SODAMODELS_DIR token...
//
// --scores prints each token's class probabilities instead.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "punctuator.h"

int main(int argc, char** argv) {
  const bool show_scores = argc > 1 && !std::strcmp(argv[1], "--scores");
  if (show_scores) {
    --argc;
    ++argv;
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: soda_punctuate [--scores] SODAMODELS_DIR token...\n");
    return 2;
  }
  soda::Punctuator punctuator;
  std::string error;
  if (!punctuator.Init(argv[1], &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  std::vector<std::string> words(argv + 2, argv + argc);
  if (show_scores) {
    std::vector<std::vector<float>> scores;
    punctuator.Predict(words, &scores);
    std::printf("token");
    for (const auto& l : punctuator.labels()) std::printf("\t%s", l.empty() ? "-" : l.c_str());
    std::printf("\n");
    for (size_t i = 0; i < words.size(); ++i) {
      std::printf("%s", words[i].c_str());
      for (float p : scores[i]) std::printf("\t%.3f", p);
      std::printf("\n");
    }
    return 0;
  }
  punctuator.Punctuate(&words, /*final=*/true);
  for (size_t i = 0; i < words.size(); ++i) std::printf("%s%s", i ? " " : "", words[i].c_str());
  std::printf("\n");
  return 0;
}
