// Cases words with a language pack's Roadrunner capitalization model
// (denorm/SODA_capitalization_model.tflite).
//
// Usage: soda_capitalize MODEL word...
//        soda_capitalize --sentences MODEL < hex lines
//
// --sentences runs the model on serialized speech.textnorm.Sentence protos,
// one hex-encoded per input line, and prints each output the same way.

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "roadrunner_capitalizer.h"

int main(int argc, char** argv) {
  const bool sentences = argc > 1 && !std::strcmp(argv[1], "--sentences");
  if (sentences) {
    --argc;
    ++argv;
  }
  if (argc < 2 || (!sentences && argc < 3)) {
    std::fprintf(stderr, "usage: soda_capitalize [--sentences] MODEL word...\n");
    return 2;
  }
  soda::RoadrunnerCapitalizer capitalizer;
  std::string error;
  if (!capitalizer.Init(argv[1], /*window=*/5, /*left=*/1, /*right=*/1, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  if (sentences) {
    for (std::string line; std::getline(std::cin, line);) {
      std::string in;
      for (size_t i = 0; i + 1 < line.size(); i += 2) in += static_cast<char>(std::stoi(line.substr(i, 2), nullptr, 16));
      for (unsigned char c : capitalizer.Run(in)) std::printf("%02x", c);
      std::printf("\n");
    }
    return 0;
  }
  std::vector<std::string> words(argv + 2, argv + argc);
  capitalizer.Apply(&words);
  for (size_t i = 0; i < words.size(); ++i) std::printf("%s%s", i ? " " : "", words[i].c_str());
  std::printf("\n");
  return 0;
}
