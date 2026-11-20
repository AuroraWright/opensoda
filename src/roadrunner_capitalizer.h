// Restores letter case with a language pack's Roadrunner capitalization
// model (RuntimeRoadrunnerCapitalizationNormalizer: id-ID, pl-PL, tr-TR,
// vi-VN), as libsoda does before formatting.
//
// denorm/SODA_capitalization_model.tflite is a text normalization model that
// reads and writes a serialized speech.textnorm.Sentence. A tagger labels
// each word "keep as is" or "rewrite"; a character seq2seq model rewrites
// the latter (in practice, capitalizes them). Its custom ops, which move
// words between the Sentence proto and tensors, are implemented here. Like
// libsoda, long inputs are cased in overlapping windows of words.

#ifndef SODA_ROADRUNNER_CAPITALIZER_H_
#define SODA_ROADRUNNER_CAPITALIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace soda {

class RoadrunnerCapitalizer {
 public:
  // `window` words per model call; `left` and `right` of them are context
  // only, except at the ends of the input.
  bool Init(const std::string& model_path, int window, int left, int right, std::string* error);

  // Replaces each word by its cased form.
  void Apply(std::vector<std::string>* words);

  // Runs the model on one serialized Sentence and returns its output.
  std::string Run(const std::string& sentence);

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  int window_ = 5;
  int left_ = 1;
  int right_ = 1;
};

}  // namespace soda

#endif  // SODA_ROADRUNNER_CAPITALIZER_H_
