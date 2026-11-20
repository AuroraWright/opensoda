// Automatic punctuation using the language pack's SODA_punctuation_model.
//
// Packs ship one of two kinds of model:
// - a pQRNN text classifier (TensorFlow seq_flow_lite, ja-JP): it takes
//   space-separated tokens of a hypothesis and predicts, for each token, the
//   punctuation mark to append. It needs two seq_flow_lite custom ops,
//   SEQUENCE_STRING_PROJECTION and the QRNN pooling op, implemented here.
// - a "Roadrunner" tagger (MODEL_TYPE_ROADRUNNER, en-US): it takes one
//   lowercased string per word, hashed into character n-grams by the
//   CharNGramFeatures custom op (implemented here), and returns one label per
//   word, the mark before it (from the config's punctuation codebook). Long
//   inputs are tagged in overlapping windows.

#ifndef SODA_PUNCTUATOR_H_
#define SODA_PUNCTUATOR_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
class MutableOpResolver;
}  // namespace tflite

namespace soda {

// Adds the seq_flow_lite custom ops pQRNN models use to `resolver`.
void AddPqrnnOps(tflite::MutableOpResolver* resolver);

class Punctuator {
 public:
  // `models_dir` is the SODAModels directory. Returns false if the pack has
  // no punctuation model.
  bool Init(const std::string& models_dir, std::string* error);

  // Inserts predicted punctuation into `words` in place. Like libsoda (and
  // as the pack's SODA_punctuation_config.pb says for ja-JP), the model sees
  // one token per character, so a mark may land inside a word. For a partial
  // result (`final` false), no mark follows the last character.
  // `sentence_ends`, if given, receives for each word whether the model
  // predicted an end-of-sentence mark after it (even where the word already
  // had a mark, which then stays as it is). `model_words`, if given, are
  // the words as a Roadrunner model sees them (without dictated marks).
  void Punctuate(std::vector<std::string>* words, bool final,
                 std::vector<bool>* sentence_ends = nullptr,
                 const std::vector<std::string>* model_words = nullptr);

  // Returns, for each model token, the punctuation to append ("" for none).
  // If `scores` is given, it receives each token's class probabilities, in
  // the order of labels().
  std::vector<std::string> Predict(const std::vector<std::string>& tokens,
                                   std::vector<std::vector<float>>* scores = nullptr);
  const std::vector<std::string>& labels() const { return labels_; }
  // The marks the config lets the model write.
  const std::vector<std::string>& supported_marks() const { return supported_marks_; }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::vector<std::string> labels_;
  std::vector<std::string> supported_marks_;  // From the punctuation config.
  std::vector<std::string> end_of_sentence_marks_;
  // BasicTextSegmenterConfig.SegmentBy: what a pQRNN model token is.
  static constexpr int kSimpleWhitespace = 1;
  int segment_by_ = 2;  // UNICODE_CHARACTER.

  // Roadrunner models (RoadrunnerPunctuationNormalizerParams).
  void Capitalize(std::vector<std::string>* words) const;
  void PunctuateRoadrunner(std::vector<std::string>* words, bool final,
                           std::vector<bool>* sentence_ends,
                           const std::vector<std::string>* model_words);
  bool IsSentenceEnd(const std::string& mark) const;
  std::vector<int> RoadrunnerLabels(const std::vector<std::string>& tokens);
  bool roadrunner_ = false;
  std::vector<std::string> codebook_;
  bool append_dummy_token_ = false;
  int window_size_ = 30;
  int window_left_ = 2;
  int window_right_ = 5;
  int window_min_remainder_ = 0;
  int scores_output_ = -1;
  int labels_output_ = -1;
};

}  // namespace soda

#endif  // SODA_PUNCTUATOR_H_
