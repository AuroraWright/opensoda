// greco's PortablePqrnnSpokenPunctuationNormalizer (de-DE, fr-FR dictation),
// which runs speech_formatting's SpokenPunctuationInference: spoken
// punctuation commands become marks ("hallo anna komma wie geht es dir
// fragezeichen" → "Hallo anna, wie geht es dir?").
//
// The words are split at the punctuation they already contain ("12,50" →
// "12" "," "50"); a pQRNN model (denorm/model.tflite) labels each piece
// PUNCTUATION or KEEP, and runs of PUNCTUATION pieces that spell one of the
// config's spoken forms (matched greedily, longest prefix first) become its
// mark, placed by the symbol's type: FINAL marks join the word before,
// INITIAL marks the word after, INTERMEDIATE marks both, and STANDALONE
// ones (line breaks) stand alone. A mark that ends a sentence capitalizes
// the next word, and then also the first one. Pieces of one word are joined
// back afterwards. Settings: speech_formatting.SpokenPunctuationInference-
// Settings in the config's PortablePqrnnSpokenPunctuationNormalizerParams.

#ifndef SODA_PQRNN_SPOKEN_PUNCTUATION_H_
#define SODA_PQRNN_SPOKEN_PUNCTUATION_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "text_normalizer.h"

namespace soda {

class PqrnnSpokenPunctuationNormalizer : public TextNormalizer {
 public:
  // `settings` is the serialized SpokenPunctuationInferenceSettings.
  bool Init(const std::string& model_file, std::string_view settings, std::string* error);

  // `text` is the result's words joined by spaces.
  std::string Apply(const std::string& text) const override;

  // The same on timed words, as libsoda's DecodedWords: the result's words
  // with the times of the words they come from. False if nothing was done.
  struct Word {
    std::string text;
    int64_t start_ms = 0, end_ms = 0;
  };
  bool ApplyWords(std::vector<Word>* words) const;

 private:
  enum Type { kUndefined = 0, kInitial = 1, kIntermediate = 2, kFinal = 3, kStandalone = 4 };
  enum Side { kNoSide = 0, kLeft = 1, kRight = 2 };
  struct Symbol {
    std::string mark;     // orthographic_form
    std::string partner;  // The other mark of a pair.
    int type = kUndefined;
    int side = kNoSide;
    bool separate_by_space = false;
    bool end_of_sentence = false;
  };
  // A postnormalizer: replaces `from` in a token's text (2), its appended
  // (3) or prepended (4) punctuation. The packs' patterns are single marks.
  struct Replacement {
    std::string from, to;
    int type = 0;
  };
  struct Token;

  // Formats the words; empty if the model failed.
  std::vector<Token> Run(const std::vector<std::string>& words) const;
  std::vector<Token> Segment(const std::string& word, int index) const;
  bool Label(const std::vector<Token>& tokens, std::vector<bool>* punctuation) const;
  std::vector<Token> Format(const std::vector<Token>& tokens, const std::vector<bool>& punctuation) const;

  int capitalization_ = 0;  // PositionalCapitalizationBehavior.
  bool separate_words_by_space_ = false;
  bool by_whitespace_ = true;  // segment_by is not UNICODE_CHARACTER.
  std::vector<std::string> marks_;             // The segmenter's punctuation.
  std::vector<std::string> sentence_end_marks_;
  std::vector<Replacement> postnormalizers_;
  std::vector<Symbol> symbols_;
  std::map<std::string, std::vector<int>> forms_;  // Spoken form → symbols, in config order.

  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  int scores_output_ = -1, labels_output_ = -1;
};

}  // namespace soda

#endif  // SODA_PQRNN_SPOKEN_PUNCTUATION_H_
