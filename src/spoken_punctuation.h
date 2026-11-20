// greco's PunctuationNormalizer, the ja-JP dictation stage that turns spoken
// punctuation commands into marks (「当店」 said alone → 「、」, 「疑問 符」 →
// 「？」). An LSTM (denorm/PUNCTUATION_LSTM.model.int8.tflite) flags the
// tokens of a result that are commands, and flagged runs that spell a spoken
// form from denorm/punctuation_converter_config.pb become its mark.
//
// Tokens come from the result with its spaces removed where the script needs
// none (all of them between Japanese words): the whole text is one token if
// it is in the model's vocabulary (PUNCTUATION_LSTM.syms), else one token per
// character. In libsoda the step that would split the text at spoken forms
// never matches a non-ASCII byte (its trie walker reads bytes as signed), so
// in practice a command converts only when it is the whole result.

#ifndef SODA_SPOKEN_PUNCTUATION_H_
#define SODA_SPOKEN_PUNCTUATION_H_

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "transducer.h"

namespace soda {

class SpokenPunctuationNormalizer {
 public:
  struct Word {
    std::string text;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
  };

  bool Init(const std::string& converter_file, const std::string& symbol_file,
            const std::string& model_file, std::string* error);

  // Converts the commands in `words`; returns false if nothing changed.
  bool Apply(std::vector<Word>* words);

 private:
  enum Side { kNone = 0, kLeft = 1, kRight = 2 };
  struct Symbol {
    std::string orthographic_form;
    std::string textual_form;
    std::string partner;
    Side side = kNone;
    const std::string& replacement() const {
      return textual_form.empty() ? orthographic_form : textual_form;
    }
  };
  struct Unsegmented {
    std::string key;           // The form without spaces.
    std::vector<size_t> cuts;  // Byte offsets in `key` where the form had spaces.
  };

  bool LoadConfig(const std::string& path, std::string* error);
  std::string Join(const std::vector<Word>& words) const;
  bool Gate(const std::string& text) const;
  std::string Segment(const std::string& text) const;
  void Tokenize(const std::string& text, std::vector<std::string>* model,
                std::vector<std::string>* orig) const;
  bool Flag(const std::vector<std::string>& tokens, std::vector<bool>* flags);
  std::string LongestForm(const std::string& text) const;
  int Find(const std::string& token) const;

  bool unsegmented_ = false;
  std::vector<Symbol> symbols_;
  // Spoken form words (enabled forms only) to their symbols, in config order.
  std::map<std::string, std::vector<int>> forms_;
  std::vector<Unsegmented> unsegmented_forms_;  // All forms, disabled too.
  std::vector<std::string> marks_;  // Each symbol's replacement.
  size_t max_words_ = 1;            // Most words in a spoken form.
  std::unordered_map<std::string, int> vocab_;
  int input_dim_ = 0;
  TfLiteModel model_;
};

}  // namespace soda

#endif  // SODA_SPOKEN_PUNCTUATION_H_
