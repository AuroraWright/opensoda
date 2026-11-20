// Contextual biasing toward caller-supplied phrases (en-US "twiddler").
//
// libsoda's class_substitution_twiddler turns each phrase into word pieces
// with the pack's portable word-piece model and builds an FST over them that
// both decoders consult: every matching piece earns a reward of the phrase's
// boost plus a base weight, a partial match that breaks off gives its reward back,
// and a completed phrase keeps it. Continuing after a completed phrase costs
// retwiddle_penalty once. Pieces are matched as spelled (case included).

#ifndef SODA_BIASING_H_
#define SODA_BIASING_H_

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "symbol_table.h"

namespace soda {

// greco's PortableWordPieceModel (*.wpm.portable): byte-pair encoding over
// characters, where piece ids give the merge order.
class WordPieceModel {
 public:
  bool Load(const std::string& path, std::string* error);
  // Pieces of one word, the first with the word-start mark "▁". Empty if the
  // word has characters outside the model.
  std::vector<std::string> Encode(const std::string& word) const;

 private:
  std::map<std::string, int> ids_;
  std::vector<std::string> pieces_;  // By id.
  std::map<std::pair<int, int>, int> merges_;  // (left, right) -> merged id.
};

struct BiasPhrase {
  std::string text;
  float boost = 1.0f;
  std::string context;  // The name of the context input it came in.
};

class ContextBiaser {
 public:
  struct Params {
    float token_weight = 0.0f;       // Added to the boost for each piece.
    bool respect_boost = true;       // respect_grammar_phrase_boost.
    float retwiddle_penalty = 0.0f;  // FstTwiddlerBuilderParams (a reward).
  };

  // Phrases with no piece in `symbols` are skipped.
  void Build(const std::vector<BiasPhrase>& phrases, const WordPieceModel& wpm,
             const SymbolTable& symbols, const Params& params);
  bool empty() const { return nodes_.size() <= 1; }

  static constexpr int kStart = 0;
  // Moves from `state` over `label`, returning the change in cost (negative
  // for a reward).
  float Step(int state, int label, int* next) const;

 private:
  struct Node {
    std::map<int, int> children;
    float reward = 0.0f;  // Accumulated along the phrase so far.
    int fail = 0;         // Longest proper suffix that is a node.
    bool final = false;
  };
  std::vector<Node> nodes_;
  float retwiddle_penalty_ = 0.0f;
};

}  // namespace soda

#endif  // SODA_BIASING_H_
