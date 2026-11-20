// Restores letter case with a language pack's capitalization normalizer
// (greco's GeneralCapitalizationNormalizer), as libsoda does before
// formatting: denorm/transducer.pruned.fst lists each word's case variants
// ("sns" -> "SNS" or "sns") and denorm/lm.pruned.sorted.fst, an OpenGrm
// n-gram model, picks the likeliest sequence. Both are OpenFst VectorFsts.

#ifndef SODA_TRUECASER_H_
#define SODA_TRUECASER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace soda {

class Truecaser {
 public:
  bool Init(const std::string& transducer_path, const std::string& lm_path,
            std::string* error);

  // Replaces each word by its likeliest case variant. Words without variants
  // are kept as they are (and still give the language model context); words
  // the transducer does not know are kept and left out of the context.
  void Apply(std::vector<std::string>* words) const;

 private:
  struct Arc {
    int32_t ilabel;
    int32_t olabel;
    float weight;
    int32_t next;
  };
  struct Fst {
    int start = -1;
    std::vector<float> final;
    std::vector<uint32_t> first;  // Per state, its first arc; size states + 1.
    std::vector<Arc> arcs;
    std::vector<std::string> symbols;  // Input symbol table, by label.
  };
  static bool LoadFst(const std::string& path, Fst* fst, std::string* error);

  // The ways to read word `w` after LM state `state`, directly or after
  // backoff arcs: (cost, next state) each.
  void Step(int state, int w, std::vector<std::pair<float, int>>* out) const;
  // Cost of ending the sentence in `state`.
  float FinalCost(int state) const;

  Fst lm_;
  std::unordered_map<std::string, int> lm_ids_;
  // Words with a case variant other than themselves: (LM label, text) each.
  std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> variants_;
  // The transducer's symbols: its words in lower case and their variants.
  std::unordered_set<std::string> known_;
  int unk_ = -1;
  int end_of_sentence_ = -1;  // </S>, when the transducer writes it to end a sentence.
};

}  // namespace soda

#endif  // SODA_TRUECASER_H_
