#include "truecaser.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "config.h"
#include "unicode_case.h"

namespace soda {
namespace {

constexpr uint32_t kFstMagic = 0x7eb2fdd6;
constexpr int32_t kHasISymbols = 1, kHasOSymbols = 2, kIsAligned = 4;
// Cost charged for a word the language model cannot score at all.
constexpr float kUnscoredCost = 20.0f;
// Marks spoken punctuation attaches to words.
constexpr const char* kAttachedMarks[] = {
    ".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "\"", "'", "#", "@", "&", "*",
    "/", "\\", "_", "^", "|", "~", "-", "\u2013", "\u2026", "\u201e", "\u201c", "\u201d",
    "\u201a", "\u2018", "\u2019", "\u00ab", "\u00bb", "\u00bf", "\u00a1"};

class Reader {
 public:
  explicit Reader(const std::string& data) : data_(data) {}
  template <typename T>
  bool Read(T* v) {
    if (data_.size() - pos_ < sizeof(T)) return false;
    std::memcpy(v, data_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return true;
  }
  bool ReadString(std::string* s) {
    int32_t n;
    if (!Read(&n) || n < 0 || data_.size() - pos_ < static_cast<size_t>(n)) return false;
    s->assign(data_, pos_, n);
    pos_ += n;
    return true;
  }
  // OpenFst SymbolTable::Read: magic, name, available key, size, entries.
  bool ReadSymbols(std::vector<std::string>* symbols) {
    int32_t magic;
    std::string name;
    int64_t available, size;
    if (!Read(&magic) || !ReadString(&name) || !Read(&available) || !Read(&size)) return false;
    for (int64_t i = 0; i < size; ++i) {
      std::string symbol;
      int64_t key;
      if (!ReadString(&symbol) || !Read(&key) || key < 0) return false;
      if (symbols && key < (1 << 24)) {
        if (symbols->size() <= static_cast<size_t>(key)) symbols->resize(key + 1);
        (*symbols)[key] = std::move(symbol);
      }
    }
    return true;
  }

 private:
  const std::string& data_;
  size_t pos_ = 0;
};

}  // namespace

bool Truecaser::LoadFst(const std::string& path, Fst* fst, std::string* error) {
  std::string data;
  if (!ReadFile(path, &data)) {
    *error = "cannot read " + path;
    return false;
  }
  Reader r(data);
  uint32_t magic;
  std::string fst_type, arc_type;
  int32_t version, flags;
  uint64_t properties;
  int64_t start, num_states, num_arcs;
  if (!r.Read(&magic) || magic != kFstMagic || !r.ReadString(&fst_type) ||
      !r.ReadString(&arc_type) || !r.Read(&version) || !r.Read(&flags) ||
      !r.Read(&properties) || !r.Read(&start) || !r.Read(&num_states) || !r.Read(&num_arcs)) {
    *error = path + " is not an OpenFst file";
    return false;
  }
  if (fst_type != "vector" || arc_type != "standard" || (flags & kIsAligned)) {
    *error = path + ": unsupported FST (" + fst_type + "/" + arc_type + ")";
    return false;
  }
  if ((flags & kHasISymbols) && !r.ReadSymbols(&fst->symbols)) {
    *error = path + ": bad input symbol table";
    return false;
  }
  if ((flags & kHasOSymbols) && !r.ReadSymbols(nullptr)) {
    *error = path + ": bad output symbol table";
    return false;
  }
  fst->start = static_cast<int>(start);
  fst->final.resize(num_states);
  fst->first.resize(num_states + 1);
  for (int64_t s = 0; s < num_states; ++s) {
    int64_t n;
    if (!r.Read(&fst->final[s]) || !r.Read(&n) || n < 0) {
      *error = path + ": truncated";
      return false;
    }
    fst->first[s] = static_cast<uint32_t>(fst->arcs.size());
    for (int64_t k = 0; k < n; ++k) {
      Arc a;
      if (!r.Read(&a.ilabel) || !r.Read(&a.olabel) || !r.Read(&a.weight) || !r.Read(&a.next)) {
        *error = path + ": truncated";
        return false;
      }
      fst->arcs.push_back(a);
    }
  }
  fst->first[num_states] = static_cast<uint32_t>(fst->arcs.size());
  return true;
}

bool Truecaser::Init(const std::string& transducer_path, const std::string& lm_path,
                     std::string* error) {
  Fst transducer;
  if (!LoadFst(transducer_path, &transducer, error) || !LoadFst(lm_path, &lm_, error)) {
    return false;
  }
  for (size_t i = 0; i < lm_.symbols.size(); ++i) {
    if (!lm_.symbols[i].empty()) lm_ids_.emplace(lm_.symbols[i], static_cast<int>(i));
  }
  if (auto it = lm_ids_.find("<UNK>"); it != lm_ids_.end()) unk_ = it->second;
  // The language model's arcs must be sorted by label for Step's search.
  for (size_t s = 0; s + 1 < lm_.first.size(); ++s) {
    if (!std::is_sorted(lm_.arcs.begin() + lm_.first[s], lm_.arcs.begin() + lm_.first[s + 1],
                        [](const Arc& a, const Arc& b) { return a.ilabel < b.ilabel; })) {
      *error = lm_path + ": arcs are not sorted by label";
      return false;
    }
  }
  // The transducer maps each word to its variants, from its start state
  // (ja-JP) or from the states its start reaches without input (de-DE first
  // writes a sentence start, <S>).
  const auto& sym = transducer.symbols;
  std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> all;
  std::vector<int> queue = {transducer.start};
  std::vector<bool> seen(transducer.first.size(), false);
  seen[transducer.start] = true;
  while (!queue.empty()) {
    const int s = queue.back();
    queue.pop_back();
    for (uint32_t k = transducer.first[s]; k < transducer.first[s + 1]; ++k) {
      const Arc& a = transducer.arcs[k];
      if (a.ilabel == 0) {
        // A sentence end the transducer writes (de-DE, fr-FR, es-ES; not
        // ru-RU or it-IT, which map words only) is scored by the LM.
        if (a.olabel > 0 && static_cast<size_t>(a.olabel) < sym.size() && sym[a.olabel] == "</S>") {
          if (auto it = lm_ids_.find("</S>"); it != lm_ids_.end()) end_of_sentence_ = it->second;
        }
        if (a.next >= 0 && !seen[a.next]) {
          seen[a.next] = true;
          queue.push_back(a.next);
        }
        continue;
      }
      if (a.olabel <= 0 || static_cast<size_t>(std::max(a.ilabel, a.olabel)) >= sym.size()) continue;
      const std::string& out = sym[a.olabel];
      auto id = lm_ids_.find(out);
      all[sym[a.ilabel]].emplace_back(id == lm_ids_.end() ? unk_ : id->second, out);
    }
  }
  for (const auto& symbol : sym) {
    if (!symbol.empty()) known_.insert(symbol);
  }
  for (auto& [word, v] : all) {
    if (v.size() > 1 || v.front().second != word) variants_.emplace(word, std::move(v));
  }
  return true;
}

void Truecaser::Step(int state, int w, std::vector<std::pair<float, int>>* out) const {
  out->clear();
  // libsoda composes with the LM's backoff arcs as plain epsilons, so a word
  // may also be read after backing off from a history that has it.
  for (int label : {w, unk_}) {
    if (label <= 0) continue;
    float cost = 0.0f;
    for (int s = state;;) {
      const Arc* begin = lm_.arcs.data() + lm_.first[s];
      const Arc* end = lm_.arcs.data() + lm_.first[s + 1];
      const Arc* a = std::lower_bound(begin, end, label,
                                      [](const Arc& arc, int l) { return arc.ilabel < l; });
      if (a != end && a->ilabel == label) out->emplace_back(cost + a->weight, a->next);
      if (begin == end || begin->ilabel != 0) {
        if (out->empty() && label == unk_) out->emplace_back(cost + kUnscoredCost, s);
        break;
      }
      cost += begin->weight;  // Back off to a shorter history.
      s = begin->next;
    }
    if (!out->empty()) return;  // Unknown words fall back to <UNK>.
  }
  if (out->empty()) out->emplace_back(kUnscoredCost, state);
}

float Truecaser::FinalCost(int state) const {
  auto final_cost = [this](int s) {
    float best = kUnscoredCost, cost = 0.0f;
    for (;;) {
      if (std::isfinite(lm_.final[s])) best = std::min(best, cost + lm_.final[s]);
      const Arc* begin = lm_.arcs.data() + lm_.first[s];
      if (lm_.first[s] == lm_.first[s + 1] || begin->ilabel != 0) return best;
      cost += begin->weight;
      s = begin->next;
    }
  };
  // de-DE's and fr-FR's transducers end every sentence with </S>, which the
  // LM scores like a word; otherwise the LM's final weights end it.
  if (end_of_sentence_ <= 0) return final_cost(state);
  std::vector<std::pair<float, int>> ends;
  Step(state, end_of_sentence_, &ends);
  float best = std::numeric_limits<float>::infinity();
  for (const auto& [cost, next] : ends) best = std::min(best, cost + final_cost(next));
  return best;
}

void Truecaser::Apply(std::vector<std::string>* words) const {
  if (words->empty() || lm_.start < 0) return;
  // libsoda cases each word's text without the punctuation an earlier stage
  // attached to it (DecodedWord.word_parts), looked up in lower case. It
  // takes out the words the transducer's symbol table lacks, cases the rest
  // as one sentence and puts them back, so a word that is only punctuation
  // ("«") or unknown ("guillemets") is not part of the LM context.
  struct Parts {
    std::string before, core, after;
  };
  std::vector<Parts> parts;
  std::vector<size_t> kept;  // The words parts[] stands for.
  bool any = false;
  for (size_t i = 0; i < words->size(); ++i) {
    const std::string& w = (*words)[i];
    Parts p;
    size_t b = 0, e = w.size();
    for (bool more = true; more;) {
      more = false;
      for (const char* m : kAttachedMarks) {
        const size_t n = std::strlen(m);
        if (e - b >= n && w.compare(b, n, m) == 0) b += n, more = true;
        if (e - b > n && w.compare(e - n, n, m) == 0) e -= n, more = true;
      }
    }
    p.before = w.substr(0, b);
    p.core = w.substr(b, e - b);
    p.after = w.substr(e);
    if (variants_.count(Lower(p.core))) p.core = Lower(p.core);
    any |= variants_.count(p.core) > 0;
    if (p.core.empty() || !known_.count(Lower(p.core))) continue;
    parts.push_back(std::move(p));
    kept.push_back(i);
  }
  if (!any) return;

  // Viterbi over case variants; the DP state is the LM state.
  struct Node {
    int lm_state;
    float cost;
    int prev;     // Index into the previous position's nodes.
    int variant;  // Index into that position's candidates.
  };
  std::vector<std::vector<std::pair<int, std::string>>> candidates(parts.size());
  for (size_t i = 0; i < parts.size(); ++i) {
    const std::string& w = parts[i].core;
    if (auto it = variants_.find(w); it != variants_.end()) {
      candidates[i] = it->second;
    } else {
      auto id = lm_ids_.find(w);
      candidates[i].emplace_back(id == lm_ids_.end() ? unk_ : id->second, w);
    }
  }
  std::vector<std::vector<Node>> lattice(parts.size() + 1);
  // libsoda scores the first word without sentence-start context: from the
  // state the LM's start backs off to in the end (de-DE: <S> herr 8.3 beats
  // <S> Herr 11.7, yet libsoda writes "Herr"; unigram costs 10.9 / 10.2).
  int start = lm_.start;
  while (lm_.first[start] != lm_.first[start + 1] && lm_.arcs[lm_.first[start]].ilabel == 0) {
    start = lm_.arcs[lm_.first[start]].next;
  }
  lattice[0].push_back({start, 0.0f, -1, -1});
  std::vector<std::pair<float, int>> steps;
  for (size_t i = 0; i < parts.size(); ++i) {
    std::unordered_map<int, size_t> best;  // LM state -> node index.
    for (size_t p = 0; p < lattice[i].size(); ++p) {
      const Node& from = lattice[i][p];
      for (size_t c = 0; c < candidates[i].size(); ++c) {
        Step(from.lm_state, candidates[i][c].first, &steps);
        for (const auto& [step, next] : steps) {
          const float cost = from.cost + step;
          auto [it, inserted] = best.emplace(next, lattice[i + 1].size());
          if (inserted) {
            lattice[i + 1].push_back({next, cost, static_cast<int>(p), static_cast<int>(c)});
          } else if (cost < lattice[i + 1][it->second].cost) {
            lattice[i + 1][it->second] = {next, cost, static_cast<int>(p), static_cast<int>(c)};
          }
        }
      }
    }
  }
  const auto& last = lattice.back();
  size_t best_end = 0;
  float best_cost = std::numeric_limits<float>::infinity();
  for (size_t k = 0; k < last.size(); ++k) {
    const float cost = last[k].cost + FinalCost(last[k].lm_state);
    if (cost < best_cost) {
      best_cost = cost;
      best_end = k;
    }
  }
  int k = static_cast<int>(best_end);
  for (size_t i = parts.size(); i-- > 0;) {
    const Node& n = lattice[i + 1][k];
    (*words)[kept[i]] = parts[i].before + candidates[i][n.variant].second + parts[i].after;
    k = n.prev;
  }
}

}  // namespace soda
