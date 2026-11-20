#include "biasing.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>

#include "proto_wire.h"

namespace soda {
namespace {

constexpr char kWordStart[] = "\xe2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK

std::vector<std::string> SplitUtf8(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    n = std::min(n, s.size() - i);
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

std::vector<std::string> SplitWords(const std::string& s) {
  std::vector<std::string> words;
  std::istringstream in(s);
  for (std::string w; in >> w;) words.push_back(w);
  return words;
}

}  // namespace

bool WordPieceModel::Load(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *error = "cannot read " + path;
    return false;
  }
  const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  // 3: {1: id, 2: piece, 3: count}; 4: {1: left id, 2: right id, 3: merged id}.
  ProtoReader r(data);
  while (r.Next()) {
    if (r.field() != 3 && r.field() != 4) continue;
    ProtoReader m(r.bytes());
    int a = 0, b = 0, c = 0;
    std::string piece;
    while (m.Next()) {
      if (m.field() == 1) a = static_cast<int>(m.int64());
      if (m.field() == 2 && r.field() == 3) piece = m.str();
      if (m.field() == 2 && r.field() == 4) b = static_cast<int>(m.int64());
      if (m.field() == 3 && r.field() == 4) c = static_cast<int>(m.int64());
    }
    if (r.field() == 3) {
      if (a < 0 || piece.empty()) continue;
      if (static_cast<size_t>(a) >= pieces_.size()) pieces_.resize(a + 1);
      pieces_[a] = piece;
      ids_[piece] = a;
    } else {
      merges_[{a, b}] = c;
    }
  }
  if (!r.ok() || ids_.empty()) {
    *error = "malformed word-piece model " + path;
    return false;
  }
  return true;
}

std::vector<std::string> WordPieceModel::Encode(const std::string& word) const {
  std::vector<int> ids;
  auto add = [&](const std::string& s) {
    auto it = ids_.find(s);
    if (it == ids_.end()) return false;
    ids.push_back(it->second);
    return true;
  };
  if (!add(kWordStart)) return {};
  for (const std::string& ch : SplitUtf8(word)) {
    if (!add(ch)) return {};
  }
  // Apply the earliest merge among adjacent pairs until none applies.
  for (;;) {
    size_t at = 0;
    int best = -1;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
      auto it = merges_.find({ids[i], ids[i + 1]});
      if (it != merges_.end() && (best < 0 || it->second < best)) {
        best = it->second;
        at = i;
      }
    }
    if (best < 0) break;
    ids[at] = best;
    ids.erase(ids.begin() + at + 1);
  }
  std::vector<std::string> out;
  for (int id : ids) out.push_back(pieces_[id]);
  return out;
}

void ContextBiaser::Build(const std::vector<BiasPhrase>& phrases, const WordPieceModel& wpm,
                          const SymbolTable& symbols, const Params& params) {
  nodes_.assign(1, Node());
  retwiddle_penalty_ = params.retwiddle_penalty;
  auto add = [&](const std::string& text, float weight) {
    std::vector<int> labels;
    for (const std::string& word : SplitWords(text)) {
      std::vector<std::string> pieces = wpm.Encode(word);
      if (pieces.empty()) return;
      for (const std::string& p : pieces) {
        const int label = symbols.Find(p);
        if (label < 0) return;
        labels.push_back(label);
      }
    }
    if (labels.empty()) return;
    int node = kStart;
    for (int label : labels) {
      auto it = nodes_[node].children.find(label);
      const float reward = nodes_[node].reward + weight;
      if (it == nodes_[node].children.end()) {
        nodes_.push_back(Node());
        nodes_.back().reward = reward;
        it = nodes_[node].children.emplace(label, static_cast<int>(nodes_.size()) - 1).first;
      } else {
        // Determinized in the tropical semiring: the smaller reward first.
        nodes_[it->second].reward = std::min(nodes_[it->second].reward, reward);
      }
      node = it->second;
    }
    nodes_[node].final = true;
  };
  for (const BiasPhrase& p : phrases) {
    // libsoda's twiddler token generator gives every piece the phrase's
    // boost plus grammar_token_base_weight (BOOST_STRATEGY_PER_TOKEN_CONSTANT
    // with a uniform distribution).
    const float weight = params.token_weight + (params.respect_boost ? p.boost : 1.0f);
    if (!(weight > 0.0f)) continue;
    add(p.text, weight);
  }
  // Failure links, breadth first (Aho-Corasick).
  std::deque<int> queue;
  for (const auto& [label, child] : nodes_[kStart].children) {
    nodes_[child].fail = kStart;
    queue.push_back(child);
  }
  while (!queue.empty()) {
    const int node = queue.front();
    queue.pop_front();
    for (const auto& [label, child] : nodes_[node].children) {
      int f = nodes_[node].fail;
      for (;;) {
        auto it = nodes_[f].children.find(label);
        if (it != nodes_[f].children.end() && it->second != child) {
          nodes_[child].fail = it->second;
          break;
        }
        if (f == kStart) {
          nodes_[child].fail = kStart;
          break;
        }
        f = nodes_[f].fail;
      }
      queue.push_back(child);
    }
  }
}

float ContextBiaser::Step(int state, int label, int* next) const {
  float reward = 0.0f;
  for (;;) {
    const Node& n = nodes_[state];
    auto it = n.children.find(label);
    if (it != n.children.end()) {
      *next = it->second;
      return -(reward + nodes_[it->second].reward - n.reward);
    }
    if (state == kStart) {
      *next = kStart;
      return -reward;
    }
    if (n.final) {
      // A completed phrase keeps its reward; going on costs the penalty.
      reward += retwiddle_penalty_;
      state = kStart;
    } else {
      reward += nodes_[n.fail].reward - n.reward;
      state = n.fail;
    }
  }
}

}  // namespace soda
