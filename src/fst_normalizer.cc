#include "fst_normalizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

#include "config.h"

namespace soda {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// The records of an archive or FST between `begin` and `end`: key -> value.
bool ReadRecords(const std::string& data, size_t begin, size_t end,
                 std::map<std::string, std::pair<size_t, size_t>>* out) {
  size_t i = begin;
  while (i < end) {
    const size_t nul = data.find('\0', i);
    if (nul == std::string::npos || nul >= end) return false;
    const std::string key = data.substr(i, nul - i);
    i = (nul + 1 + 3) / 4 * 4;
    uint64_t n;
    if (i + sizeof(n) > end) return false;
    std::memcpy(&n, data.data() + i, sizeof(n));
    i = (i + sizeof(n) + 15) / 16 * 16;
    if (n > end - std::min(i, end)) return false;
    (*out)[key] = {i, static_cast<size_t>(n)};
    i = (i + n + 15) / 16 * 16;
  }
  return true;
}

template <typename T>
T Get(const std::string& data, size_t offset) {
  T v;
  std::memcpy(&v, data.data() + offset, sizeof(v));
  return v;
}

}  // namespace

bool FstNormalizer::Init(const std::string& path, std::string* error) {
  std::string data;
  if (!ReadFile(path, &data)) {
    *error = "cannot read " + path;
    return false;
  }
  std::map<std::string, std::pair<size_t, size_t>> archive;
  auto bad = [&](const char* what) {
    *error = path + ": " + what;
    return false;
  };
  if (!ReadRecords(data, 0, data.size(), &archive)) return bad("not an mfar archive");
  auto version = archive.find("v_");
  if (version == archive.end() || version->second.second < 4 ||
      Get<uint32_t>(data, version->second.first) != 2 || archive.find("f_0") == archive.end()) {
    return bad("unsupported mfar archive");
  }
  fsts_.clear();
  for (int i = 0;; ++i) {
    auto f = archive.find("f_" + std::to_string(i));
    if (f == archive.end()) break;
    std::map<std::string, std::pair<size_t, size_t>> fst;
    const auto [fst_begin, fst_size] = f->second;
    if (!ReadRecords(data, fst_begin, fst_begin + fst_size, &fst)) return bad("bad FST");
    for (const char* key : {"mfv_", "0_", "s_", "c_"}) {
      if (fst.find(key) == fst.end() || fst[key].second < 4) return bad("incomplete FST");
    }
    if (Get<uint32_t>(data, fst["mfv_"].first) != 1) return bad("unsupported FST version");
    Fst& out = fsts_.emplace_back();
    out.start = Get<int32_t>(data, fst["0_"].first);

    const auto [states_at, states_size] = fst["s_"];
    const uint32_t num_states = Get<uint32_t>(data, states_at);
    const auto [arcs_at, arcs_size] = fst["c_"];
    const uint32_t num_arcs = Get<uint32_t>(data, arcs_at);
    if (states_size < 16 + 4 * static_cast<size_t>(num_states) ||
        arcs_size < 16 + 12 * static_cast<size_t>(num_arcs) || out.start < 0 ||
        static_cast<uint32_t>(out.start) >= num_states) {
      return bad("truncated FST");
    }
    std::vector<uint32_t> first(num_states);
    std::memcpy(first.data(), data.data() + states_at + 16, 4 * first.size());
    first.push_back(num_arcs);
    if (first.back() > num_arcs || !std::is_sorted(first.begin(), first.end())) {
      return bad("bad state table");
    }

    // Final weights move out of the arc list.
    out.first.assign(1, 0);
    out.final.assign(num_states, kInfinity);
    for (uint32_t s = 0; s < num_states; ++s) {
      const size_t begin = out.arcs.size();
      for (uint32_t a = first[s]; a < first[s + 1]; ++a) {
        const size_t at = arcs_at + 16 + 12 * static_cast<size_t>(a);
        const uint32_t labels = Get<uint32_t>(data, at);
        const float weight = Get<float>(data, at + 4);
        const int32_t next = Get<int32_t>(data, at + 8);
        if (next < 0) {
          out.final[s] = weight;
          continue;
        }
        if (static_cast<uint32_t>(next) >= num_states) return bad("arc to a missing state");
        out.arcs.push_back({static_cast<uint8_t>(labels & 0xff),
                            static_cast<uint8_t>((labels >> 8) & 0xff), weight, next});
      }
      std::stable_sort(out.arcs.begin() + begin, out.arcs.end(),
                       [](const Arc& a, const Arc& b) { return a.input < b.input; });
      out.first.push_back(static_cast<uint32_t>(out.arcs.size()));
    }
  }
  return true;
}

// The classes greco's ClassNormalizer knows (class_normalizer.cc), whose
// tags it recognizes as "<money>" and "</money>".
constexpr const char* kClasses[] = {
    "addressnum", "alphanumseq", "app",     "apt",          "areacode",  "contacts",
    "countrycode", "day",        "fulldate", "fullphonenum", "grammars", "hotword",
    "money",      "month",       "operand", "ordinal",      "percent",   "personal",
    "phonenum",   "postalcode",  "recency", "reject",       "restaurant", "root",
    "song",       "street",      "time",    "unknown",      "year",      "dynamicclass"};

bool FstNormalizer::ClassTag(const std::string& token, std::string* name, bool* closing) {
  if (token.size() < 3 || token.front() != '<' || token.back() != '>') return false;
  *closing = token[1] == '/';
  *name = token.substr(*closing ? 2 : 1, token.size() - (*closing ? 3 : 2));
  for (const char* c : kClasses) {
    if (*name == c) return true;
  }
  return false;
}

std::string FstNormalizer::Apply(const std::string& text) const {
  return class_spans_ ? RewriteSpans(text) : RewriteAll(text);
}

std::string FstNormalizer::RewriteAll(const std::string& text) const {
  std::string out = text;
  for (const Fst& fst : fsts_) out = Rewrite(fst, out);
  return out;
}

// libsoda hands the transducer each span from an opening tag to the next
// closing tag of its class, tags included, and puts the output in its place.
// A tag of another class inside the span, a missing closing tag or a span
// the transducer rejects lose only the opening tag; the words go on, and a
// stray closing tag is dropped too.
std::string FstNormalizer::RewriteSpans(const std::string& text) const {
  std::vector<std::string> tokens;
  for (size_t i = 0, j; i <= text.size(); i = j + 1) {
    j = std::min(text.find(' ', i), text.size());
    if (j > i) tokens.push_back(text.substr(i, j - i));
  }
  std::string out;
  auto append = [&out](const std::string& t) {
    if (!t.empty()) out += (out.empty() ? "" : " ") + t;
  };
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string name, other;
    bool closing;
    if (!ClassTag(tokens[i], &name, &closing)) {
      append(tokens[i]);
      continue;
    }
    if (closing) continue;
    size_t end = 0;
    for (size_t k = i + 1; k < tokens.size(); ++k) {
      bool other_closing;
      if (!ClassTag(tokens[k], &other, &other_closing)) continue;
      if (other != name) break;
      if (other_closing) {
        end = k;
        break;
      }
    }
    if (end == 0) continue;
    std::string span = tokens[i];
    for (size_t k = i + 1; k <= end; ++k) span += " " + tokens[k];
    const std::string rewritten = RewriteAll(span);
    if (rewritten == span || rewritten.empty()) continue;
    append(rewritten);
    i = end;
  }
  return out;
}

std::string FstNormalizer::Rewrite(const Fst& fst, const std::string& text) {
  if (fst.arcs.empty()) return text;
  // Viterbi over (byte position, state); each node remembers its predecessor
  // (at the same position after an arc reading no byte) and the byte it
  // wrote.
  struct Node {
    int state;
    float cost;
    int prev_pos;
    int prev;  // Index into nodes[prev_pos].
    uint8_t output;
  };
  std::vector<std::vector<Node>> nodes(text.size() + 1);
  std::vector<int> slot(fst.final.size(), -1);
  auto relax = [&](std::vector<Node>& at, const Node& n) {
    int& k = slot[n.state];
    if (k < 0) {
      k = static_cast<int>(at.size());
      at.push_back(n);
      return true;
    }
    if (n.cost < at[k].cost) {
      at[k] = n;
      return true;
    }
    return false;
  };
  // Arcs reading no byte (input 0, sorted first) insert their output.
  auto close = [&](int pos) {
    std::vector<Node>& at = nodes[pos];
    for (size_t n = 0; n < at.size(); ++n) slot[at[n].state] = static_cast<int>(n);
    std::vector<int> queue(at.size());
    for (size_t n = 0; n < queue.size(); ++n) queue[n] = static_cast<int>(n);
    for (size_t q = 0; q < queue.size() && q < 64 * fst.final.size(); ++q) {
      const Node from = at[queue[q]];
      for (auto a = fst.arcs.begin() + fst.first[from.state];
           a != fst.arcs.begin() + fst.first[from.state + 1] && a->input == 0; ++a) {
        if (relax(at, {a->next, from.cost + a->weight, pos, queue[q], a->output})) {
          queue.push_back(slot[a->next]);
        }
      }
    }
    for (const Node& n : at) slot[n.state] = -1;
  };
  nodes[0].push_back({fst.start, 0.0f, -1, -1, 0});
  close(0);
  for (size_t i = 0; i < text.size(); ++i) {
    const uint8_t c = static_cast<uint8_t>(text[i]);
    std::vector<Node>& next = nodes[i + 1];
    for (size_t n = 0; n < nodes[i].size(); ++n) {
      const Node& from = nodes[i][n];
      auto lo = std::lower_bound(fst.arcs.begin() + fst.first[from.state],
                                 fst.arcs.begin() + fst.first[from.state + 1], c,
                                 [](const Arc& a, uint8_t v) { return a.input < v; });
      for (auto a = lo; a != fst.arcs.begin() + fst.first[from.state + 1] && a->input == c; ++a) {
        relax(next, {a->next, from.cost + a->weight, static_cast<int>(i), static_cast<int>(n),
                     a->output});
      }
    }
    for (const Node& n : next) slot[n.state] = -1;
    if (next.empty()) return text;
    close(static_cast<int>(i + 1));
  }
  int best = -1;
  float best_cost = kInfinity;
  const std::vector<Node>& last = nodes.back();
  for (size_t n = 0; n < last.size(); ++n) {
    const float cost = last[n].cost + fst.final[last[n].state];
    if (cost < best_cost) {
      best_cost = cost;
      best = static_cast<int>(n);
    }
  }
  if (best < 0) return text;
  std::string out;
  for (int pos = static_cast<int>(text.size()); best >= 0;) {
    const Node& n = nodes[pos][best];
    if (n.output != 0) out.push_back(static_cast<char>(n.output));
    pos = n.prev_pos;
    best = n.prev;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace soda
