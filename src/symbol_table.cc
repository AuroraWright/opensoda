#include "symbol_table.h"

#include <cstdint>
#include <cstring>

#include <marisa.h>

#include "config.h"

namespace soda {
namespace {

template <typename T>
bool Take(const std::string& buf, size_t* off, T* out) {
  if (*off + sizeof(T) > buf.size()) return false;
  std::memcpy(out, buf.data() + *off, sizeof(T));
  *off += sizeof(T);
  return true;
}

bool SkipSizedBlob(const std::string& buf, size_t* off) {
  int32_t n;
  if (!Take(buf, off, &n) || n < 0 || *off + n > buf.size()) return false;
  *off += n;
  return true;
}

}  // namespace

bool SymbolTable::Load(const std::string& path, std::string* error) {
  std::string buf;
  if (!ReadFile(path, &buf)) {
    *error = "cannot read " + path;
    return false;
  }
  size_t off = 0;
  int32_t version;
  uint64_t trie_size;
  if (!Take(buf, &off, &version) || !SkipSizedBlob(buf, &off) ||
      !SkipSizedBlob(buf, &off) || !Take(buf, &off, &trie_size)) {
    *error = "truncated symbol table header in " + path;
    return false;
  }
  off = (off + 15) & ~size_t{15};
  if (off + trie_size > buf.size()) {
    *error = "truncated symbol trie in " + path;
    return false;
  }
  marisa::Trie trie;
  try {
    trie.map(buf.data() + off, trie_size);
  } catch (const std::exception& e) {
    *error = std::string("bad symbol trie: ") + e.what();
    return false;
  }
  off += trie_size;
  uint64_t total, elem_size, data_bytes;
  bool ok = Take(buf, &off, &total);
  off = (off + 15) & ~size_t{15};  // The map itself is 16-byte aligned.
  if (!ok || !Take(buf, &off, &elem_size) ||
      !Take(buf, &off, &data_bytes) || (elem_size != 2 && elem_size != 4)) {
    *error = "bad symbol id map in " + path;
    return false;
  }
  const size_t n = trie.num_keys();
  if (off + n * elem_size > buf.size()) {
    *error = "truncated symbol id map in " + path;
    return false;
  }
  symbols_.assign(n, std::string());
  marisa::Agent agent;
  for (size_t id = 0; id < n; ++id) {
    uint32_t key_plus_one = 0;
    std::memcpy(&key_plus_one, buf.data() + off + id * elem_size, elem_size);
    if (key_plus_one == 0 || key_plus_one > n) continue;
    agent.set_query(key_plus_one - 1);
    trie.reverse_lookup(agent);
    symbols_[id].assign(agent.key().ptr(), agent.key().length());
  }
  return true;
}

int SymbolTable::Find(const std::string& symbol) const {
  for (int i = 0; i < size(); ++i) {
    if (symbols_[i] == symbol) return i;
  }
  return -1;
}

}  // namespace soda
