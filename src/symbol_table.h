// Reader for SODA's CompactSymbolTable files (*.syms.compact).

#ifndef SODA_SYMBOL_TABLE_H_
#define SODA_SYMBOL_TABLE_H_

#include <string>
#include <vector>

namespace soda {

class SymbolTable {
 public:
  // Layout (little endian):
  //   int32 version, int32 n + n bytes source path,
  //   int32 n + n bytes fingerprint,
  //   uint64 trie_size, MARISA trie (16-byte aligned, trie_size bytes),
  //   uint64 total, uint64 elem_size, uint64 data_bytes,
  //   uint{16,32}[num_keys], where entry[symbol_id] = trie_key_id + 1.
  bool Load(const std::string& path, std::string* error);

  int size() const { return static_cast<int>(symbols_.size()); }
  const std::string& Symbol(int id) const { return symbols_[id]; }
  // Returns -1 if the symbol is unknown.
  int Find(const std::string& symbol) const;

 private:
  std::vector<std::string> symbols_;
};

}  // namespace soda

#endif  // SODA_SYMBOL_TABLE_H_
