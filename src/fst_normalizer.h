// greco's FstNormalizer: rewrites text with a byte-level transducer from a
// language pack's denorm/*.mfar archive, such as porn_normalizer_ondevice.mfar
// (BAD_WORDS_STAR_RULE), which masks offensive words ("shit" -> "s***").
//
// An .mfar file is greco's memory-mapped FST archive: a list of records, each
// a NUL-terminated key padded to 4 bytes, a 64-bit length and the value
// aligned to 16 bytes. "v_" (2) is the version, "n_" the FST count, "name_<i>"
// and "f_<i>" each FST. An FST ("mfv_" 1) holds its start state ("0_"), the
// first arc of each state ("s_": count, then an offset per state) and its arcs
// ("c_": count, then 12-byte arcs). An arc is an input byte, an output byte
// (0 for either: none, so an arc reading none inserts its output), a
// tropical weight and the next state; next state -1 instead gives the
// state's final weight.

#ifndef SODA_FST_NORMALIZER_H_
#define SODA_FST_NORMALIZER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "text_normalizer.h"

namespace soda {

class FstNormalizer : public TextNormalizer {
 public:
  bool Init(const std::string& path, std::string* error);

  // The output of the cheapest path accepting `text`'s bytes, or `text`
  // itself if the transducer rejects it; an archive of several transducers
  // applies each in turn (fr-FR's emoji grammar inserts a marker, rewrites,
  // then deletes the marker). A ClassNormalizer's transducer instead
  // rewrites each span of words the recognizer tagged with a class.
  std::string Apply(const std::string& text) const override;

  // greco's ClassNormalizer (embedded_class_denorm.mfar): the transducer
  // only accepts a whole span such as "<money> sieben euro </money>"
  // ("7 €"), and Apply rewrites the text's spans one at a time.
  void set_class_spans(bool on) { class_spans_ = on; }

  // `token` is the opening or closing tag of one of greco's word classes;
  // sets `*name` to the class and `*closing`.
  static bool ClassTag(const std::string& token, std::string* name, bool* closing);

 private:
  struct Arc {
    uint8_t input;
    uint8_t output;
    float weight;
    int32_t next;
  };
  struct Fst {
    int start = 0;
    std::vector<uint32_t> first;  // Per state, its first arc; size states + 1.
    std::vector<Arc> arcs;        // Sorted by input within each state.
    std::vector<float> final;     // Infinity where a state is not final.
  };

  static std::string Rewrite(const Fst& fst, const std::string& text);
  std::string RewriteAll(const std::string& text) const;
  std::string RewriteSpans(const std::string& text) const;

  bool class_spans_ = false;
  std::vector<Fst> fsts_;  // The archive's transducers, in order.
};

}  // namespace soda

#endif  // SODA_FST_NORMALIZER_H_
