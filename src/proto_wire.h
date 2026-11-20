// Minimal protobuf wire-format reader and writer.
//
// The SODA language pack stores its configuration as binary protobufs, and the
// libsoda C API exchanges serialized protos with its callers. Both only need a
// handful of fields, so this avoids depending on libprotobuf.

#ifndef SODA_PROTO_WIRE_H_
#define SODA_PROTO_WIRE_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace soda {

enum WireType : int {
  kVarint = 0,
  kFixed64 = 1,
  kLengthDelimited = 2,
  kFixed32 = 5,
};

// Iterates over the fields of one serialized message.
class ProtoReader {
 public:
  explicit ProtoReader(std::string_view data)
      : p_(reinterpret_cast<const uint8_t*>(data.data())),
        end_(p_ + data.size()) {}

  // Advances to the next field. Returns false at the end of the message or on
  // malformed input (check ok() to tell the two apart).
  bool Next() {
    if (p_ >= end_) return false;
    uint64_t tag;
    if (!ReadVarint(&tag)) return Fail();
    field_ = static_cast<uint32_t>(tag >> 3);
    wire_type_ = static_cast<int>(tag & 7);
    switch (wire_type_) {
      case kVarint:
        if (!ReadVarint(&varint_)) return Fail();
        return true;
      case kFixed64:
        if (end_ - p_ < 8) return Fail();
        std::memcpy(&varint_, p_, 8);
        p_ += 8;
        return true;
      case kFixed32: {
        if (end_ - p_ < 4) return Fail();
        uint32_t v;
        std::memcpy(&v, p_, 4);
        varint_ = v;
        p_ += 4;
        return true;
      }
      case kLengthDelimited: {
        uint64_t len;
        if (!ReadVarint(&len) || len > static_cast<uint64_t>(end_ - p_)) {
          return Fail();
        }
        bytes_ = std::string_view(reinterpret_cast<const char*>(p_), len);
        p_ += len;
        return true;
      }
      default:
        return Fail();
    }
  }

  uint32_t field() const { return field_; }
  int wire_type() const { return wire_type_; }
  bool ok() const { return ok_; }

  uint64_t varint() const { return varint_; }
  int64_t int64() const { return static_cast<int64_t>(varint_); }
  bool boolean() const { return varint_ != 0; }
  float fixed_float() const {
    uint32_t bits = static_cast<uint32_t>(varint_);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  std::string_view bytes() const { return bytes_; }
  std::string str() const { return std::string(bytes_); }

  // Decodes a packed or single varint field into `out`.
  void AppendInts(std::vector<int>* out) const {
    if (wire_type_ == kVarint) {
      out->push_back(static_cast<int>(varint_));
    } else if (wire_type_ == kLengthDelimited) {
      ProtoReader packed(bytes_);
      uint64_t v;
      while (packed.p_ < packed.end_ && packed.ReadVarint(&v)) out->push_back(static_cast<int>(v));
    }
  }

  // Decodes a packed or single float field into `out`.
  void AppendFloats(std::vector<float>* out) const {
    if (wire_type_ == kFixed32) {
      out->push_back(fixed_float());
    } else if (wire_type_ == kLengthDelimited) {
      size_t n = bytes_.size() / 4;
      size_t old = out->size();
      out->resize(old + n);
      std::memcpy(out->data() + old, bytes_.data(), n * 4);
    }
  }

 private:
  bool ReadVarint(uint64_t* out) {
    uint64_t v = 0;
    for (int shift = 0; shift < 64 && p_ < end_; shift += 7) {
      uint8_t b = *p_++;
      v |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) {
        *out = v;
        return true;
      }
    }
    return false;
  }
  bool Fail() {
    ok_ = false;
    p_ = end_;
    return false;
  }

  const uint8_t* p_;
  const uint8_t* end_;
  uint32_t field_ = 0;
  int wire_type_ = 0;
  uint64_t varint_ = 0;
  std::string_view bytes_;
  bool ok_ = true;
};

// Appends fields to a serialized message.
class ProtoWriter {
 public:
  void Varint(uint32_t field, uint64_t v) {
    Tag(field, kVarint);
    PutVarint(v);
  }
  void Bool(uint32_t field, bool v) { Varint(field, v ? 1 : 0); }
  void Float(uint32_t field, float v) {
    Tag(field, kFixed32);
    char b[4];
    std::memcpy(b, &v, 4);
    out_.append(b, 4);
  }
  void Bytes(uint32_t field, std::string_view v) {
    Tag(field, kLengthDelimited);
    PutVarint(v.size());
    out_.append(v.data(), v.size());
  }
  void Message(uint32_t field, const ProtoWriter& m) { Bytes(field, m.out_); }

  const std::string& data() const { return out_; }

 private:
  void Tag(uint32_t field, int wt) { PutVarint((uint64_t{field} << 3) | wt); }
  void PutVarint(uint64_t v) {
    while (v >= 0x80) {
      out_.push_back(static_cast<char>((v & 0x7f) | 0x80));
      v >>= 7;
    }
    out_.push_back(static_cast<char>(v));
  }

  std::string out_;
};

}  // namespace soda

#endif  // SODA_PROTO_WIRE_H_
