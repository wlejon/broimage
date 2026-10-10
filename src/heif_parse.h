// The HEIF container (ISO/IEC 23008-12), read from the ISO base media file
// format boxes it is made of (ISO/IEC 14496-12), as MIAF (ISO/IEC 23000-22)
// and AVIF (AV1 Image File Format) profile it. Written from those specs.
//
// heif_parse() reads `ftyp`, the `meta` box (hdlr, pitm, iinf/infe, iloc,
// iref, iprp/ipco/ipma, idat) and, for an image sequence, the first video
// track of `moov`. It keeps pointers into the caller's bytes: a HeifFile is
// valid while they are. Nothing is decoded here; decode_heif.cpp turns
// items into pixels.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace broimage::detail {

using FourCC = std::array<char, 4>;

inline bool fourcc_is(const FourCC& f, const char* s) { return std::memcmp(f.data(), s, 4) == 0; }

struct HeifExtent {
    uint64_t offset = 0;
    uint64_t length = 0;  // 0: to the end of the file (construction method 0) or of idat (1)
};

struct HeifPropRef {
    uint16_t index = 0;   // 1-based into HeifFile::props
    bool essential = false;
};

struct HeifItem {
    uint32_t id = 0;
    FourCC type{};          // 'av01', 'hvc1', 'grid', 'iovl', 'Exif', 'mime', 'jpeg', ...
    bool hidden = false;    // infe flags bit 0
    std::string name;
    std::string content_type;  // for 'mime' items
    // Where its bytes are (iloc). Construction method 0: offsets into the
    // file; 1: into idat; 2 (item offset) is not supported.
    bool located = false;
    int construction_method = 0;
    uint16_t data_reference_index = 0;  // 0: this file
    uint64_t base_offset = 0;
    std::vector<HeifExtent> extents;
    std::vector<HeifPropRef> props;     // ipma, in association order
};

struct HeifRef {
    FourCC type{};  // 'dimg', 'auxl', 'thmb', 'cdsc', 'prem', ...
    uint32_t from = 0;
    std::vector<uint32_t> to;
};

struct HeifProp {
    FourCC type{};
    const uint8_t* body = nullptr;  // the box payload, after its header
    std::size_t size = 0;
};

// The first video track of an image sequence ('avis', 'msf1'): enough to
// read its first sample, a sync sample, as a still image.
struct HeifTrack {
    bool present = false;
    FourCC handler{};     // 'pict' or 'vide'
    FourCC codec{};       // sample entry type: 'av01', 'hvc1', ...
    uint32_t width = 0, height = 0;
    const uint8_t* config = nullptr;  // the av1C / hvcC payload in the sample entry
    std::size_t config_size = 0;
    FourCC config_type{};
    uint64_t first_sample_offset = 0;
    uint64_t first_sample_size = 0;
    uint32_t sample_count = 0;
};

struct HeifFile {
    const uint8_t* data = nullptr;
    std::size_t size = 0;

    FourCC major_brand{};
    std::vector<FourCC> compatible_brands;

    bool has_meta = false;
    uint32_t primary = 0;  // pitm; 0 when absent
    std::vector<HeifItem> items;
    std::vector<HeifRef> refs;
    std::vector<HeifProp> props;
    const uint8_t* idat = nullptr;
    std::size_t idat_size = 0;

    HeifTrack track;

    bool has_brand(const char* b) const;
    const HeifItem* item(uint32_t id) const;
    // The item's first associated property of `type`, or null.
    const HeifProp* prop(const HeifItem& it, const char* type) const;
    // Items `from` references with a reference of `type`, in order.
    std::vector<uint32_t> refs_from(uint32_t from, const char* type) const;
    // Items with a reference of `type` to `to`.
    std::vector<uint32_t> refs_to(uint32_t to, const char* type) const;
    // The item's bytes, its extents concatenated. False (and `why`) when
    // they lie outside the data or use an unsupported construction.
    bool item_data(const HeifItem& it, std::vector<uint8_t>& out, std::string* why = nullptr) const;
    // The item's bytes in place, when they are one contiguous range.
    bool item_span(const HeifItem& it, const uint8_t*& p, std::size_t& n) const;
};

// `ftyp` first, with a brand of a HEIF image or image sequence.
bool heif_sniff(const uint8_t* data, std::size_t size);

// Read the structure. False (and `why`) when `data` is not a HEIF file or
// its meta box is malformed.
bool heif_parse(const uint8_t* data, std::size_t size, HeifFile& out, std::string* why = nullptr);

// ---- Properties, decoded --------------------------------------------------

struct HeifIspe { uint32_t width = 0, height = 0; };

struct HeifNclx {
    bool present = false;
    uint16_t primaries = 2, transfer = 2, matrix = 2;  // 2: unspecified
    bool full_range = false;
};

struct HeifClap {
    // Fractions as stored: width/height and the centre offsets.
    uint32_t width_n = 0, width_d = 1, height_n = 0, height_d = 1;
    int32_t horiz_off_n = 0;
    uint32_t horiz_off_d = 1;
    int32_t vert_off_n = 0;
    uint32_t vert_off_d = 1;
};

bool read_ispe(const HeifProp& p, HeifIspe& out);
bool read_nclx(const HeifProp& p, HeifNclx& out);       // a 'colr' of type nclx
bool read_icc(const HeifProp& p, const uint8_t*& icc, std::size_t& n);  // 'colr' prof / rICC
bool read_irot(const HeifProp& p, int& angle);          // 0..3, anti-clockwise quarter turns
bool read_imir(const HeifProp& p, int& mode);           // 0: top-to-bottom, 1: left-to-right
bool read_clap(const HeifProp& p, HeifClap& out);
bool read_auxc(const HeifProp& p, std::string& aux_type);
bool read_pixi(const HeifProp& p, std::vector<uint8_t>& bits);
bool read_clli(const HeifProp& p, uint16_t& max_cll, uint16_t& max_fall);

// The clean aperture as a pixel rectangle of a w x h image; false when it
// does not fit (the property is then ignored).
bool clap_rect(const HeifClap& c, int w, int h, int& x, int& y, int& cw, int& ch);

}  // namespace broimage::detail
