#pragma once
// Internal: a file's bytes (or its first `max` bytes).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace broimage::detail {

// `path` is UTF-8 (what bro and brothumb pass), opened on Windows through the
// wide API rather than the ANSI code page, so a photo under "Été" opens.
inline bool read_file_bytes(const std::string& path, std::vector<uint8_t>& out, std::size_t max = ~std::size_t(0)) {
    out.clear();
    // A directory opens for reading on Linux and seeks to an end of LONG_MAX,
    // which was a bad_alloc for decode('').
    const std::filesystem::path fsPath(
        std::u8string_view(reinterpret_cast<const char8_t*>(path.data()), path.size()));
    std::error_code ec;
    if (!std::filesystem::is_regular_file(fsPath, ec)) return false;
#ifdef _MSC_VER
    std::FILE* f = nullptr;
    if (_wfopen_s(&f, fsPath.c_str(), L"rb") != 0) f = nullptr;
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) return false;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long len = ok ? std::ftell(f) : -1;
    ok = len >= 0 && std::fseek(f, 0, SEEK_SET) == 0;
    if (ok) {
        std::size_t n = std::size_t(len);
        if (n > max) n = max;
        out.resize(n);
        out.resize(std::fread(out.data(), 1, n, f));
    }
    std::fclose(f);
    return ok;
}

} // namespace broimage::detail
