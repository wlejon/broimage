// JPEG decode benchmark: stb_image against libjpeg-turbo at full size and
// each DCT scale, the thumbnail and screen-sized scaled decodes, the EXIF
// thumbnail, and how far turbo's pixels are from stb's.
//
//   broimage_bench_jpeg <file-or-directory>... [--runs N] [--thumb PX] [--screen WxH]
//
// Not a test (it needs photos, which are not in the repo); times are the
// median of N runs, in milliseconds.

#include <broimage/decode.h>
#include <broimage/geometric.h>

#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_runs = 3;

double median_ms(const std::function<void()>& fn) {
    std::vector<double> t;
    for (int i = 0; i < g_runs; ++i) {
        const auto a = std::chrono::steady_clock::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

bool is_jpeg_name(const fs::path& p) {
    std::string e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return e == ".jpg" || e == ".jpeg" || e == ".cr3";
}

const char* path_name(broimage::ScaledDecodePath p) {
    switch (p) {
        case broimage::ScaledDecodePath::Full: return "full";
        case broimage::ScaledDecodePath::Dct: return "dct";
        case broimage::ScaledDecodePath::ExifThumbnail: return "exif";
        case broimage::ScaledDecodePath::HeifThumbnail: return "heif";
        case broimage::ScaledDecodePath::RawPreview: return "raw-preview";
    }
    return "?";
}

// A CR3: what its embedded JPEGs cost (no stb or per-scale columns: the RAW
// is never decoded).
void bench_cr3(const fs::path& path, const std::vector<uint8_t>& bytes, int thumb, int screen_w, int screen_h) {
    broimage::Image img;
    const double t_full = median_ms([&] { broimage::decode_memory_oriented(bytes.data(), bytes.size(), img); });
    const int fw = img.width, fh = img.height;
    broimage::ScaledDecodeOptions o;
    broimage::ScaledDecodeInfo ti, si;
    o.width = o.height = thumb;
    const double t_thumb = median_ms([&] { broimage::decode_memory_scaled(bytes.data(), bytes.size(), o, img, nullptr, &ti); });
    const int tw = img.width, th = img.height;
    o.width = screen_w;
    o.height = screen_h;
    const double t_screen = median_ms([&] { broimage::decode_memory_scaled(bytes.data(), bytes.size(), o, img, nullptr, &si); });
    const double t_file = median_ms([&] {
        broimage::ScaledDecodeOptions f;
        f.width = f.height = thumb;
        broimage::Image x;
        broimage::decode_file_scaled(path.string(), f, x);
    });
    std::printf("%-14s %7.1fMB | full-size JPEG %dx%d %.1f | thumb %dx%d %.1f (%s %d/8) | screen %dx%d %.1f (%s %d/8) "
                "| thumb from file %.1f\n",
                path.filename().string().c_str(), bytes.size() / 1048576.0, fw, fh, t_full, tw, th, t_thumb,
                path_name(ti.path), ti.dct_scale, img.width, img.height, t_screen, path_name(si.path), si.dct_scale,
                t_file);
}

struct Sum {
    double v[10] = {};
    int n = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<fs::path> files;
    int thumb = 256, screen_w = 2560, screen_h = 1440;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) g_runs = std::max(1, std::atoi(argv[++i]));
        else if (a == "--thumb" && i + 1 < argc) thumb = std::atoi(argv[++i]);
        else if (a == "--screen" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &screen_w, &screen_h);
        else if (fs::is_directory(a)) {
            for (const auto& e : fs::directory_iterator(a))
                if (e.is_regular_file() && is_jpeg_name(e.path())) files.push_back(e.path());
        } else files.emplace_back(a);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "usage: broimage_bench_jpeg <file-or-dir>... [--runs N] [--thumb PX] [--screen WxH]\n");
        return 2;
    }

    std::printf("%-14s %9s | %8s %8s %8s %8s %8s | %8s %8s | %7s %7s | %7s %8s %6s\n", "file", "size",
                "stb", "turbo", "1/2", "1/4", "1/8", "thumb", "stbthumb", "screen", "exif", "maxdiff", "meandiff",
                ">4");
    Sum sum;
    std::vector<fs::path> cr3_rows;
    for (const fs::path& path : files) {
        std::ifstream f(path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.empty()) continue;
        if (bytes.size() >= 12 && std::memcmp(bytes.data() + 4, "ftypcrx ", 8) == 0) {
            cr3_rows.push_back(path);
            continue;
        }
        const uint8_t* d = bytes.data();
        const int n = int(bytes.size());

        int w = 0, h = 0, c = 0;
        stbi_uc* stb_px = nullptr;
        const double t_stb = median_ms([&] {
            if (stb_px) stbi_image_free(stb_px);
            stb_px = stbi_load_from_memory(d, n, &w, &h, &c, 4);
        });
        broimage::Image full;
        const double t_full = median_ms([&] { broimage::decode_memory(d, bytes.size(), full); });

        double t_scale[3];
        const int scales[3] = {4, 2, 1};
        for (int k = 0; k < 3; ++k) {
            broimage::ScaledDecodeOptions o;
            o.width = (w * scales[k] + 7) / 8;
            o.height = (h * scales[k] + 7) / 8;
            o.embedded_thumbnail = false;
            broimage::Image img;
            // The box is the stored size at that scale; orientation may swap it,
            // so ask with both sides and Cover to land on that scale exactly.
            o.width = std::min(o.width, o.height);
            o.height = o.width;
            o.fit = broimage::ScaleFit::Cover;
            broimage::ScaledDecodeInfo info;
            t_scale[k] = median_ms([&] { broimage::decode_memory_scaled(d, bytes.size(), o, img, nullptr, &info); });
            if (info.path != broimage::ScaledDecodePath::Dct || info.dct_scale != scales[k])
                std::fprintf(stderr, "%s: asked for %d/8, got path %d scale %d\n", path.filename().string().c_str(),
                             scales[k], int(info.path), info.dct_scale);
        }

        broimage::ScaledDecodeOptions to;
        to.width = to.height = thumb;
        broimage::Image th;
        const double t_thumb = median_ms([&] { broimage::decode_memory_scaled(d, bytes.size(), to, th); });

        // The way a thumbnail was made before: stb at full size, upright, area-resized.
        const double t_stbthumb = median_ms([&] {
            int sw = 0, sh = 0, sc = 0;
            stbi_uc* px = stbi_load_from_memory(d, n, &sw, &sh, &sc, 4);
            broimage::Image img;
            img.width = sw;
            img.height = sh;
            img.pixels.assign(px, px + std::size_t(sw) * sh * 4);
            stbi_image_free(px);
            broimage::apply_exif_orientation(img, broimage::read_exif_orientation(d, bytes.size()));
            const double s = std::min(double(thumb) / img.width, double(thumb) / img.height);
            const int tw = std::max(1, int(img.width * s + 0.5)), tht = std::max(1, int(img.height * s + 0.5));
            std::vector<uint8_t> out(std::size_t(tw) * tht * 4);
            broimage::resize_hwc_u8(img.pixels.data(), img.width, img.height, 4, out.data(), tw, tht,
                                    broimage::Filter::Area);
        });

        broimage::ScaledDecodeOptions so;
        so.width = screen_w;
        so.height = screen_h;
        broimage::Image scr;
        const double t_screen = median_ms([&] { broimage::decode_memory_scaled(d, bytes.size(), so, scr); });

        broimage::Image ex;
        double t_exif = median_ms([&] { broimage::decode_exif_thumbnail(d, bytes.size(), ex); });
        if (ex.empty()) t_exif = -1;

        int maxdiff = 0;
        double meandiff = 0, over4 = 0;
        if (stb_px && full.width == w && full.height == h) {
            const std::size_t count = std::size_t(w) * h * 4;
            uint64_t total = 0, big = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const int dd = std::abs(int(stb_px[i]) - int(full.pixels[i]));
                maxdiff = std::max(maxdiff, dd);
                total += uint64_t(dd);
                big += dd > 4;
            }
            meandiff = double(total) / double(count);
            over4 = 100.0 * double(big) / double(count);
        } else {
            maxdiff = -1;
        }
        if (stb_px) stbi_image_free(stb_px);

        std::printf("%-14s %7.1fMB | %8.1f %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f | %7.1f %7.2f | %7d %8.3f %5.2f%%\n",
                    path.filename().string().c_str(), bytes.size() / 1048576.0, t_stb, t_full, t_scale[0],
                    t_scale[1], t_scale[2], t_thumb, t_stbthumb, t_screen, t_exif, maxdiff, meandiff, over4);
        const double row[10] = {t_stb, t_full, t_scale[0], t_scale[1], t_scale[2], t_thumb, t_stbthumb, t_screen,
                                t_exif, 0};
        for (int k = 0; k < 10; ++k) sum.v[k] += row[k];
        ++sum.n;
        std::fflush(stdout);
    }
    if (sum.n > 1) {
        const double k = 1.0 / sum.n;
        std::printf("%-14s %9s | %8.1f %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f | %7.1f %7.2f |\n", "mean", "",
                    sum.v[0] * k, sum.v[1] * k, sum.v[2] * k, sum.v[3] * k, sum.v[4] * k, sum.v[5] * k, sum.v[6] * k,
                    sum.v[7] * k, sum.v[8] * k);
    }
    for (const fs::path& path : cr3_rows) {
        std::ifstream f(path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        bench_cr3(path, bytes, thumb, screen_w, screen_h);
    }
    return 0;
}
