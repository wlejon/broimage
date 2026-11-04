#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

struct RasterWindow {
    int x = 0;
    int y = 0;
    int width = 0;   // 0 = full width from x
    int height = 0;  // 0 = full height from y
};

struct GeoTiffInfo {
    int width = 0;
    int height = 0;
    int channels = 1;
    int bits_per_sample = 0;
    int sample_format = 1; // 1 = uint, 2 = int, 3 = float
    int tile_width = 0;    // 0 if stripped
    int tile_height = 0;   // rows_per_strip if stripped
    bool is_tiled = false;
    bool is_bigtiff = false;
    bool has_pixel_scale = false;
    double pixel_scale[3] = {0.0, 0.0, 0.0};
    std::vector<double> tiepoints; // groups of 6: (i, j, k, x, y, z)
    std::vector<double> model_transform; // 16 doubles if ModelTransformationTag (34264) present
    bool has_nodata = false;
    double nodata = 0.0;
    int epsg = 0; // ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048) if present
};

struct RasterF32 {
    int width = 0;
    int height = 0;
    int channels = 1;
    GeoTiffInfo info;
    std::vector<float> samples; // width * height * channels (native float values, e.g. meters)
    bool empty() const { return samples.empty(); }
};

bool read_geotiff_info_memory(const uint8_t* data, std::size_t size, GeoTiffInfo& out, std::string* error = nullptr);
bool read_geotiff_info_file(const std::string& path, GeoTiffInfo& out, std::string* error = nullptr);

bool decode_geotiff_memory_f32(const uint8_t* data, std::size_t size, RasterF32& out,
                               const RasterWindow& window = {}, std::string* error = nullptr);
bool decode_geotiff_file_f32(const std::string& path, RasterF32& out,
                             const RasterWindow& window = {}, std::string* error = nullptr);

} // namespace broimage
