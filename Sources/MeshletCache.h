#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "Shared/SharedTypes.h"

// Meshlet binary cache (.meshlet.bin) for persisting Phase 2 results.
// Cache path convention: <gltf_path_without_extension>_<primitive_index>.meshlet.bin

namespace MeshletCache {

static constexpr uint32_t MAGIC   = 0x4D534854; // 'MSHT'
static constexpr uint32_t VERSION = 1;

struct Header {
    uint32_t magic         = MAGIC;
    uint32_t version       = VERSION;
    uint32_t meshletCount  = 0;
    uint32_t vertexCount   = 0; // total unique vertices across all meshlets
    uint32_t triangleCount = 0;
    uint32_t boundsCount   = 0;
};

struct CacheData {
    std::vector<Meshlet>         meshlets;
    std::vector<uint32_t>        meshletVertices;   // vertex indirection table
    std::vector<MeshletTriangle> meshletTriangles;
    std::vector<MeshletBounds>   meshletBounds;
    std::vector<float>           positions;         // float3 per vertex
    std::vector<uint32_t>        packedNormals;     // RGB10A2_SNORM per vertex
    std::vector<uint32_t>        packedUVs;         // RG16_FLOAT per vertex
};

// Write meshlet data to a .bin file. Returns true on success.
bool WriteBin(const std::string& path, const CacheData& data);

// Read meshlet data from a .bin file. Returns true on success.
bool ReadBin(const std::string& path, CacheData& data);

// Check if cache file exists and is newer than source file.
bool IsCacheValid(const std::string& cachePath, const std::string& sourcePath);

} // namespace MeshletCache