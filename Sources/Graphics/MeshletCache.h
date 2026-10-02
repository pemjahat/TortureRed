#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "Shared/SharedTypes.h"

// Meshlet binary cache (.meshlet.bin) for persisting the meshlet build
// (Phase-1 meshopt optimization + Phase-2 meshlet generation), so model load
// skips straight to resource upload on a fresh cache.
// Cache path convention: <exeDir>/Clusters/<gltf_stem>/<primitive_index>.meshlet.bin
// — one folder per map, one file per primitive (the ShaderCache pattern:
// caches live beside the exe, never beside sources).

namespace MeshletCache {

static constexpr uint32_t MAGIC   = 0x4D534854; // 'MSHT'
static constexpr uint32_t VERSION = 0; // development: format NOT finalized — no compatibility
                                       // maintained across changes. Any layout change while v0
                                       // means old caches are rejected/regenerated; freeze at 1
                                       // with real versioning once the layout stabilizes.

struct Header {
    uint32_t magic         = MAGIC;
    uint32_t version       = VERSION;
    uint32_t meshletCount  = 0;
    uint32_t vertexCount   = 0; // vertex INDIRECTION table length (sum of per-meshlet counts; >= unique)
    uint32_t uniqueVertexCount = 0; // the vertex STREAM length (positions/3 = normals = uvs)
    uint32_t triangleCount = 0;
    uint32_t boundsCount   = 0;
    uint32_t indexCount    = 0; // post-optimization (remapped) index buffer length
    float    primSphere[4] = { 0, 0, 0, 0 }; // local-space bounding sphere (center xyz, radius)
};

struct CacheData {
    std::vector<Meshlet>         meshlets;
    std::vector<uint32_t>        meshletVertices;   // vertex indirection table
    std::vector<MeshletTriangle> meshletTriangles;
    std::vector<MeshletBounds>   meshletBounds;
    std::vector<float>           positions;         // float3 per vertex
    std::vector<uint32_t>        packedNormals;     // RGB10A2_SNORM per vertex
    std::vector<uint32_t>        packedUVs;         // RG16_FLOAT per vertex
    std::vector<uint32_t>        indices;           // post-optimization index buffer (RT path consumes it)
    float                        primSphere[4] = { 0, 0, 0, 0 }; // primitive bounding sphere
};

// Write meshlet data to a .bin file. Returns true on success.
bool WriteBin(const std::string& path, const CacheData& data);

// Read meshlet data from a .bin file. Returns true on success.
bool ReadBin(const std::string& path, CacheData& data);

// Check if a cache file exists — the per-map path is the key; format changes
// are caught by the header VERSION. No mtime freshness during development.
bool IsCacheValid(const std::string& cachePath);

} // namespace MeshletCache