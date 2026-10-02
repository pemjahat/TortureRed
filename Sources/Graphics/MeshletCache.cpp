#include "pch.h"
#include "MeshletCache.h"
#include <fstream>
#include <filesystem>
#include <cstring>

namespace MeshletCache {

bool WriteBin(const std::string& path, const CacheData& data)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[MeshletCache] Failed to open cache for writing: " << path << std::endl;
        return false;
    }

    Header header;
    header.meshletCount  = static_cast<uint32_t>(data.meshlets.size());
    header.vertexCount   = static_cast<uint32_t>(data.meshletVertices.size());
    header.uniqueVertexCount = static_cast<uint32_t>(data.positions.size() / 3); // the STREAM length, not the indirection length
    header.triangleCount = static_cast<uint32_t>(data.meshletTriangles.size());
    header.boundsCount   = static_cast<uint32_t>(data.meshletBounds.size());
    header.indexCount    = static_cast<uint32_t>(data.indices.size());
    memcpy(header.primSphere, data.primSphere, sizeof(header.primSphere));

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));

    // Write arrays
    auto WriteVec = [&](const auto& vec) {
        file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(vec[0]));
    };

    WriteVec(data.meshlets);
    WriteVec(data.meshletVertices);
    WriteVec(data.meshletTriangles);
    WriteVec(data.meshletBounds);
    WriteVec(data.positions);
    WriteVec(data.packedNormals);
    WriteVec(data.packedUVs);
    WriteVec(data.indices);

    file.close();
    std::cout << "[MeshletCache] Wrote " << path << " ("
              << header.meshletCount << " meshlets, "
              << header.uniqueVertexCount << " vertices)" << std::endl;
    return true;
}

bool ReadBin(const std::string& path, CacheData& data)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    Header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(Header));

    if (header.magic != MAGIC) {
        std::cerr << "[MeshletCache] Invalid cache magic: " << path << std::endl;
        return false;
    }
    if (header.version != VERSION) {
        std::cerr << "[MeshletCache] Cache version mismatch (got " << header.version 
                  << ", expected " << VERSION << "): " << path << std::endl;
        return false;
    }

    auto ReadVec = [&](auto& vec, uint32_t count) {
        vec.resize(count);
        file.read(reinterpret_cast<char*>(vec.data()), count * sizeof(vec[0]));
    };

    ReadVec(data.meshlets,         header.meshletCount);
    ReadVec(data.meshletVertices,  header.vertexCount);
    ReadVec(data.meshletTriangles, header.triangleCount);
    ReadVec(data.meshletBounds,    header.boundsCount);

    // Vertex streams are sized by the UNIQUE vertex count — the
    // indirection-table length (header.vertexCount) double-counts vertices
    // shared between meshlets and would read past the file.
    const uint32_t uniqueVerts = header.uniqueVertexCount;
    if (uniqueVerts == 0) {
        std::cerr << "[MeshletCache] Corrupt cache (zero vertices): " << path << std::endl;
        return false;
    }

    // Read positions: uniqueVerts * 3 floats
    {
        uint32_t positionCount = uniqueVerts * 3;
        data.positions.resize(positionCount);
        file.read(reinterpret_cast<char*>(data.positions.data()), positionCount * sizeof(float));
    }

    // Read packed normals: uniqueVerts uints
    ReadVec(data.packedNormals, uniqueVerts);

    // Read packed UVs: uniqueVerts uints
    ReadVec(data.packedUVs, uniqueVerts);

    // Indices (the primitive sphere rides IN THE HEADER — there is no
    // trailing blob; reading one over-runs the file by exactly 16 bytes)
    ReadVec(data.indices, header.indexCount);
    memcpy(data.primSphere, header.primSphere, sizeof(data.primSphere));

    if (file.fail()) {
        std::cerr << "[MeshletCache] Failed to read cache data: " << path << std::endl;
        return false;
    }

    return true;
}

bool IsCacheValid(const std::string& cachePath)
{
    std::error_code ec;
    return std::filesystem::exists(cachePath, ec);
}

} // namespace MeshletCache