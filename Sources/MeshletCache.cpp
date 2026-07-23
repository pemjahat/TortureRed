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
    header.triangleCount = static_cast<uint32_t>(data.meshletTriangles.size());
    header.boundsCount   = static_cast<uint32_t>(data.meshletBounds.size());

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

    file.close();
    std::cout << "[MeshletCache] Wrote " << path << " (" 
              << header.meshletCount << " meshlets, "
              << header.vertexCount << " vertices)" << std::endl;
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

    // Read positions: vertexCount * 3 floats
    {
        uint32_t positionCount = header.vertexCount * 3;
        data.positions.resize(positionCount);
        file.read(reinterpret_cast<char*>(data.positions.data()), positionCount * sizeof(float));
    }

    // Read packed normals: vertexCount uints
    ReadVec(data.packedNormals, header.vertexCount);

    // Read packed UVs: vertexCount uints
    ReadVec(data.packedUVs, header.vertexCount);

    if (file.fail()) {
        std::cerr << "[MeshletCache] Failed to read cache data: " << path << std::endl;
        return false;
    }

    return true;
}

bool IsCacheValid(const std::string& cachePath, const std::string& sourcePath)
{
    namespace fs = std::filesystem;
    std::error_code ecSrc, ecCache;

    if (!fs::exists(cachePath, ecCache))
        return false;
    if (!fs::exists(sourcePath, ecSrc))
        return false;

    auto srcTime  = fs::last_write_time(sourcePath, ecSrc);
    auto cacheTime = fs::last_write_time(cachePath, ecCache);

    if (ecSrc || ecCache)
        return false;

    return cacheTime >= srcTime;
}

} // namespace MeshletCache