#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <vector>
#include <string>

// Forward declarations for cgltf
struct cgltf_data;

// GLTF Model structures
struct GLTFVertex
{
    float position[3];
    float normal[3];
    float texCoord[2];
};

struct GLTFMaterial
{
    float baseColorFactor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
};

struct GLTFMesh
{
    std::vector<GLTFVertex> vertices;
    std::vector<uint32_t> indices;
    GLTFMaterial material;
    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
};

struct GLTFModel
{
    std::vector<GLTFMesh> meshes;
    cgltf_data* data = nullptr; // Raw cgltf data
};

class Model
{
public:
    Model();
    ~Model();

    bool LoadGLTFModel(ID3D12Device* device, const std::string& filepath);
    void Render(ID3D12GraphicsCommandList* commandList);

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    void CreateGLTFResources(ID3D12Device* device);

    GLTFModel m_GltfModel;
};