#pragma once

#include <d3d12.h>
#include <d3dx12.h>
#include <wrl.h>
#include <vector>
#include <string>
#include <DirectXMath.h>
#include <DirectXCollision.h>
#include <DirectXTex.h>
#include "GraphicsTypes.h"

// Forward declarations
struct cgltf_data;
class Renderer;

// Material constants matching the shader
struct MaterialConstants
{
    DirectX::XMFLOAT4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
};

struct InstanceData
{
    DirectX::XMFLOAT4X4 transform;
};

struct GLTFVertex
{
    float position[3];
    float normal[3];
    float texCoord[2];
};

struct GLTFImage
{
    GPUTexture texture;
    DirectX::ScratchImage* image = nullptr;
};

struct GLTFTexture
{
    GLTFImage* source = nullptr;
};

struct GLTFMaterial
{
    float baseColorFactor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    GLTFTexture* baseColorTexture = nullptr;
    GLTFTexture* normalTexture = nullptr;
    // Add more as needed
};

enum class AlphaMode
{
    Opaque,
    Mask,
    Blend
};

struct GLTFPrimitive
{
    std::vector<GLTFVertex> vertices;
    std::vector<uint32_t> indices;
    GLTFMaterial material;
    UINT materialIndex = 0;
    AlphaMode alphaMode = AlphaMode::Opaque;
    GPUBuffer vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    GPUBuffer indexBuffer;
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
    GPUBuffer vertexStaging;
    GPUBuffer indexStaging;
    DirectX::BoundingBox aabb;
};

struct GLTFMesh
{
    std::string name;
    std::vector<GLTFPrimitive> primitives;
};

struct GLTFNode
{
    std::string name;
    GLTFMesh* mesh = nullptr;
    std::vector<GLTFNode*> children;
    DirectX::XMFLOAT4X4 transform;
    GLTFNode* parent = nullptr;
    DirectX::BoundingBox worldAabb;
    // TRS for animation
    DirectX::XMFLOAT3 translation = {0.0f, 0.0f, 0.0f};
    DirectX::XMFLOAT4 rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    DirectX::XMFLOAT3 scale = {1.0f, 1.0f, 1.0f};
};

struct GLTFAnimationChannel
{
    enum Type { Translation, Rotation, Scale };
    Type type;
    GLTFNode* targetNode = nullptr;
    std::vector<float> times;
    std::vector<DirectX::XMFLOAT3> translations; // for translation
    std::vector<DirectX::XMFLOAT4> rotations; // for rotation
    std::vector<DirectX::XMFLOAT3> scales; // for scale
};

struct GLTFAnimation
{
    std::string name;
    std::vector<GLTFAnimationChannel> channels;
};

struct GLTFModel
{
    std::vector<GLTFMesh> meshes;
    std::vector<GLTFNode> nodes;
    std::vector<GLTFAnimation> animations;
    std::vector<GLTFImage> images;
    std::vector<GLTFTexture> textures;
    std::vector<GLTFNode*> rootNodes;
    cgltf_data* data = nullptr; // Raw cgltf data
};

class Model
{
public:
    Model();
    ~Model();

    bool LoadGLTFModel(Renderer* renderer, const std::string& filepath);
    void UpdateAnimation(float deltaTime);
    void Render(ID3D12GraphicsCommandList* commandList, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode = AlphaMode::Opaque);
    void UploadTextures(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* cmdQueue, ID3D12CommandAllocator* cmdAllocator, Renderer* renderer);

    // Getters for debug counters
    size_t GetTotalNodes() const { return m_TotalNodes; }
    size_t GetTotalRootNodes() const { return m_TotalRootNodes; }
    size_t GetNodesSurviveFrustum() const { return m_NodesSurviveFrustum; }

    // Get all primitives for AS building
    void GetAllPrimitives(std::vector<const struct GLTFPrimitive*>& primitives) const;

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    void CreateGLTFResources(Renderer* renderer);
    void RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode);    void GetPrimitivesRecursive(const struct GLTFNode* node, std::vector<const struct GLTFPrimitive*>& primitives) const;    void ComputeWorldAABBs(GLTFNode* node, DirectX::XMMATRIX parentTransform);
    void LoadTextures(Renderer* renderer);
    void LoadMaterials();
    void BuildNodeHierarchy();
    void LoadAnimations();

    GLTFModel m_GltfModel;
    std::wstring fileDirectory;
    UINT srvDescriptorSize;

    // Materials
    std::vector<MaterialConstants> m_Materials;
    GPUBuffer m_MaterialBuffer;
    GPUBuffer m_MaterialStagingBuffer;

    // Animation
    GLTFAnimation* m_CurrentAnimation = nullptr;
    float m_AnimationTime = 0.0f;

    // Debug counters
    size_t m_TotalNodes = 0;
    size_t m_TotalRootNodes = 0;
    size_t m_NodesSurviveFrustum = 0;
};