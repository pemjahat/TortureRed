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

struct DrawNodeData
{
    DirectX::XMFLOAT4X4 world;
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t materialID;
    uint32_t padding;
};

struct IndirectDrawCommand
{
    D3D12_DRAW_INDEXED_ARGUMENTS drawArgs;
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
    UINT materialIndex = 0;
    AlphaMode alphaMode = AlphaMode::Opaque;
    uint32_t globalVertexOffset = 0;
    uint32_t globalIndexOffset = 0;
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
    uint32_t nodeDataOffset = 0;
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
    void GetDrawNodePrimitives(std::vector<const struct GLTFPrimitive*>& primitives) const;
    const std::vector<DrawNodeData>& GetDrawNodeData() const { return m_DrawNodeData; }
    
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalVertexBufferAddress() const { return m_GlobalVertexBuffer.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalIndexBufferAddress() const { return m_GlobalIndexBuffer.gpuAddress; }

    D3D12_GPU_VIRTUAL_ADDRESS GetMaterialBufferAddress() const { return m_MaterialBuffer.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetDrawNodeBufferAddress() const { return m_DrawNodeBuffer.gpuAddress; }

    // Update node buffer with current node transforms
    void UpdateNodeBuffer();

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    void CreateGLTFResources(Renderer* renderer);
    void RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode);
    void ComputeWorldAABBs(GLTFNode* node, DirectX::XMMATRIX parentTransform);
    void UpdateNodeBufferRecursive(GLTFNode* node, DirectX::XMMATRIX parentTransform);
    void LoadTextures(Renderer* renderer);
    void LoadMaterials();
    void BuildNodeHierarchy();
    void LoadAnimations();

    GLTFModel m_GltfModel;
    std::wstring fileDirectory;
    UINT srvDescriptorSize;

    // GPU Materials
    std::vector<MaterialConstants> m_MaterialConstants;
    GPUBuffer m_MaterialBuffer;
    GPUBuffer m_MaterialStagingBuffer;

    // Draw Node Data (Combined Transform and Draw Metadata)
    std::vector<DrawNodeData> m_DrawNodeData;
    GPUBuffer m_DrawNodeBuffer;

    // Indirect Draw Commands
    std::vector<IndirectDrawCommand> m_OpaqueCommands;
    GPUBuffer m_OpaqueCommandBuffer;
    GPUBuffer m_OpaqueCommandStagingBuffer;

    std::vector<IndirectDrawCommand> m_TransparentCommands;
    GPUBuffer m_TransparentCommandBuffer;
    GPUBuffer m_TransparentCommandStagingBuffer;

    // Global Vertex/Index Buffers
    std::vector<GLTFVertex> m_GlobalVertices;
    std::vector<uint32_t> m_GlobalIndices;
    GPUBuffer m_GlobalVertexBuffer;
    GPUBuffer m_GlobalVertexStaging;
    GPUBuffer m_GlobalIndexBuffer;
    GPUBuffer m_GlobalIndexStaging;

    // Animation
    GLTFAnimation* m_CurrentAnimation = nullptr;
    float m_AnimationTime = 0.0f;

    // Debug counters
    size_t m_TotalNodes = 0;
    size_t m_TotalRootNodes = 0;
    size_t m_NodesSurviveFrustum = 0;
};