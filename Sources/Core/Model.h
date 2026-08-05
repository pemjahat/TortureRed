#pragma once

#include "Graphics/GraphicsTypes.h"
#include "Graphics/MeshletCache.h"

// Forward declarations
struct cgltf_data;
class Renderer;

// Material constants, DrawNodeData moved to Shared/SharedTypes.h

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

    // Meshlet data (CPU-side, uploaded to GPU in CreateMeshletResources)
    std::vector<Meshlet>         meshlets;
    std::vector<uint32_t>        meshletVertices;   // vertex indirection table
    std::vector<MeshletTriangle> meshletTriangles;
    std::vector<MeshletBounds>   meshletBounds;      // per-meshlet bounding sphere

    // Primitive-level bounding sphere — enclosing sphere of the union of
    // meshletBounds, computed in Model::BuildMeshlets(). Feeds InstanceBounds
    DirectX::XMFLOAT3 boundsSphereCenter = { 0.0f, 0.0f, 0.0f };
    float              boundsSphereRadius = 0.0f;
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
    void UploadBuffers(Renderer* renderer);

    // Getters for debug counters
    size_t GetTotalNodes() const { return m_TotalNodes; }
    size_t GetTotalRootNodes() const { return m_TotalRootNodes; }
    size_t GetNodesSurviveFrustum() const { return m_NodesSurviveFrustum; }

    // Meshlet pipeline
    bool IsMeshletReady() const { return m_MeshletReady; }
    D3D12_GPU_VIRTUAL_ADDRESS GetMeshDataBufferAddress() const { return m_MeshDataBuffer.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetInstanceDataBufferAddress() const { return m_InstanceDataBuffer.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalPositionsBufferAddress() const { return m_GlobalPositions.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalNormalsBufferAddress() const { return m_GlobalNormals.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalUVsBufferAddress() const { return m_GlobalUVs.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalMeshletsBufferAddress() const { return m_GlobalMeshlets.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalMeshletVerticesBufferAddress() const { return m_GlobalMeshletVertices.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalMeshletTrianglesBufferAddress() const { return m_GlobalMeshletTriangles.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetGlobalMeshletBoundsBufferAddress() const { return m_GlobalMeshletBounds.gpuAddress; }
    D3D12_GPU_VIRTUAL_ADDRESS GetInstanceBoundsBufferAddress() const { return m_InstanceBoundsBuffer.gpuAddress; }
    size_t GetTotalMeshletCount() const { return m_TotalMeshletCount; }
    size_t GetInstanceCount()     const { return m_InstanceDataArray.size(); }
    // Returns the SRV heap index of GlobalPositions — the first of 9 contiguous meshlet stream SRVs
    // (GlobalPositions[0], GlobalNormals[1], GlobalUVs[2], GlobalMeshlets[3], GlobalMeshletVertices[4],
    //  GlobalMeshletTriangles[5], GlobalMeshletBounds[6], MeshData[7], InstanceData[8]).
    // Root param 14 descriptor table (t0-t15 space3) must be set to this base.
    int GetMeshletStreamSRVBase() const { return m_GlobalPositions.srvIndex; }

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
    void BuildMeshlets(GLTFPrimitive& prim);
    void CreateMeshletResources(Renderer* renderer);
    void RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode);
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

    // Draw Node Data (Combined Transform and Draw Metadata)
    std::vector<DrawNodeData> m_DrawNodeData;
    GPUBuffer m_DrawNodeBuffer;

    // Indirect Draw Commands
    std::vector<IndirectDrawCommand> m_OpaqueCommands;
    GPUBuffer m_OpaqueCommandBuffer;

    std::vector<IndirectDrawCommand> m_TransparentCommands;
    GPUBuffer m_TransparentCommandBuffer;

    // Global Vertex/Index Buffers
    std::vector<GLTFVertex> m_GlobalVertices;
    std::vector<uint32_t> m_GlobalIndices;
    GPUBuffer m_GlobalVertexBuffer;
    GPUBuffer m_GlobalIndexBuffer;

    // Animation
    GLTFAnimation* m_CurrentAnimation = nullptr;
    float m_AnimationTime = 0.0f;

    // Debug counters
    size_t m_TotalNodes = 0;
    size_t m_TotalRootNodes = 0;
    size_t m_NodesSurviveFrustum = 0;

    // ----- Meshlet Pipeline -----
    bool m_MeshletReady = false;

    // Global stream buffers (one per stream type, all meshes concatenated)
    GPUBuffer m_GlobalPositions;          // StructuredBuffer<float3>
    GPUBuffer m_GlobalNormals;            // StructuredBuffer<uint>  (RGB10A2_SNORM)
    GPUBuffer m_GlobalUVs;                // StructuredBuffer<uint>  (RG16_FLOAT)
    GPUBuffer m_GlobalMeshlets;           // StructuredBuffer<Meshlet>
    GPUBuffer m_GlobalMeshletVertices;    // StructuredBuffer<uint>  (vertex indirection)
    GPUBuffer m_GlobalMeshletTriangles;   // StructuredBuffer<MeshletTriangle>
    GPUBuffer m_GlobalMeshletBounds;      // StructuredBuffer<MeshletBounds>

    // CPU-side consolidated meshlet stream data (mirrors m_GlobalVertices/m_GlobalIndices pattern).
    // Built once in CreateMeshletResources(), uploaded directly via .data()/.size() in UploadBuffers().
    std::vector<float3>            m_AllPositions;
    std::vector<uint32_t>          m_AllPackedNormals;
    std::vector<uint32_t>          m_AllPackedUVs;
    std::vector<Meshlet>           m_AllMeshlets;
    std::vector<uint32_t>          m_AllMeshletVertices;
    std::vector<MeshletTriangle>   m_AllMeshletTriangles;
    std::vector<MeshletBounds>     m_AllMeshletBounds;

    // Per-mesh metadata (offsets into the global stream buffers)
    // Built in Pass 1: iterate GLTFModel::meshes[].primitives[].
    // One entry per unique primitive. Order is independent of InstanceData.
    std::vector<MeshData> m_MeshDataArray;
    GPUBuffer m_MeshDataBuffer;

    // Per-instance data — built in Pass 2: iterate GLTFModel::nodes[].
    // Each entry's MeshDataIndex points into m_MeshDataArray.
    // Multiple instances can reference the same MeshData (shared mesh).
    std::vector<InstanceData> m_InstanceDataArray;
    GPUBuffer m_InstanceDataBuffer;

    // Per-SubMesh LOCAL-space AABB (union of the primitive's meshlet local bounds).
    // One entry per MeshData — built in Pass 1, indexed by MeshDataIndex.
    // Cull shaders transform it by the per-frame InstanceData.LocalToWorld,
    // so culling follows animated instances (no stale load-time transform).
    std::vector<InstanceBounds> m_InstanceBoundsArray;
    GPUBuffer m_InstanceBoundsBuffer;

    size_t m_TotalMeshletCount = 0;
};