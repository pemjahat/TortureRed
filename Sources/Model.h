#pragma once

#include <d3d12.h>
#include <d3dx12.h>
#include <wrl.h>
#include <vector>
#include <string>
#include <DirectXMath.h>
#include <DirectXCollision.h>
#include <DirectXTex.h>

// Forward declarations
struct cgltf_data;
class Renderer;

// Material constants matching the shader
struct MaterialConstants
{
    DirectX::XMFLOAT4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    UINT hasBaseColorTexture;
    UINT padding[1]; // Pad to 16 bytes
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

struct GLTFTexture
{
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    UINT srvIndex = UINT(-1); // Descriptor index
    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle;
    DirectX::ScratchImage* image = nullptr;
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

struct GLTFPrimitive
{
    std::vector<GLTFVertex> vertices;
    std::vector<uint32_t> indices;
    GLTFMaterial material;
    Microsoft::WRL::ComPtr<ID3D12Resource> vertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW vertexBufferView;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexBuffer;
    D3D12_INDEX_BUFFER_VIEW indexBufferView;
    Microsoft::WRL::ComPtr<ID3D12Resource> vertexStaging;
    Microsoft::WRL::ComPtr<ID3D12Resource> indexStaging;
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
    std::vector<GLTFTexture> textures;
    std::vector<GLTFNode*> rootNodes;
    cgltf_data* data = nullptr; // Raw cgltf data
};

class Model
{
public:
    Model();
    ~Model();

    bool LoadGLTFModel(ID3D12Device* device, const std::string& filepath);
    void UpdateAnimation(float deltaTime);
    void Render(ID3D12GraphicsCommandList* commandList, Renderer* renderer, const DirectX::BoundingFrustum& frustum);
    void UploadTextures(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* cmdQueue, ID3D12CommandAllocator* cmdAllocator, ID3D12DescriptorHeap* srvHeap);

    // Getters for debug counters
    size_t GetTotalNodes() const { return m_TotalNodes; }
    size_t GetTotalRootNodes() const { return m_TotalRootNodes; }
    size_t GetNodesSurviveFrustum() const { return m_NodesSurviveFrustum; }

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    void CreateGLTFResources(ID3D12Device* device);
    void RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer, const DirectX::BoundingFrustum& frustum);
    void ComputeWorldAABBs(GLTFNode* node, DirectX::XMMATRIX parentTransform);
    void LoadTextures(ID3D12Device* device);
    void BuildNodeHierarchy();
    void LoadAnimations();

    GLTFModel m_GltfModel;
    std::wstring fileDirectory;
    CD3DX12_GPU_DESCRIPTOR_HANDLE srvHeapStart;
    UINT srvDescriptorSize;
    // Animation
    GLTFAnimation* m_CurrentAnimation = nullptr;
    float m_AnimationTime = 0.0f;

    // Debug counters
    size_t m_TotalNodes = 0;
    size_t m_TotalRootNodes = 0;
    size_t m_NodesSurviveFrustum = 0;
};