#pragma once

#include <memory>
#include <unordered_map>
#include <filesystem>
#include "Graphics/GraphicsTypes.h"
#include "Graphics/GraphicsHelper.h"
#include "AccelerationStructure.h"
#include "Meshlet.h"
#include "PathTracing.h"
#include "Denoise.h"
#include "RestirDI.h"
#include "RestirGI.h"
#include "TAA.h"
#include "Shadow.h"
#include "GBuffer.h"
#include "DeferredLighting.h"
#include "Transparency.h"

// Forward declarations to avoid circular dependencies
struct GLTFVertex;
struct GLTFPrimitive;

const int WINDOW_WIDTH = 1920;
const int WINDOW_HEIGHT = 1080;

class Renderer
{
public:
    Renderer();
    ~Renderer();

    bool Initialize(HWND hwnd);
    void Shutdown();
    void Resize(uint32_t width, uint32_t height);

    // Rendering functions
    void BeginFrame();
    void EndFrame();
    void Present();

    void BuildAccelerationStructures(class Model* model);
    void DispatchRays(class Model* model, const FrameConstants& frame, const LightConstants& light);
    void DispatchRestirGI(class Model* model, const FrameConstants& frame);
    void CopyTextureToBackBuffer(const GPUTexture& texture);
    void DrawPathVizLines(const FrameConstants& frame);

    // Resource creation
    void CreateRootSignature();
    void CreatePipelineState();
    void CreateRayTracingPipeline();
    void CreateRasterIndirectGIResources();
    void CreateRasterIndirectGIPipelines();
    void CreateRestirDIResources();
    void CreateRestirDIPipelines();
    void DispatchRestirDI(class Model* model, const FrameConstants& frame);

    // TAA / Temporal Super-Resolution
    void CreateTaaResources(uint32_t outputW, uint32_t outputH, uint32_t internalW, uint32_t internalH);
    void CreateTaaPipelines();
    void DispatchNaiveTsr(const FrameConstants& frame, const GPUTexture& inputColor);
    void GenerateMotionVectors(const FrameConstants& frame);
    GPUTexture& GetTaaOutputTex() { return m_TAA.GetOutputTex(); }
    GPUTexture& GetNrdMotionVectorsTex() { return m_Denoise.GetMotionVectorsTex(); }
    bool IsTaaEnabled() const { return m_TAA.IsEnabled(); }

    // Internal resolution resource management
    void CreateInternalResolutionResources(uint32_t w, uint32_t h);
    uint32_t GetInternalWidth() const { return m_InternalWidth; }
    uint32_t GetInternalHeight() const { return m_InternalHeight; }

    // GBuffer management
    void CreateGBuffer(uint32_t w, uint32_t h);
    void ExecuteGBufferPass(class Model* model, const DirectX::BoundingFrustum& frustum, bool enableDepthPrePass);
    void ExecuteLightingPass(class Model* model, const FrameConstants& frame, bool rasterTaaActive,
                              bool debugActive, bool debugShadowMap, uint32_t outputWidth, uint32_t outputHeight);
    void ExecuteTransparencyPass(class Model* model, const DirectX::BoundingFrustum& frustum,
                                  bool rasterTaaActive, uint32_t outputWidth, uint32_t outputHeight);

    // Shader compilation

    std::vector<char> LoadShader(const std::string& filename);

    // Constant buffer management
    void UpdateFrameCB(const FrameConstants& frameConstants);

    // Getters
    ID3D12Device* GetDevice() const { return m_Device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_CommandList.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_CommandQueue.Get(); }
    ID3D12CommandQueue* GetCopyQueue() const { return m_CopyQueue.Get(); }
    ID3D12CommandAllocator* GetCommandAllocator() const { return m_CommandAllocator.Get(); }
    ID3D12RootSignature* GetRootSignature() const { return m_RootSignature.Get(); }
    ID3D12CommandSignature* GetCommandSignature() const { return m_CommandSignature.Get(); }
    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }
    ID3D12PipelineState* GetProbeSphereDebugPSO() const { return m_ProbeSphereDebugPSO.Get(); }
    
    // Ray Tracing Getters
    bool IsRayTracingSupported() const { return m_RayTracingSupported; }
    void ExecuteCommandList();

    // Pass management
    D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentBackBufferRTV() const;
    ID3D12Resource* GetCurrentBackBuffer() const;

    D3D12_GPU_VIRTUAL_ADDRESS GetFrameGPUAddress() const { return m_FrameCB.gpuAddress; }
    void TransitionBackBuffer(D3D12_RESOURCE_STATES newState);

    // Background color
    float m_BackgroundColor[3] = { 0.098f, 0.098f, 0.439f }; // Default: Dark blue

    // GBuffer access
    GBuffer& GetGBuffer() { return m_GBufferPass.GetGBuffer(); }
    GPUTexture& GetShadowMap() { return m_Shadow.GetShadowMap(); }
    GPUTexture& GetPathTracerOutput() { return m_PathTracing.GetOutput(); }
    GPUTexture& GetPathTracerHdrOutput() { return m_PathTracing.GetHdrOutput(); }
    GPUTexture& GetDIDiffuseIntermediate()  { return m_RestirDI.GetDIDiffuseIntermediate(); }
    GPUTexture& GetDISpecularIntermediate() { return m_RestirDI.GetDISpecularIntermediate(); }
    GPUTexture& GetGIDiffuseIntermediate()  { return m_RestirGI.GetGIDiffuseIntermediate(); }
    GPUTexture& GetGISpecularIntermediate() { return m_RestirGI.GetGISpecularIntermediate(); }
    GPUTexture& GetNrdDenoisedDiffuseTex()  { return m_Denoise.GetDenoisedDiffuseTex(); }
    GPUTexture& GetNrdDenoisedSpecularTex() { return m_Denoise.GetDenoisedSpecularTex(); }
    GPUTexture& GetFinalDiffuseTex()        { return m_FinalDiffuseTex; }
    GPUTexture& GetFinalSpecularTex()       { return m_FinalSpecularTex; }
    GPUTexture& GetRasterHdrOutputTex() { return m_RasterHdrOutputTex; }
    GPUTexture& GetRestirDebugHeatmap()       { return m_PathTracing.GetRestirDebugHeatmap(); }
    GPUTexture& GetFullScreenDebugTex()       { return m_FullScreenDebugTex; }
    ID3D12PipelineState* GetFullScreenDebugPSO() const { return m_FullScreenDebugPSO.Get(); }
    ID3D12PipelineState* GetFullScreenDebugHdrPSO() const { return m_FullScreenDebugHdrPSO.Get(); }
    UINT GetIrCacheSRVIndex() const { return (UINT)m_IrCacheIrradianceBuf.uavIndex; }
    const IrCacheBindlessIndices& GetIrCacheBindlessIndices() const { return m_IrCacheIndices; }
    void DrawProbeSpheresDebug();

    // Lights
    void CreateLightsBuffer() { m_DeferredLighting.CreateLightsBuffer(); }
    void UpdateLightsBuffer(const std::vector<LightConstants>& lights) { m_DeferredLighting.UpdateLightsBuffer(lights); }
    D3D12_GPU_VIRTUAL_ADDRESS GetLightsBufferGPUAddress() const { return m_DeferredLighting.GetLightsBufferGPUAddress(); }
    UINT GetLightsDescriptorIndex() const { return m_DeferredLighting.GetLightsDescriptorIndex(); }

    // Light LUT buffer for O(1) importance sampling
    void CreateLightLUTBuffer() { m_DeferredLighting.CreateLightLUTBuffer(); }
    void UpdateLightLUTBuffer(const std::vector<LightConstants>& lights) { m_DeferredLighting.UpdateLightLUTBuffer(lights); }
    UINT GetLightLUTDescriptorIndex() const { return m_DeferredLighting.GetLightLUTDescriptorIndex(); }

    // Shader hot-reload: call once per frame; GPU-syncs and rebuilds only changed PSOs.
    void CheckAndReloadShaders();

    // Meshlet pipeline
    void CreateMeshletResources();
    void CreateMeshletPipelines();
    // Hierarchical two-stage meshlet culling: CullInstancesCS → CullMeshletsCS.
    // occlusionEnabled=1, phase 0/1: two-phase occlusion culling (vs prev-HZB / fresh HZB).
    // occlusionEnabled=0 (phase must be 0): frustum-only single-phase hierarchical cull.
    void DispatchMeshletTwoPassCull(class Model* model, const FrameConstants& frame,
                                     bool occlusionEnabled, int phase);
    void DispatchMeshletBinning();                    // 4-pass GPU sort: Classify → Allocate → Write
    void DispatchMeshletRasterize(class Model* model); // Mesh Shader rasterize per bin
    void DispatchMeshletRasterizeDebug(class Model* model); // CPU-driven per-meshlet debug dispatch (no culling/binning)
    bool IsMeshShaderSupported() const { return m_MeshShaderSupported; }

    // HZB (Hierarchical Z-Buffer).
    // Builds the HZB mip chain from the current GBuffer depth via AMD FidelityFX SPD. Not yet
    // consumed for occlusion culling — Cull()'s two-phase rewrite is kept for a follow-up step.
    void DispatchBuildHZB() { m_Meshlet.BuildHZB(m_CommandList.Get(), m_GBufferPass.GetGBuffer().depth); }
    GPUTexture& GetHZB() { return m_Meshlet.GetHZB(); }
    uint32_t GetHZBMips() const { return m_Meshlet.GetHZBMips(); }

    // Visibility buffer for meshlet debug overlay (plan001)
    GPUTexture& GetVisibilityBuffer() { return m_Meshlet.GetVisibilityBuffer(); }
    int GetVisibleMeshletsSRVIndex() const { return m_Meshlet.GetVisibleMeshletsSRVIndex(); }
    ID3D12PipelineState* GetMeshletDebugViewPSO() const { return m_Meshlet.GetDebugViewPSO(); }
    int GetMeshletDebugMode() const { return m_Meshlet.GetDebugMode(); }
    void SetMeshletDebugMode(int mode) { m_Meshlet.SetDebugMode(mode); }

private:
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter);
    void WaitForPreviousFrame();
    bool InitializeNrd();
    void ShutdownNrd();
    bool NRDDenoise(const FrameConstants& frame);

    // Shader hot-reload helpers
    void SetupShaderTimestamps();

    // DirectX 12 objects
    Microsoft::WRL::ComPtr<ID3D12Device> m_Device;
    Microsoft::WRL::ComPtr<ID3D12Device2> m_Device2; // For CreatePipelineState (pipeline state streams, mesh shader PSOs)
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CopyQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_RenderTargets[2];
    D3D12_RESOURCE_STATES m_BackBufferStates[2] = { D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT };
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;

    // Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ProbeSphereDebugPSO;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_CommandSignature;

    // Ray Tracing
    bool m_RayTracingSupported = false;
    PathTracing m_PathTracing;

    // Light Resources
    DeferredLighting m_DeferredLighting;

    AccelerationStructure m_AccelStructure;

    // Rasterizer Indirect GI
    // ------- spatial irradiance cache buffers -------
    GPUBuffer m_IrCacheMetaBuf;          // 4 x uint meta counters
    GPUBuffer m_IrCachePoolBuf;          // uint free-list  [MAX_ENTRIES]
    GPUBuffer m_IrCacheGridMetaBuf;      // uint per cell (entryIdx<<3 | flags)  [TOTAL_CELLS]
    GPUBuffer m_IrCacheEntryCellBuf;     // uint entry->cell [MAX_ENTRIES]
    GPUBuffer m_IrCacheIrradianceBuf;    // Reservoir payload [MAX_ENTRIES]
    GPUBuffer m_IrCacheLifeBuf;          // uint life       [MAX_ENTRIES]
    GPUBuffer m_IrCacheIndirectionBuf;   // uint compact list [MAX_ENTRIES]
    GPUBuffer m_IrCacheTraceArgsBuf;     // uint3 indirect dispatch args
    // --- position voting buffers ---
    GPUBuffer m_IrCachePosBuf;           // float4[MAX_ENTRIES] — applied probe positions (read by Update)
    GPUBuffer m_IrCacheRepropBuf;        // float4[MAX_ENTRIES] — voted proposal positions (written by Update)
    GPUBuffer m_IrCacheRepropCountBuf;   // uint[MAX_ENTRIES]   — vote counters (cleared by Age, inc'd by Update)
    bool      m_IrCacheInitialized = false;
    IrCacheBindlessIndices m_IrCacheIndices = {};

    // ------- ReSTIR GI (SHaRC + split diffuse/specular)
    RestirGI m_RestirGI;

    GPUTexture m_RasterHdrOutputTex;          // Internal-res HDR output for rasterizer TAA
    // ---- NRD/Denoise textures/state: moved to Rendering/Denoise.h/.cpp ----
    Denoise m_Denoise;
    GPUTexture m_FinalDiffuseTex;          // Universal interchange: SSO writes, NrdPackNoise+Lighting read
    GPUTexture m_FinalSpecularTex;         // Universal interchange: SSO writes, NrdPackNoise+Lighting read
    GPUTexture m_FullScreenDebugTex;       // Unified debug output: SHaRC/heatmap/field debug → FullScreenDebug.hlsl

    // ------- spatial ircache PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePoolInitPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareTracePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignature;

    
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_FullScreenDebugPSO;    // Debug PSO targeting R8G8B8A8_UNORM (LDR, with tonemapping)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_FullScreenDebugHdrPSO; // Debug PSO targeting R16G16B16A16_FLOAT (HDR, no tonemapping)

    // ------- ReSTIR DI
    RestirDI m_RestirDI;

    // ------- ReSTIR GI/SHaRC PSOs; only the shared bridge stays here -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdStoreShadingOutputPSO;    // Generic 2-input/2-output bridge → Final* (shared by RestirDI + RestirGI, stays in Renderer for now)

    // ------- Path Visualization -------

    // ------- Internal resolution tracking -------
    uint32_t m_InternalWidth = WINDOW_WIDTH;
    uint32_t m_InternalHeight = WINDOW_HEIGHT;

    // ------- TAA / Temporal Super-Resolution -------
    TAA m_TAA;

    // GBuffer resources
    GBufferPass m_GBufferPass;
    Shadow m_Shadow;
    Transparency m_Transparency;

    // Constant Buffers
    GPUBuffer m_FrameCB;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Shader hot-reload: tracks all .hlsl timestamps under Sources/Shaders/
    std::unordered_map<std::string, std::filesystem::file_time_type>  m_ShaderTimestamps;

    // ----- Meshlet Pipeline -----
    bool m_MeshShaderSupported = false;
    MeshletPass m_Meshlet;

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
};