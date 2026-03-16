#pragma once

#include <unordered_map>
#include "GraphicsTypes.h"
#include "GraphicsHelper.h"

// Forward declarations to avoid circular dependencies
struct GLTFVertex;
struct GLTFPrimitive;

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

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
    void DispatchRasterIndirectGI(class Model* model, const FrameConstants& frame);
    void CopyTextureToBackBuffer(const GPUTexture& texture);

    // Resource creation
    void CreateRootSignature();
    void CreatePipelineState();
    void CreateRayTracingPipeline();
    void CreateShaderBindingTable();
    void CreateRasterIndirectGIResources();
    void CreateRasterIndirectGIPipelines();
    bool ReloadShadersIfNeeded(bool forceReload = false);

    // GBuffer management
    void CreateGBuffer();

    // Shader compilation

    std::vector<char> LoadShader(const std::string& filename);

    // Constant buffer management
    void UpdateFrameCB(const FrameConstants& frameConstants);

    // Getters
    ID3D12Device* GetDevice() const { return m_Device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_CommandList.Get(); }
    ID3D12CommandQueue* GetCommandQueue() const { return m_CommandQueue.Get(); }
    ID3D12CommandAllocator* GetCommandAllocator() const { return m_CommandAllocator.Get(); }
    ID3D12RootSignature* GetRootSignature() const { return m_RootSignature.Get(); }
    ID3D12CommandSignature* GetCommandSignature() const { return m_CommandSignature.Get(); }
    ID3D12PipelineState* GetTransparentPSO() const { return m_TransparentPSO.Get(); }
    ID3D12PipelineState* GetDepthPrePassPSO() const { return m_DepthPrePassPSO.Get(); }
    ID3D12PipelineState* GetGBufferPSO() const { return m_GBufferPSO.Get(); }
    ID3D12PipelineState* GetGBufferWritePSO() const { return m_GBufferWritePSO.Get(); }
    ID3D12PipelineState* GetLightingPSO() const { return m_LightingPSO.Get(); }
    ID3D12PipelineState* GetDebugPSO() const { return m_DebugPSO.Get(); }
    ID3D12PipelineState* GetShadowPSO() const { return m_ShadowPSO.Get(); }
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
    GBuffer& GetGBuffer() { return m_GBuffer; }
    GPUTexture& GetShadowMap() { return m_ShadowMap; }
    GPUTexture& GetPathTracerOutput() { return m_PathTracerOutput; }
    GPUTexture& GetRasterIndirectLightingTex() { return m_RasterIndirectLightingTex; }
    UINT GetIrCacheSRVIndex() const { return (UINT)m_IrCacheIrradianceBuf.uavIndex; }
    const IrCacheBindlessIndices& GetIrCacheBindlessIndices() const { return m_IrCacheIndices; }
    void DrawProbeSpheresDebug();

    // Lights
    void CreateLightsBuffer();
    void UpdateLightsBuffer(const std::vector<LightConstants>& lights);
    D3D12_GPU_VIRTUAL_ADDRESS GetLightsBufferGPUAddress() const;
    UINT GetLightsDescriptorIndex() const { return (UINT)m_LightsBuffer.srvIndex; }
    
    // Light LUT buffer for O(1) importance sampling
    void CreateLightLUTBuffer();
    void UpdateLightLUTBuffer(const std::vector<LightConstants>& lights);
    UINT GetLightLUTDescriptorIndex() const { return (UINT)m_LightLUTBuffer.srvIndex; }

private:
    void GetHardwareAdapter(IDXGIFactory1* pFactory, IDXGIAdapter1** ppAdapter);
    void WaitForPreviousFrame();
    void RebuildShaderPipelines();

    // DirectX 12 objects
    Microsoft::WRL::ComPtr<ID3D12Device> m_Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_CommandQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> m_SwapChain;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_RTVHeap;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_RenderTargets[2];
    D3D12_RESOURCE_STATES m_BackBufferStates[2] = { D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_PRESENT };
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_CommandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_CommandList;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_Fence;

    // Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_TransparentPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DepthPrePassPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GBufferPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GBufferWritePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LightingPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DebugPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ShadowPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_ProbeSphereDebugPSO;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> m_RootSignature;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_CommandSignature;

    // Ray Tracing
    bool m_RayTracingSupported = false;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathTracerPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirResolvePSO;
    
    // Light Resources
    GPUBuffer m_LightsBuffer;
    GPUBuffer m_LightLUTBuffer; // LUT for O(1) importance sampling
    static constexpr UINT LIGHT_LUT_RESOLUTION = 256;
    UINT m_MaxLights = 256;
    
    // RTXDI SDK Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirResolvePSO;

    std::unordered_map<const struct GLTFPrimitive*, GPUBuffer> m_BlasPool;
    GPUBuffer m_TLAS;
    GPUTexture m_PathTracerOutput;
    GPUTexture m_AccumulationBuffer;
    GPUBuffer m_ReservoirBuffer[2]; // ReSTIR Reservoirs (Current and Previous)
    GPUBuffer m_ReservoirIntermediate;
    
    // RTXDI Reservoir Buffer (RTXDI_PackedGIReservoir)
    GPUBuffer m_RtxdiReservoirBuffer[2];
    GPUBuffer m_RtxdiNeighborOffsetsBuffer;
    
    int m_CurrentReservoirIndex = 0;

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

    // ------- SHaRC (Spatial Hash Radiance Cache) buffers ~160 MB -------
    static constexpr UINT SHARC_HASH_ENTRIES_NUM = 4 * 1024 * 1024;
    GPUBuffer m_SharcHashEntriesBuf;    // uint64_t × SHARC_HASH_ENTRIES_NUM = 32 MB
    GPUBuffer m_SharcAccumulationBuf;   // SharcAccumulationData (uint4) × SHARC_HASH_ENTRIES_NUM = 64 MB
    GPUBuffer m_SharcResolvedBuf;       // SharcPackedData (float16_t4+2×uint) × SHARC_HASH_ENTRIES_NUM = 64 MB
    SharcBindlessIndices m_SharcIndices = {};

    GPUBuffer m_RasterReservoirs[2];
    GPUBuffer m_RasterReservoirIntermediate;
    GPUTexture m_RasterIndirectLightingTex;

    // ------- spatial ircache PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePoolInitPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareTracePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignature;

    // ------- Restir GI PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirGIRasterTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirGIRasterSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirGIRasterResolvePSO;

    // ------- SHaRC PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcDebugPSO;

    // GBuffer resources
    GBuffer m_GBuffer;
    GPUTexture m_ShadowMap;

    // Constant Buffers
    GPUBuffer m_FrameCB;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
};