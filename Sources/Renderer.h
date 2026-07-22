#pragma once

#include <memory>
#include <unordered_map>
#include <filesystem>
#include "GraphicsTypes.h"
#include "GraphicsHelper.h"

// Forward declarations to avoid circular dependencies
struct GLTFVertex;
struct GLTFPrimitive;
namespace nrd { struct Integration; }

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
    GPUTexture& GetTaaOutputTex() { return m_TaaOutputTex; }
    GPUTexture& GetNrdMotionVectorsTex() { return m_NrdMotionVectorsTex; }
    bool IsTaaEnabled() const { return m_TaaEnabled; }

    // Internal resolution resource management
    void CreateInternalResolutionResources(uint32_t w, uint32_t h);
    uint32_t GetInternalWidth() const { return m_InternalWidth; }
    uint32_t GetInternalHeight() const { return m_InternalHeight; }

    // GBuffer management
    void CreateGBuffer(uint32_t w, uint32_t h);

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
    GPUTexture& GetPathTracerOutput() { return m_PathTracerPresentOutput; }
    GPUTexture& GetPathTracerHdrOutput() { return m_PathTracerOutput; }
    GPUTexture& GetDIDiffuseIntermediate()  { return m_DIDiffuseIntermediate; }
    GPUTexture& GetDISpecularIntermediate() { return m_DISpecularIntermediate; }
    GPUTexture& GetGIDiffuseIntermediate()  { return m_GIDiffuseIntermediate; }
    GPUTexture& GetGISpecularIntermediate() { return m_GISpecularIntermediate; }
    GPUTexture& GetNrdDenoisedDiffuseTex()  { return m_NrdDenoisedDiffuseTex; }
    GPUTexture& GetNrdDenoisedSpecularTex() { return m_NrdDenoisedSpecularTex; }
    GPUTexture& GetFinalDiffuseTex()        { return m_FinalDiffuseTex; }
    GPUTexture& GetFinalSpecularTex()       { return m_FinalSpecularTex; }
    GPUTexture& GetRasterHdrOutputTex() { return m_RasterHdrOutputTex; }
    GPUTexture& GetRestirDebugHeatmap()       { return m_RestirDebugHeatmap; }
    GPUTexture& GetFullScreenDebugTex()       { return m_FullScreenDebugTex; }
    ID3D12PipelineState* GetLightingHdrPSO() const { return m_LightingHdrPSO.Get(); }
    ID3D12PipelineState* GetFullScreenDebugPSO() const { return m_FullScreenDebugPSO.Get(); }
    ID3D12PipelineState* GetFullScreenDebugHdrPSO() const { return m_FullScreenDebugHdrPSO.Get(); }
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

    // Shader hot-reload: call once per frame; GPU-syncs and rebuilds only changed PSOs.
    void CheckAndReloadShaders();

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
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathTracerPresentPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirReservoirDebugPSO;
    
    // Light Resources
    GPUBuffer m_LightsBuffer;
    GPUBuffer m_LightLUTBuffer; // LUT for O(1) importance sampling
    static constexpr UINT LIGHT_LUT_RESOLUTION = 256;
    UINT m_MaxLights = 256;
    
    // RTXDI SDK Pipeline States
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RtxdiRestirReservoirDebugPSO;

    std::unordered_map<const struct GLTFPrimitive*, GPUBuffer> m_BlasPool;
    GPUBuffer m_TLAS;
    GPUTexture m_PathTracerOutput;
    GPUTexture m_PathTracerPresentOutput;
    GPUTexture m_AccumulationBuffer;
    GPUTexture m_RestirDebugHeatmap;
    GPUBuffer m_ReservoirBuffer[2]; // ReSTIR Reservoirs (Current and Previous)
    GPUBuffer m_ReservoirIntermediate;
    
    // RTXDI Reservoir Buffer (RTXDI_PackedGIReservoir)
    GPUBuffer m_RtxdiReservoirBuffer[2];
    GPUBuffer m_RtxdiReservoirIntermediate;
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

    // ------- Split Diffuse / Specular ReSTIR buffers -------
    GPUBuffer m_DiffuseReservoirBuffer[2];       // Ping-pong diffuse reservoirs
    GPUBuffer m_SpecularReservoirBuffer[2];      // Ping-pong specular reservoirs
    GPUBuffer m_DiffuseReservoirIntermediate;    // Post-spatial diffuse
    GPUBuffer m_SpecularReservoirIntermediate;   // Post-spatial specular
    GPUBuffer m_DiffuseCandidateBuffer;          // DiffuseCandidate per pixel (RTDGI → RTR)
    GPUTexture m_GIDiffuseIntermediate;          // NEW: GI resolved diffuse (raw float4: radiance, hitT)
    GPUTexture m_GISpecularIntermediate;         // NEW: GI resolved specular (raw float4: radiance, hitT)

    GPUTexture m_RasterHdrOutputTex;          // Internal-res HDR output for rasterizer TAA
    GPUTexture m_NrdMotionVectorsTex;
    GPUTexture m_NrdNormalRoughnessTex;
    GPUTexture m_NrdViewZTex;
    GPUTexture m_NrdRelaxDiffuseTex;       // RELAX-packed input for NRD denoiser
    GPUTexture m_NrdRelaxSpecularTex;      // RELAX-packed input for NRD denoiser
    GPUTexture m_NrdDenoisedDiffuseTex;
    GPUTexture m_NrdDenoisedSpecularTex;
    GPUTexture m_NrdValidationTex;
    GPUTexture m_FinalDiffuseTex;          // Universal interchange: SSO writes, NrdPackNoise+Lighting read
    GPUTexture m_FinalSpecularTex;         // Universal interchange: SSO writes, NrdPackNoise+Lighting read
    GPUTexture m_FullScreenDebugTex;       // Unified debug output: SHaRC/heatmap/field debug → FullScreenDebug.hlsl
    std::unique_ptr<nrd::Integration> m_NrdIntegration;
    bool m_NrdInitialized = false;
    bool m_NrdWasActiveLastFrame = false; // Tracks whether NRD ran last frame; used to force RESTART on re-enable

    // ------- spatial ircache PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePoolInitPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheAgePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCachePrepareTracePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_IrCacheUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DispatchCommandSignature;

    // ------- Split Diffuse / Specular ReSTIR PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DiffuseTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SpecularTemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_DiffuseSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SpecularSpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdPrepareGuidesPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdCompositePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_LightingHdrPSO; // Lighting PSO targeting R16G16B16A16_FLOAT (HDR, no tonemapping)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_FullScreenDebugPSO;    // Debug PSO targeting R8G8B8A8_UNORM (LDR, with tonemapping)
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_FullScreenDebugHdrPSO; // Debug PSO targeting R16G16B16A16_FLOAT (HDR, no tonemapping)

    // ------- ReSTIR DI buffers and textures -------
    GPUBuffer  m_DIReservoirBuffer[2];       // Ping-pong DI reservoirs (DIRreservoir per pixel)
    GPUBuffer  m_DIReservoirIntermediate;    // Post-spatial DI reservoirs
    GPUTexture m_DIDiffuseIntermediate;     // Split DI diffuse (NRD-normalized float4)
    GPUTexture m_DISpecularIntermediate;    // Split DI specular (NRD-normalized float4)
    int        m_CurrentDIReservoirIndex = 0;

    // ------- ReSTIR DI PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDITemporalPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDISpatialPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_RestirDISplitShadePSO;  // Split diffuse/specular shade
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_GIResolveIntermediatesPSO;   // GI reservoir → float4 intermediates
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdStoreShadingOutputPSO;    // Generic 2-input/2-output bridge → Final*
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NrdPackNoisePSO;             // Raw float4 → RELAX front-end pack

    // ------- SHaRC PSOs -------
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcUpdatePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_SharcDebugPSO;

    // ------- Path Visualization -------
    GPUBuffer m_PathVizLineBuffer;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_PathVizLinePSO;

    // ------- Internal resolution tracking -------
    uint32_t m_InternalWidth = WINDOW_WIDTH;
    uint32_t m_InternalHeight = WINDOW_HEIGHT;

    // ------- TAA / Temporal Super-Resolution -------
    bool m_TaaEnabled = false;
    int  m_TaaHistoryIndex = 0; // Ping-pong index (0 or 1)
    GPUTexture m_TaaHistoryTex[2];           // Output-res: rgb + coverage
    GPUTexture m_TaaReprojectedHistoryTex;   // Output-res: reprojected history
    GPUTexture m_TaaClosestVelocityTex;      // Output-res: dilated closest velocity
    GPUTexture m_TaaOutputTex;               // Output-res: final TAA output
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NaiveTsrReprojectPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_NaiveTsrResolvePSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> m_MotionVectorsPSO;

    // GBuffer resources
    GBuffer m_GBuffer;
    GPUTexture m_ShadowMap;

    // Constant Buffers
    GPUBuffer m_FrameCB;

    // Synchronization
    UINT m_FrameIndex;
    HANDLE m_FenceEvent;
    UINT64 m_FenceValue;

    // Shader hot-reload: tracks all .hlsl timestamps under Sources/Shaders/
    std::unordered_map<std::string, std::filesystem::file_time_type>  m_ShaderTimestamps;

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
};