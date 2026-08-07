#include "pch.h"

#include "Meshlet.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

void MeshletPass::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    // ---- Dispatch mesh args (single indirect DispatchMesh argument, 3 uints) ----
    if (!CreateStructuredBuffer(m_DispatchMeshArgs, sizeof(uint32_t), 3,
                                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_DispatchMeshArgs"))
    {
        std::cerr << "[Meshlet] Failed to create DispatchMeshArgs buffer" << std::endl;
        return;
    }

    std::cout << "[Meshlet] Resources created (no binning)" << std::endl;

    // Visibility buffer for debug overlay (plan001)
    if (!CreateTexture(m_VisibilityBuffer, internalWidth, internalHeight,
                       DXGI_FORMAT_R32_UINT,
                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "GBuffer_VisibilityBuffer"))
    {
        std::cerr << "[Meshlet] Failed to create visibility buffer" << std::endl;
    }

    // Culling, HZB, and debug-recording resources are now in GPUCulling.
}

void MeshletPass::RecreateVisibilityBuffer(uint32_t internalWidth, uint32_t internalHeight)
{
    if (!CreateTexture(m_VisibilityBuffer, internalWidth, internalHeight,
                       DXGI_FORMAT_R32_UINT,
                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, 1, 1, "GBuffer_VisibilityBuffer"))
    {
        std::cerr << "[Meshlet] Failed to recreate visibility buffer" << std::endl;
    }
}

void MeshletPass::CreatePipelines(ID3D12Device* device, ID3D12Device2* device2,
                                   ID3D12RootSignature* mainRootSignature,
                                   bool meshShaderSupported)
{
    m_MeshShaderSupported = meshShaderSupported;

    // --- BuildDispatchMeshArgs CS (1 thread, builds indirect DispatchMesh arguments) ---
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/MeshletBinning.hlsl", "BuildDispatchMeshArgsCS", "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = mainRootSignature;
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_BuildDispatchMeshArgsPSO)),
                     "[Meshlet] CreateComputePipelineState (BuildDispatchMeshArgs) failed");
        }
    }

    // --- Mesh Shader Raster PSOs (MS+PS) — single combined PSO, no ALPHA_MASK permutation ---
    // Alpha discard runs unconditionally in the pixel shader; back-face cull for all.
    if (m_MeshShaderSupported)
    {
        auto buildMeshPSO = [&](
            const std::vector<char>& ms,
            const std::vector<char>& ps,
            bool directToGBuffer,
            Microsoft::WRL::ComPtr<ID3D12PipelineState>& outPSO,
            const char* label)
        {
            if (ms.empty() || ps.empty()) return;

            struct MeshShaderPSOStream
            {
                CD3DX12_PIPELINE_STATE_STREAM_ROOT_SIGNATURE        RootSignature;
                CD3DX12_PIPELINE_STATE_STREAM_MS                    MS;
                CD3DX12_PIPELINE_STATE_STREAM_PS                    PS;
                CD3DX12_PIPELINE_STATE_STREAM_RASTERIZER            RasterizerState;
                CD3DX12_PIPELINE_STATE_STREAM_BLEND_DESC            BlendState;
                CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL         DepthStencilState;
                CD3DX12_PIPELINE_STATE_STREAM_DEPTH_STENCIL_FORMAT  DSVFormat;
                CD3DX12_PIPELINE_STATE_STREAM_RENDER_TARGET_FORMATS RTVFormats;
                CD3DX12_PIPELINE_STATE_STREAM_SAMPLE_DESC           SampleDesc;
                CD3DX12_PIPELINE_STATE_STREAM_PRIMITIVE_TOPOLOGY    PrimitiveTopologyType;
            } stream;

            stream.RootSignature = mainRootSignature;
            stream.MS = D3D12_SHADER_BYTECODE{ ms.data(), ms.size() };
            stream.PS = D3D12_SHADER_BYTECODE{ ps.data(), ps.size() };

            CD3DX12_RASTERIZER_DESC raster(D3D12_DEFAULT);
            raster.FrontCounterClockwise = TRUE;
            raster.CullMode = D3D12_CULL_MODE_BACK;  // Always back-face cull (alpha mask uses discard, not NoCull)
            stream.RasterizerState = raster;
            stream.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

            CD3DX12_DEPTH_STENCIL_DESC ds(D3D12_DEFAULT);
            ds.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
            stream.DepthStencilState = ds;

            stream.DSVFormat = DXGI_FORMAT_D32_FLOAT;

            // Visibility-only output: a single R32_UINT visibility token
            // (GBuffer is reconstructed by the separate VisibilityGBuffer full-screen
            // resolve pass). directToGBuffer=true instead writes the classic 4-target
            // layout (albedo/normal/material/visToken) used when the Visibility
            // Buffer toggle is off (MeshletRasterizeGBufferMS.hlsl).
            D3D12_RT_FORMAT_ARRAY rtFormats = {};
            if (directToGBuffer)
            {
                rtFormats.NumRenderTargets = 4;
                rtFormats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                rtFormats.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
                rtFormats.RTFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;
                rtFormats.RTFormats[3] = DXGI_FORMAT_R32_UINT;
            }
            else
            {
                rtFormats.NumRenderTargets = 1;
                rtFormats.RTFormats[0] = DXGI_FORMAT_R32_UINT;
            }
            stream.RTVFormats = rtFormats;

            DXGI_SAMPLE_DESC sampleDesc = { 1, 0 };
            stream.SampleDesc = sampleDesc;
            stream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

            D3D12_PIPELINE_STATE_STREAM_DESC streamDesc = { sizeof(stream), &stream };
            CHECK_HR(device2->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&outPSO)), label);
        };

        // Visibility-only variant (used when m_UseVisibilityBuffer=true)
        {
            auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "MSMain", "ms_6_8");
            auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeMS.hlsl", "PSMain", "ps_6_8");
            buildMeshPSO(ms, ps, /*directToGBuffer=*/false, m_MeshletRasterPSO,
                         "[Meshlet] CreatePipelineState (mesh shader raster, visibility) failed");
        }

        // Direct-to-GBuffer variant (used when m_UseVisibilityBuffer=false)
        {
            auto ms = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeGBufferMS.hlsl", "MSMain", "ms_6_8");
            auto ps = GraphicsHelper::CompileShader("Shaders/MeshletRasterizeGBufferMS.hlsl", "PSMain", "ps_6_8");
            buildMeshPSO(ms, ps, /*directToGBuffer=*/true, m_MeshletRasterGBufferPSO,
                         "[Meshlet] CreatePipelineState (mesh shader raster, direct GBuffer) failed");
        }
    }

    // --- Indirect DispatchMesh command signature ---
    if (m_MeshShaderSupported)
    {
        D3D12_INDIRECT_ARGUMENT_DESC meshArg = {};
        meshArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        D3D12_COMMAND_SIGNATURE_DESC sigDesc = {};
        sigDesc.ByteStride = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
        sigDesc.NumArgumentDescs = 1;
        sigDesc.pArgumentDescs = &meshArg;
        CHECK_HR(device->CreateCommandSignature(&sigDesc, nullptr,
                 IID_PPV_ARGS(&m_DispatchMeshSignature)),
                 "[Meshlet] CreateCommandSignature (dispatch mesh) failed");
    }

    // --- Meshlet Debug View PSO (CS) for visibility buffer overlay ---
    {
        auto cs = GraphicsHelper::CompileShader("Shaders/VisibilityDebugView.hlsl", "DebugRenderCS", "cs_6_6");
        if (!cs.empty())
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = mainRootSignature;
            desc.CS = { cs.data(), cs.size() };
            CHECK_HR(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&m_MeshletDebugViewPSO)),
                     "[Meshlet] CreateComputePipelineState (debug view) failed");
        }
    }

    // --- VisibilityGBuffer PSO (full-screen resolve: visibility token -> albedo/normal/material) ---
    {
        std::vector<char> vs = GraphicsHelper::CompileShader("Shaders/VisibilityGBuffer.hlsl", "VSMain", "vs_6_8");
        std::vector<char> ps = GraphicsHelper::CompileShader("Shaders/VisibilityGBuffer.hlsl", "PSMain", "ps_6_8");
        if (!vs.empty() && !ps.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
            desc.pRootSignature = mainRootSignature;
            desc.VS = { reinterpret_cast<UINT8*>(vs.data()), vs.size() };
            desc.PS = { reinterpret_cast<UINT8*>(ps.data()), ps.size() };
            desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
            desc.DepthStencilState.DepthEnable = FALSE;
            desc.DepthStencilState.StencilEnable = FALSE;
            desc.SampleMask = UINT_MAX;
            desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            desc.SampleDesc.Count = 1;
            desc.NumRenderTargets = 3;
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;      // albedo
            desc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;  // normal
            desc.RTVFormats[2] = DXGI_FORMAT_R8G8B8A8_UNORM;      // material (roughness|metallic)
            desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
            CHECK_HR(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&m_VisibilityGBufferPSO)),
                     "[Meshlet] CreateGraphicsPipelineState (VisibilityGBuffer resolve) failed");
        }
    }

    std::cout << "[Meshlet] Pipelines created (no binning)" << std::endl;
}

void MeshletPass::BuildDispatchMeshArgs(ID3D12GraphicsCommandList* cmdList,
                                         ID3D12RootSignature* mainRootSignature,
                                         D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                                         int visibleMeshletsCounterSRVIdx, uint32_t phase)
{
    if (!m_BuildDispatchMeshArgsPSO)
        return;

    GPU_MARKER(cmdList, L"Meshlet BuildDispatchMeshArgs");

    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->SetComputeRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);

    // Pass RasterParams with counter SRV and dispatch args UAV via root constants b1 (param 12)
    RasterParams params = {};
    params.VisibleMeshletsIdx        = 0; // unused by BuildDispatchMeshArgsCS
    params.DispatchMeshArgsIdx       = (uint)m_DispatchMeshArgs.uavIndex;
    params.VisibleMeshletsCounterIdx = (uint)visibleMeshletsCounterSRVIdx;
    params.Phase                     = phase; // selects this phase's own VisibleMeshletsCounter slot
    cmdList->SetComputeRoot32BitConstants(12, sizeof(RasterParams) / 4, &params, 0);

    cmdList->SetPipelineState(m_BuildDispatchMeshArgsPSO.Get());
    cmdList->Dispatch(1, 1, 1);

    D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_DispatchMeshArgs.resource.Get());
    cmdList->ResourceBarrier(1, &barrier);
}

void MeshletPass::Rasterize(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                             D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                             int visibleMeshletsSRVIdx, int visibleMeshletsCounterSRVIdx,
                             bool useVisibilityBuffer, uint32_t phase)
{
    if (!model->IsMeshletReady() || !m_MeshShaderSupported)
        return;

    auto* pso = useVisibilityBuffer ? m_MeshletRasterPSO.Get() : m_MeshletRasterGBufferPSO.Get();
    if (!pso) return;

    GPU_MARKER(cmdList, L"Meshlet Rasterize (Mesh Shader, no binning)");

    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);
    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));
    cmdList->SetGraphicsRootDescriptorTable(14, GraphicsHelper::GetSRVGPUHandle((UINT)model->GetMeshletStreamSRVBase()));

    RasterParams rp = {};
    rp.VisibleMeshletsIdx        = (uint)visibleMeshletsSRVIdx;
    rp.DispatchMeshArgsIdx       = 0; // unused by rasterize
    // Phase 2's mesh shader needs Phase 1's final count as its base offset into
    // VisibleMeshlets[] — see docs/bug_flyingworld_meshlet_flicker.md.
    rp.VisibleMeshletsCounterIdx = (uint)visibleMeshletsCounterSRVIdx;
    rp.Phase                     = phase;
    cmdList->SetGraphicsRoot32BitConstants(12, sizeof(RasterParams) / 4, &rp, 0);

    cmdList->SetPipelineState(pso);
    cmdList->ExecuteIndirect(
        m_DispatchMeshSignature.Get(), 1,
        m_DispatchMeshArgs.resource.Get(),
        0, nullptr, 0);

    // Restore to UAV for next frame
    GraphicsHelper::TransitionResource(cmdList, m_DispatchMeshArgs,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void MeshletPass::ResolveVisibilityGBuffer(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                                            D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, Model* model,
                                            int visibleMeshletsSRVIdx)
{
    if (!m_VisibilityGBufferPSO || !model->IsMeshletReady() || !m_MeshShaderSupported)
        return;

    GPU_MARKER(cmdList, L"Visibility GBuffer Resolve (Full-Screen)");

    // Matches VisibilityGBufferParams in Shaders/VisibilityGBuffer.hlsl (register b1/ root param 12).
    struct VisibilityGBufferParams
    {
        uint32_t VisBufSRVIdx;
        uint32_t CandidatesSRVIdx;
    } rp = {};
    rp.VisBufSRVIdx     = (uint32_t)m_VisibilityBuffer.srvIndex;
    rp.CandidatesSRVIdx = (uint32_t)visibleMeshletsSRVIdx;

    cmdList->SetGraphicsRootSignature(mainRootSignature);
    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);
    cmdList->SetGraphicsRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));
    cmdList->SetGraphicsRootDescriptorTable(14, GraphicsHelper::GetSRVGPUHandle((UINT)model->GetMeshletStreamSRVBase()));
    cmdList->SetGraphicsRoot32BitConstants(12, sizeof(rp) / 4, &rp, 0);

    cmdList->SetPipelineState(m_VisibilityGBufferPSO.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->DrawInstanced(3, 1, 0, 0); // Fullscreen triangle
}
