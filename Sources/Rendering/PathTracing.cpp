#include "pch.h"

#include "PathTracing.h"
#include "Renderer.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"
#include "Rtxdi/RtxdiUtils.h"
#include "Rtxdi/GI/ReSTIRGIParameters.h"

bool PathTracing::CreateResources(ID3D12Device* device, bool rayTracingSupported, uint32_t internalWidth, uint32_t internalHeight)
{
    m_RayTracingSupported = rayTracingSupported;
    if (!m_RayTracingSupported)
        return true;

    if (!CreateTexture(m_AccumulationBuffer, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_AccumulationBuffer"))
    {
        std::cerr << "Failed to create accumulation buffer" << std::endl;
        return false;
    }

    if (!CreateTexture(m_PathTracerOutput, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_PathTracerOutput"))
    {
        std::cerr << "Failed to create path tracer output texture" << std::endl;
        return false;
    }

    if (!CreateTexture(m_PathTracerPresentOutput, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_PathTracerPresentOutput"))
    {
        std::cerr << "Failed to create path tracer present texture" << std::endl;
        return false;
    }

    if (!CreateTexture(m_RestirDebugHeatmap, WINDOW_WIDTH, WINDOW_HEIGHT, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_RestirDebugHeatmap"))
    {
        std::cerr << "Failed to create ReSTIR debug heatmap texture" << std::endl;
        return false;
    }

    // Create ReSTIR Reservoirs — initially at WINDOW_WIDTH x WINDOW_HEIGHT.
    // Will be recreated at internal resolution by OnResolutionChanged().
    for (int i = 0; i < 2; ++i)
    {
        if (!CreateStructuredBuffer(m_ReservoirBuffer[i], sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_Reservoir0" : "SB_Reservoir1"))
        {
            std::cerr << "Failed to create ReSTIR reservoir buffer" << std::endl;
            return false;
        }
    }

    if (!CreateStructuredBuffer(m_ReservoirIntermediate, sizeof(Reservoir), WINDOW_WIDTH * WINDOW_HEIGHT, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_ReservoirIntermediate"))
    {
        std::cerr << "Failed to create ReSTIR intermediate reservoir buffer" << std::endl;
        return false;
    }

    // Create Path Visualization Line Buffer (small: MAX_PATH_VIZ_LINES * sizeof(PathVizLine))
    if (!CreateStructuredBuffer(m_PathVizLineBuffer, sizeof(PathVizLine), MAX_PATH_VIZ_LINES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_PathVizLineBuffer"))
    {
        std::cerr << "Failed to create PathViz line buffer" << std::endl;
        return false;
    }

    // Create RTXDI Reservoirs
    // See RtxdiUtils.cpp: CalculateReservoirBufferParameters
    uint32_t renderWidthBlocks = (internalWidth + 15) / 16;
    uint32_t renderHeightBlocks = (internalHeight + 15) / 16;
    uint32_t reservoirArrayPitch = renderWidthBlocks * 256 * renderHeightBlocks;

    for (int i = 0; i < 2; ++i)
    {
        if (!CreateStructuredBuffer(m_RtxdiReservoirBuffer[i], sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_RtxdiReservoir0" : "SB_RtxdiReservoir1"))
        {
            std::cerr << "Failed to create RTXDI reservoir buffer " << i << std::endl;
            return false;
        }
    }

    if (!CreateStructuredBuffer(m_RtxdiReservoirIntermediate, sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_RtxdiReservoirIntermediate"))
    {
        std::cerr << "Failed to create RTXDI intermediate reservoir buffer" << std::endl;
        return false;
    }

    // Create Neighbor Offsets Buffer
    const uint32_t neighborOffsetCount = 8192;
    if (!CreateBuffer(m_RtxdiNeighborOffsetsBuffer, neighborOffsetCount * 2, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "SB_RtxdiNeighborOffsets"))
    {
        std::cerr << "Failed to create RTXDI neighbor offset buffer" << std::endl;
        return false;
    }

    std::vector<uint8_t> offsets(neighborOffsetCount * 2);
    rtxdi::FillNeighborOffsetBuffer(offsets.data(), neighborOffsetCount);
    memcpy(m_RtxdiNeighborOffsetsBuffer.cpuPtr, offsets.data(), offsets.size());

    // Create Typed SRV for Neighbor Offsets
    m_RtxdiNeighborOffsetsBuffer.srvIndex = GraphicsHelper::AllocateSRV();
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8G8_SNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = neighborOffsetCount;
    srvDesc.Buffer.StructureByteStride = 0;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    device->CreateShaderResourceView(m_RtxdiNeighborOffsetsBuffer.resource.Get(), &srvDesc, GraphicsHelper::GetSRVCPUHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));

    return true;
}

void PathTracing::OnResolutionChanged(uint32_t w, uint32_t h)
{
    if (!m_RayTracingSupported)
        return;

    CreateTexture(m_AccumulationBuffer, w, h, DXGI_FORMAT_R32G32B32A32_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_AccumulationBuffer");
    CreateTexture(m_PathTracerOutput, w, h, DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_PathTracerOutput");
    CreateTexture(m_PathTracerPresentOutput, w, h, DXGI_FORMAT_R8G8B8A8_UNORM,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_PathTracerPresentOutput");
    CreateTexture(m_RestirDebugHeatmap, w, h, DXGI_FORMAT_R16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_RestirDebugHeatmap");

    // Manual ReSTIR Reservoirs
    for (int i = 0; i < 2; ++i)
        CreateStructuredBuffer(m_ReservoirBuffer[i], sizeof(Reservoir), w * h,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_Reservoir0" : "SB_Reservoir1");
    CreateStructuredBuffer(m_ReservoirIntermediate, sizeof(Reservoir), w * h,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_ReservoirIntermediate");

    // RTXDI Reservoirs
    uint32_t renderWidthBlocks = (w + 15) / 16;
    uint32_t renderHeightBlocks = (h + 15) / 16;
    uint32_t reservoirArrayPitch = renderWidthBlocks * 256 * renderHeightBlocks;
    for (int i = 0; i < 2; ++i)
        CreateStructuredBuffer(m_RtxdiReservoirBuffer[i], sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, i == 0 ? "SB_RtxdiReservoir0" : "SB_RtxdiReservoir1");
    CreateStructuredBuffer(m_RtxdiReservoirIntermediate, sizeof(RTXDI_PackedGIReservoir), reservoirArrayPitch,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "SB_RtxdiReservoirIntermediate");
}

void PathTracing::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    if (!m_RayTracingSupported)
        return;

    // Load Path Tracer shader
    auto pathTracerCode = GraphicsHelper::CompileShader("Shaders/PathTracer.hlsl", "CSMain", "cs_6_6");
    if (!pathTracerCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { pathTracerCode.data(), pathTracerCode.size() };
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPSO)), "Failed to create Path Tracer Compute PSO");
    }

    auto pathTracerPresentCode = GraphicsHelper::CompileShader("Shaders/PathTracerPresent.hlsl", "CSMain", "cs_6_6");
    if (!pathTracerPresentCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { pathTracerPresentCode.data(), pathTracerPresentCode.size() };
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_PathTracerPresentPSO)), "Failed to create Path Tracer Present PSO");
    }

    // Load ReSTIR Multi-pass shaders
    auto restirTemporalCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!restirTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { restirTemporalCode.data(), restirTemporalCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirTemporalPSO)), "Failed to create ReSTIR Temporal PSO");
    }

    auto restirSpatialCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!restirSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { restirSpatialCode.data(), restirSpatialCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirSpatialPSO)), "Failed to create ReSTIR Spatial PSO");
    }

    auto restirResolveCode = GraphicsHelper::CompileShader("Shaders/RestirGI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!restirResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { restirResolveCode.data(), restirResolveCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirResolvePSO)), "Failed to create ReSTIR Resolve PSO");
    }

    auto restirDebugCode = GraphicsHelper::CompileShader("Shaders/RestirGI_ReservoirDebug.hlsl", "main", "cs_6_6");
    if (!restirDebugCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { restirDebugCode.data(), restirDebugCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RestirReservoirDebugPSO)), "Failed to create ReSTIR Reservoir Debug PSO");
    }

    // RTXDI PSOs
    auto rtxdiTemporalCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Temporal.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiTemporalCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { rtxdiTemporalCode.data(), rtxdiTemporalCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirTemporalPSO)), "Failed to create RTXDI Temporal PSO");
    }

    auto rtxdiSpatialCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Spatial.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiSpatialCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { rtxdiSpatialCode.data(), rtxdiSpatialCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirSpatialPSO)), "Failed to create RTXDI Spatial PSO");
    }

    auto rtxdiResolveCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Resolve.hlsl", "CSMain", "cs_6_6");
    if (!rtxdiResolveCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { rtxdiResolveCode.data(), rtxdiResolveCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirResolvePSO)), "Failed to create RTXDI Resolve PSO");
    }

    auto rtxdiDebugCode = GraphicsHelper::CompileShader("Shaders/RestirGI_RTXDI_Debug.hlsl", "main", "cs_6_6");
    if (!rtxdiDebugCode.empty())
    {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS = { rtxdiDebugCode.data(), rtxdiDebugCode.size() };
        CHECK_HR(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_RtxdiRestirReservoirDebugPSO)), "Failed to create RTXDI Reservoir Debug PSO");
    }

    // Path Visualization Lines PSO (graphics pipeline, line list, no depth)
    {
        auto vsCode = GraphicsHelper::CompileShader("Shaders/PathVizLines.hlsl", "VSMain", "vs_6_6");
        auto psCode = GraphicsHelper::CompileShader("Shaders/PathVizLines.hlsl", "PSMain", "ps_6_6");
        if (!vsCode.empty() && !psCode.empty())
        {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.pRootSignature = rootSignature;
            psoDesc.VS = { vsCode.data(), vsCode.size() };
            psoDesc.PS = { psCode.data(), psCode.size() };
            psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            psoDesc.NumRenderTargets = 1;
            psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
            psoDesc.SampleDesc.Count = 1;
            psoDesc.SampleMask = UINT_MAX;
            psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
            psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
            psoDesc.DepthStencilState.DepthEnable = FALSE;
            psoDesc.DepthStencilState.StencilEnable = FALSE;
            CHECK_HR(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PathVizLinePSO)), "Failed to create PathViz Lines PSO");
        }
    }
}

void PathTracing::DispatchRays(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                                Model* model, const FrameConstants& frame,
                                D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress, D3D12_GPU_VIRTUAL_ADDRESS tlasGPUAddress,
                                D3D12_GPU_VIRTUAL_ADDRESS lightsBufferAddress, D3D12_GPU_VIRTUAL_ADDRESS lightLUTBufferAddress,
                                uint32_t internalWidth, uint32_t internalHeight)
{
    if (!model) return;

    const bool useCustomRestirHeatmap = frame.enableRestir && !frame.useRTXDI &&
        frame.restirReservoirDebugMode >= RESTIR_RESERVOIR_DEBUG_SOURCE_PDF;

    // Transition UAVs
    GraphicsHelper::TransitionResource(cmdList, m_AccumulationBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_PathTracerOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_PathTracerPresentOutput, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_ReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_ReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_ReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_RtxdiReservoirBuffer[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_RtxdiReservoirBuffer[1], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_RtxdiReservoirIntermediate, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_PathVizLineBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (useCustomRestirHeatmap)
    {
        GraphicsHelper::TransitionResource(cmdList, m_RestirDebugHeatmap, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    D3D12_RESOURCE_BARRIER uavBarriers[7];
    uavBarriers[0] = CD3DX12_RESOURCE_BARRIER::UAV(m_AccumulationBuffer.resource.Get());
    uavBarriers[1] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
    uavBarriers[2] = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerPresentOutput.resource.Get());
    uavBarriers[3] = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
    uavBarriers[4] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[0].resource.Get());
    uavBarriers[5] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[1].resource.Get());
    uavBarriers[6] = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirIntermediate.resource.Get());
    cmdList->ResourceBarrier(7, uavBarriers);

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetComputeRootSignature(rootSignature);

    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootShaderResourceView(1, model->GetMaterialBufferAddress());
    cmdList->SetComputeRootShaderResourceView(2, model->GetDrawNodeBufferAddress());
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless
    cmdList->SetComputeRootShaderResourceView(4, tlasGPUAddress);
    cmdList->SetComputeRootShaderResourceView(5, model->GetGlobalIndexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(6, model->GetGlobalVertexBufferAddress());
    cmdList->SetComputeRootShaderResourceView(10, lightsBufferAddress); // Lights Buffer
    cmdList->SetComputeRootShaderResourceView(11, lightLUTBufferAddress); // Light LUT Buffer

    BindlessIndices indices = {};

    int currentReservoir = m_CurrentReservoirIndex;
    int previousReservoir = 1 - currentReservoir;

    if (frame.useRTXDI)
    {
        // NVIDIA RTXDI Path
        // Bind common RTXDI resources
        cmdList->SetComputeRootDescriptorTable(9, GraphicsHelper::GetSRVGPUHandle(m_RtxdiNeighborOffsetsBuffer.srvIndex));

        // Pass 1: Temporal Resampling
        cmdList->SetPipelineState(m_RtxdiRestirTemporalPSO.Get());
        cmdList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        cmdList->SetComputeRootDescriptorTable(8, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[previousReservoir].uavIndex));
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier1 = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirBuffer[currentReservoir].resource.Get());
        cmdList->ResourceBarrier(1, &barrier1);

        // Pass 2: Spatial Resampling
        cmdList->SetPipelineState(m_RtxdiRestirSpatialPSO.Get());
        cmdList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
        cmdList->SetComputeRootDescriptorTable(8, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirBuffer[currentReservoir].uavIndex));
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_RtxdiReservoirIntermediate.resource.Get());
        cmdList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve
        cmdList->SetPipelineState(m_RtxdiRestirResolvePSO.Get());
        cmdList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices        
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        if (frame.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF && m_RtxdiRestirReservoirDebugPSO)
        {
            D3D12_RESOURCE_BARRIER debugBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
            cmdList->ResourceBarrier(1, &debugBarrier);

            cmdList->SetPipelineState(m_RtxdiRestirReservoirDebugPSO.Get());
            cmdList->SetComputeRootDescriptorTable(7, GraphicsHelper::GetSRVGPUHandle(m_RtxdiReservoirIntermediate.uavIndex));
            indices.OutputIdx0 = m_PathTracerOutput.uavIndex;
            cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
            cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
        }
    }
    else if (frame.enableRestir)
    {
        // Torture ReSTIR (Manual Implementation)
        // Pass 1: Temporal — writes to ReservoirBuffer[current], reads history from ReservoirBuffer[previous]
        cmdList->SetPipelineState(m_RestirTemporalPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[previousReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirBuffer[currentReservoir].uavIndex;
        indices.OutputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
        indices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barriers1[2] = {
            CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirBuffer[currentReservoir].resource.Get()),
            CD3DX12_RESOURCE_BARRIER::UAV(m_PathVizLineBuffer.resource.Get())
        };
        cmdList->ResourceBarrier(2, barriers1);

        // Pass 2: Spatial — writes to Intermediate, reads temporal from ReservoirBuffer[current]
        cmdList->SetPipelineState(m_RestirSpatialPSO.Get());
        indices.InputIdx0 = m_ReservoirBuffer[currentReservoir].srvIndex;
        indices.OutputIdx0 = m_ReservoirIntermediate.uavIndex;
        indices.OutputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.uavIndex : UINT(-1);
        indices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER barrier2 = CD3DX12_RESOURCE_BARRIER::UAV(m_ReservoirIntermediate.resource.Get());
        cmdList->ResourceBarrier(1, &barrier2);

        // Pass 3: Resolve — reads spatial output from Intermediate
        cmdList->SetPipelineState(m_RestirResolvePSO.Get());
        indices.InputIdx0 = m_ReservoirIntermediate.srvIndex;
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);

        if (frame.restirReservoirDebugMode != RESTIR_RESERVOIR_DEBUG_OFF && m_RestirReservoirDebugPSO)
        {
            D3D12_RESOURCE_BARRIER debugBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
            cmdList->ResourceBarrier(1, &debugBarrier);

            if (useCustomRestirHeatmap)
            {
                GraphicsHelper::TransitionResource(cmdList, m_RestirDebugHeatmap, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }

            cmdList->SetPipelineState(m_RestirReservoirDebugPSO.Get());
            indices.InputIdx0 = m_ReservoirIntermediate.srvIndex;
            indices.InputIdx1 = useCustomRestirHeatmap ? m_RestirDebugHeatmap.srvIndex : UINT(-1);
            indices.OutputIdx0 = m_PathTracerOutput.uavIndex;
            cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
            cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
        }
    }
    else
    {
        // Old Path Trace
        indices.OutputIdx0 = m_AccumulationBuffer.uavIndex;
        indices.OutputIdx1 = m_PathTracerOutput.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0); // b1: Bindless indices
        cmdList->SetPipelineState(m_PathTracerPSO.Get());
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    m_CurrentReservoirIndex = previousReservoir; // Swap for next frame

    if (m_PathTracerPresentPSO)
    {
        D3D12_RESOURCE_BARRIER presentBarrier = CD3DX12_RESOURCE_BARRIER::UAV(m_PathTracerOutput.resource.Get());
        cmdList->ResourceBarrier(1, &presentBarrier);

        cmdList->SetPipelineState(m_PathTracerPresentPSO.Get());
        indices.InputIdx0 = m_PathTracerOutput.srvIndex;
        indices.OutputIdx0 = m_PathTracerPresentOutput.uavIndex;
        cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    // Transition for blitting/Imgui
    GraphicsHelper::TransitionResource(cmdList, m_PathTracerPresentOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);
}

void PathTracing::DrawPathVizLines(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* rootSignature,
                                    D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                                    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV)
{
    if (!m_PathVizLinePSO || !m_PathVizLineBuffer.resource) return;

    // Transition buffer from UAV (written by compute) to SRV-readable for VS
    GraphicsHelper::TransitionResource(cmdList, m_PathVizLineBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    cmdList->OMSetRenderTargets(1, &backBufferRTV, FALSE, nullptr);

    D3D12_VIEWPORT viewport = CD3DX12_VIEWPORT(0.0f, 0.0f, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    D3D12_RECT scissorRect = CD3DX12_RECT(0, 0, (LONG)WINDOW_WIDTH, (LONG)WINDOW_HEIGHT);
    cmdList->RSSetViewports(1, &viewport);
    cmdList->RSSetScissorRects(1, &scissorRect);

    cmdList->SetDescriptorHeaps(1, GraphicsHelper::GetSRVHeapAddress());
    cmdList->SetGraphicsRootSignature(rootSignature);
    cmdList->SetGraphicsRootConstantBufferView(0, frameCBAddress);
    cmdList->SetGraphicsRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0)); // Bindless

    BindlessIndices vizIndices = {};
    vizIndices.PathVizLineBufferIdx = (uint32_t)m_PathVizLineBuffer.srvIndex;
    cmdList->SetGraphicsRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &vizIndices, 0);

    cmdList->SetPipelineState(m_PathVizLinePSO.Get());
    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
    cmdList->DrawInstanced(MAX_PATH_VIZ_LINES * 2, 1, 0, 0);
}
