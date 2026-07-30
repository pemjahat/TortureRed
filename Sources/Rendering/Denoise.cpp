#include "pch.h"

#include "Denoise.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"
#include "NRI.h"
#include "Extensions/NRIHelper.h"
#include "Extensions/NRIWrapperD3D12.h"
#include "NRD.h"
#include "NRDIntegration.hpp"

namespace
{
    constexpr nrd::Identifier kNrdRelaxDiffuseSpecularIdentifier = 1u;

    nri::AccessLayoutStage ToNriAccessLayoutStage(D3D12_RESOURCE_STATES state)
    {
        if ((state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) != 0)
            return {nri::AccessBits::SHADER_RESOURCE_STORAGE, nri::Layout::SHADER_RESOURCE_STORAGE, nri::StageBits::COMPUTE_SHADER};

        if ((state & (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_GENERIC_READ)) != 0)
            return {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE, nri::StageBits::COMPUTE_SHADER};

        return {nri::AccessBits::NONE, nri::Layout::GENERAL, nri::StageBits::NONE};
    }

    nrd::Resource MakeNrdResource(GPUTexture& texture)
    {
        nrd::Resource resource = {};
        resource.d3d12.resource = texture.resource.Get();
        resource.d3d12.format = static_cast<DXGIFormat>(texture.format);
        resource.userArg = &texture;
        resource.state = ToNriAccessLayoutStage(texture.state);
        return resource;
    }

    void CopyMatrixToNrd(float dst[16], DirectX::FXMMATRIX matrix)
    {
        DirectX::XMFLOAT4X4 tmp;
        DirectX::XMStoreFloat4x4(&tmp, matrix);
        std::memcpy(dst, &tmp, sizeof(tmp));
    }
}

Denoise::Denoise() = default;
Denoise::~Denoise() = default;

bool Denoise::CreateResources(uint32_t internalWidth, uint32_t internalHeight)
{
    CreateTexture(m_NrdMotionVectorsTex, internalWidth, internalHeight, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdMotionVectors");
    CreateTexture(m_NrdNormalRoughnessTex, internalWidth, internalHeight, DXGI_FORMAT_R10G10B10A2_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdNormalRoughness");
    CreateTexture(m_NrdViewZTex, internalWidth, internalHeight, DXGI_FORMAT_R16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdViewZ");
    CreateTexture(m_NrdRelaxDiffuseTex, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdRelaxDiffuse");
    CreateTexture(m_NrdRelaxSpecularTex, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdRelaxSpecular");
    CreateTexture(m_NrdDenoisedDiffuseTex, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdDenoisedDiffuse");
    CreateTexture(m_NrdDenoisedSpecularTex, internalWidth, internalHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdDenoisedSpecular");
    CreateTexture(m_NrdValidationTex, internalWidth, internalHeight, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, 1, 1, "Tex_NrdValidation");
    return true;
}

void Denoise::OnResolutionChanged(uint32_t w, uint32_t h)
{
    CreateResources(w, h);
}

void Denoise::CreatePipelines(ID3D12Device* device, ID3D12RootSignature* rootSignature)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC computeDesc = {};
    computeDesc.pRootSignature = rootSignature;

    auto nrdGuidesCS    = GraphicsHelper::CompileShader("Shaders/NrdPrepareGuides.hlsl",     "main", "cs_6_6");
    auto nrdCompositeCS = GraphicsHelper::CompileShader("Shaders/NrdCompositeIndirect.hlsl", "main", "cs_6_6");
    auto nrdPackNoiseCS = GraphicsHelper::CompileShader("Shaders/NrdPackNoise.hlsl",         "main", "cs_6_6");

    if (!nrdGuidesCS.empty())
    {
        computeDesc.CS = { nrdGuidesCS.data(), nrdGuidesCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdPrepareGuidesPSO));
    }

    if (!nrdCompositeCS.empty())
    {
        computeDesc.CS = { nrdCompositeCS.data(), nrdCompositeCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdCompositePSO));
    }

    if (!nrdPackNoiseCS.empty())
    {
        computeDesc.CS = { nrdPackNoiseCS.data(), nrdPackNoiseCS.size() };
        device->CreateComputePipelineState(&computeDesc, IID_PPV_ARGS(&m_NrdPackNoisePSO));
    }
}

bool Denoise::Initialize(ID3D12Device* device, ID3D12CommandQueue* commandQueue, uint32_t internalWidth, uint32_t internalHeight)
{
    if (m_NrdInitialized)
        return true;

    nri::QueueFamilyD3D12Desc queueFamily = {};
    ID3D12CommandQueue* queue = commandQueue;
    queueFamily.d3d12Queues = &queue;
    queueFamily.queueNum = 1;
    queueFamily.queueType = nri::QueueType::GRAPHICS;

    nri::DeviceCreationD3D12Desc deviceDesc = {};
    deviceDesc.d3d12Device = device;
    deviceDesc.queueFamilies = &queueFamily;
    deviceDesc.queueFamilyNum = 1;
    deviceDesc.disableD3D12EnhancedBarriers = true;

    const nrd::DenoiserDesc denoisers[] = {
        { kNrdRelaxDiffuseSpecularIdentifier, nrd::Denoiser::RELAX_DIFFUSE_SPECULAR }
    };

    nrd::InstanceCreationDesc instanceDesc = {};
    instanceDesc.denoisers = denoisers;
    instanceDesc.denoisersNum = _countof(denoisers);

    nrd::IntegrationCreationDesc integrationDesc = {};
    strcpy_s(integrationDesc.name, "TortureRedNRD");
    integrationDesc.resourceWidth = static_cast<uint16_t>(internalWidth);
    integrationDesc.resourceHeight = static_cast<uint16_t>(internalHeight);
    integrationDesc.queuedFrameNum = 1;
    integrationDesc.enableWholeLifetimeDescriptorCaching = false;
    integrationDesc.autoWaitForIdle = true;

    m_NrdIntegration = std::make_unique<nrd::Integration>();
    const nrd::Result result = m_NrdIntegration->RecreateD3D12(integrationDesc, instanceDesc, deviceDesc);
    if (result != nrd::Result::SUCCESS)
    {
        std::cerr << "Failed to initialize NRD integration" << std::endl;
        m_NrdIntegration.reset();
        return false;
    }

    m_NrdInitialized = true;
    return true;
}

void Denoise::Shutdown()
{
    if (m_NrdIntegration)
    {
        m_NrdIntegration->Destroy();
        m_NrdIntegration.reset();
    }

    m_NrdInitialized = false;
}

bool Denoise::Execute(ID3D12GraphicsCommandList* cmdList, ID3D12CommandAllocator* cmdAllocator,
                       ID3D12RootSignature* rootSignature, D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                       const FrameConstants& frame, GPUTexture& finalDiffuseTex, GPUTexture& finalSpecularTex,
                       uint32_t internalWidth, uint32_t internalHeight)
{
    if (!m_NrdPrepareGuidesPSO || !m_NrdCompositePSO || !m_NrdInitialized)
        return false;

    if (!m_NrdPackNoisePSO)
        return false;

    GraphicsHelper::TransitionResource(cmdList, m_NrdMotionVectorsTex,    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdNormalRoughnessTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdViewZTex,            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdDenoisedDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdDenoisedSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdValidationTex,       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    // ---- Step 1: NRD Prepare Guides (unchanged) ----
    cmdList->SetPipelineState(m_NrdPrepareGuidesPSO.Get());
    BindlessIndices indices = {};
    indices.OutputIdx0 = m_NrdMotionVectorsTex.uavIndex;
    indices.OutputIdx1 = m_NrdNormalRoughnessTex.uavIndex;
    indices.OutputIdx2 = m_NrdViewZTex.uavIndex;
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("NRD_PrepareGuides", MP_GREEN);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    D3D12_RESOURCE_BARRIER guideBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdMotionVectorsTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdNormalRoughnessTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdViewZTex.resource.Get())
    };
    cmdList->ResourceBarrier(_countof(guideBarriers), guideBarriers);

    // ---- Step 2: NrdPackNoise — Final* → RELAX format ----
    // finalDiffuseTex/finalSpecularTex already contain the merged DI+GI signal (written by SSO calls).
    GraphicsHelper::TransitionResource(cmdList, finalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, finalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_NrdRelaxDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, m_NrdRelaxSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->SetPipelineState(m_NrdPackNoisePSO.Get());
    indices = {};
    indices.InputIdx0  = finalDiffuseTex.srvIndex;
    indices.InputIdx1  = finalSpecularTex.srvIndex;
    indices.OutputIdx0 = m_NrdRelaxDiffuseTex.uavIndex;
    indices.OutputIdx1 = m_NrdRelaxSpecularTex.uavIndex;
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("NRD_PackNoise", MP_GREEN);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    D3D12_RESOURCE_BARRIER relaxBarriers[] = {
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdRelaxDiffuseTex.resource.Get()),
        CD3DX12_RESOURCE_BARRIER::UAV(m_NrdRelaxSpecularTex.resource.Get()),
    };
    cmdList->ResourceBarrier(_countof(relaxBarriers), relaxBarriers);

    DirectX::XMMATRIX projectionInverse = DirectX::XMLoadFloat4x4(&frame.projectionInverse);
    DirectX::XMMATRIX projection = DirectX::XMMatrixInverse(nullptr, projectionInverse);
    DirectX::XMMATRIX view = DirectX::XMMatrixInverse(nullptr, DirectX::XMLoadFloat4x4(&frame.viewInverse));
    DirectX::XMMATRIX viewPrev = DirectX::XMMatrixInverse(nullptr, DirectX::XMLoadFloat4x4(&frame.viewInversePrevious));

    nrd::CommonSettings commonSettings = {};
    CopyMatrixToNrd(commonSettings.viewToClipMatrix, projection);
    CopyMatrixToNrd(commonSettings.viewToClipMatrixPrev, projection);
    CopyMatrixToNrd(commonSettings.worldToViewMatrix, view);
    CopyMatrixToNrd(commonSettings.worldToViewMatrixPrev, viewPrev);
    commonSettings.motionVectorScale[0] = 1.0f;
    commonSettings.motionVectorScale[1] = 1.0f;
    commonSettings.motionVectorScale[2] = 0.0f;
    commonSettings.resourceSize[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.resourceSize[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.resourceSizePrev[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.resourceSizePrev[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.rectSize[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.rectSize[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.rectSizePrev[0] = static_cast<uint16_t>(frame.screenWidth);
    commonSettings.rectSizePrev[1] = static_cast<uint16_t>(frame.screenHeight);
    commonSettings.denoisingRange = 1000.0f;
    commonSettings.disocclusionThreshold = 0.01f;
    commonSettings.frameIndex = frame.frameIndex;
    // Force RESTART when NRD was inactive on the previous frame (e.g. raster
    // indirect GI was toggled off then back on).  This resets NRD's internal
    // m_PrevFrameIndexFromSettings so the +1-per-frame assertion won't fire.
    const bool needRestart = (frame.frameIndex <= 1) || !m_NrdWasActiveLastFrame;
    commonSettings.accumulationMode = needRestart ? nrd::AccumulationMode::RESTART : nrd::AccumulationMode::CONTINUE;
    commonSettings.enableValidation = frame.enableNrdValidation != 0;

    nrd::RelaxSettings relaxSettings = {};
    relaxSettings.diffuseMaxAccumulatedFrameNum = 12;
    relaxSettings.specularMaxAccumulatedFrameNum = 8;
    relaxSettings.diffuseMaxFastAccumulatedFrameNum = 4;
    relaxSettings.specularMaxFastAccumulatedFrameNum = 3;
    relaxSettings.diffusePrepassBlurRadius = 20.0f;
    relaxSettings.specularPrepassBlurRadius = 40.0f;
    relaxSettings.minHitDistanceWeight = 0.05f;
    relaxSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_3X3;
    relaxSettings.enableAntiFirefly = true;

    m_NrdIntegration->NewFrame();
    if (m_NrdIntegration->SetCommonSettings(commonSettings) != nrd::Result::SUCCESS)
        return false;
    if (m_NrdIntegration->SetDenoiserSettings(kNrdRelaxDiffuseSpecularIdentifier, &relaxSettings) != nrd::Result::SUCCESS)
        return false;

    nrd::ResourceSnapshot resourceSnapshot = {};
    resourceSnapshot.restoreInitialState = true;
    resourceSnapshot.SetResource(nrd::ResourceType::IN_MV, MakeNrdResource(m_NrdMotionVectorsTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS, MakeNrdResource(m_NrdNormalRoughnessTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_VIEWZ, MakeNrdResource(m_NrdViewZTex));
    // NRD reads RELAX-packed textures from NrdPackNoise output.
    resourceSnapshot.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, MakeNrdResource(m_NrdRelaxDiffuseTex));
    resourceSnapshot.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, MakeNrdResource(m_NrdRelaxSpecularTex));
    resourceSnapshot.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, MakeNrdResource(m_NrdDenoisedDiffuseTex));
    resourceSnapshot.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, MakeNrdResource(m_NrdDenoisedSpecularTex));
    if (frame.enableNrdValidation != 0)
    {
        resourceSnapshot.SetResource(nrd::ResourceType::OUT_VALIDATION, MakeNrdResource(m_NrdValidationTex));
    }

    nri::CommandBufferD3D12Desc commandBufferDesc = {};
    commandBufferDesc.d3d12CommandList = cmdList;
    commandBufferDesc.d3d12CommandAllocator = cmdAllocator;

    const nrd::Identifier denoisers[] = { kNrdRelaxDiffuseSpecularIdentifier };
    {
        MICROPROFILE_SCOPEGPUI("NRD_RELAX", MP_GREEN);
        m_NrdIntegration->DenoiseD3D12(denoisers, _countof(denoisers), commandBufferDesc, resourceSnapshot);
    }

    GraphicsHelper::TransitionResource(cmdList, m_NrdDenoisedDiffuseTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, m_NrdDenoisedSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (frame.enableNrdValidation != 0)
    {
        GraphicsHelper::TransitionResource(cmdList, m_NrdValidationTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    ID3D12DescriptorHeap* heaps[] = { GraphicsHelper::GetSRVHeap() };
    cmdList->SetDescriptorHeaps(_countof(heaps), heaps);
    cmdList->SetComputeRootSignature(rootSignature);
    cmdList->SetComputeRootConstantBufferView(0, frameCBAddress);
    cmdList->SetComputeRootDescriptorTable(3, GraphicsHelper::GetSRVGPUHandle(0));

    // The composite pass reads NRD output (SRV) and writes denoised radiance back to Final* (UAV).
    // This is a circular write-back: Final* was read by NrdPackNoise, now overwritten with denoised data.
    GraphicsHelper::TransitionResource(cmdList, finalDiffuseTex,  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    GraphicsHelper::TransitionResource(cmdList, finalSpecularTex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cmdList->SetPipelineState(m_NrdCompositePSO.Get());
    indices = {};
    indices.InputIdx0 = m_NrdDenoisedDiffuseTex.srvIndex;
    indices.InputIdx1 = m_NrdDenoisedSpecularTex.srvIndex;
    indices.InputIdx2 = frame.enableNrdValidation != 0 ? m_NrdValidationTex.srvIndex : UINT(-1);
    indices.OutputIdx0 = finalDiffuseTex.uavIndex;
    indices.OutputIdx1 = finalSpecularTex.uavIndex;
    cmdList->SetComputeRoot32BitConstants(12, sizeof(BindlessIndices) / 4, &indices, 0);
    {
        MICROPROFILE_SCOPEGPUI("NRD_Composite", MP_GREEN);
        cmdList->Dispatch((internalWidth + 7) / 8, (internalHeight + 7) / 8, 1);
    }

    // Transition Final* to pixel-shader-readable SRV for Lighting.hlsl
    GraphicsHelper::TransitionResource(cmdList, finalDiffuseTex,  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    GraphicsHelper::TransitionResource(cmdList, finalSpecularTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    m_NrdWasActiveLastFrame = true;
    return true;
}
