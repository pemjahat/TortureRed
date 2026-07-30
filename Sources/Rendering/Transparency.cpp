#include "pch.h"

#include "Transparency.h"
#include "Core/Model.h"
#include "Renderer.h"
#include "Graphics/GraphicsHelper.h"

void Transparency::Execute(ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer,
                            const DirectX::BoundingFrustum& frustum, bool rasterTaaActive,
                            uint32_t outputWidth, uint32_t outputHeight,
                            ID3D12PipelineState* transparentPSO, ID3D12PipelineState* transparentHdrPSO)
{
    MICROPROFILE_SCOPEI("Render", "Transparency", MP_ORANGE);
    MICROPROFILE_SCOPEGPUI("Transparency", MP_ORANGE);
    GPU_MARKER(cmdList, L"Transparency Pass");

    GBuffer& gbuffer = renderer->GetGBuffer();

    // Ensure depth is in read state for forward pass
    GraphicsHelper::TransitionResource(cmdList, gbuffer.depth, D3D12_RESOURCE_STATE_DEPTH_READ);

    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = gbuffer.depth.dsvHandle;

    if (rasterTaaActive)
    {
        // Render into the HDR intermediate texture (same target as the geometry pass).
        // Viewport stays at internal resolution — already set at the top of Render().
        GraphicsHelper::TransitionResource(cmdList, renderer->GetRasterHdrOutputTex(), D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_CPU_DESCRIPTOR_HANDLE hdrRtvHandle = renderer->GetRasterHdrOutputTex().rtvHandle;
        cmdList->OMSetRenderTargets(1, &hdrRtvHandle, FALSE, &dsvHandle);
    }
    else
    {
        // Non-TAA: render directly to the back buffer at output resolution.
        // Internal resolution == output resolution in this path (enforced at
        // initialization and on AA mode toggle), so the G-buffer depth covers
        // the full viewport and can be bound as DSV safely.
        D3D12_VIEWPORT outputViewport = CD3DX12_VIEWPORT(0.0f, 0.0f, static_cast<float>(outputWidth), static_cast<float>(outputHeight));
        D3D12_RECT outputScissor = CD3DX12_RECT(0, 0, outputWidth, outputHeight);
        cmdList->RSSetViewports(1, &outputViewport);
        cmdList->RSSetScissorRects(1, &outputScissor);

        renderer->TransitionBackBuffer(D3D12_RESOURCE_STATE_RENDER_TARGET);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = renderer->GetCurrentBackBufferRTV();
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    }

    // TAA path: use HDR PSO (R16G16B16A16_FLOAT, no tonemapping) to match RasterHdrOutputTex.
    // Non-TAA path: use LDR PSO (R8G8B8A8_UNORM, with tonemapping) to match back buffer.
    ID3D12PipelineState* pso = rasterTaaActive ? transparentHdrPSO : transparentPSO;

    if (pso)
    {
        cmdList->SetPipelineState(pso);
        model->Render(cmdList, renderer, frustum, AlphaMode::Blend);
    }
}
