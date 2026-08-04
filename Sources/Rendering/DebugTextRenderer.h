#pragma once

// DebugTextRenderer — GPU on-screen debug text/lines
//
// Owns the shared render-data buffer (counters + character/line instances), the
// font atlas + glyph metrics (built from an ImFontAtlas), the indirect draw args
// buffer, and the three PSOs (args builder, glyph raster, line raster).
//
// Producer shaders append instances via DebugTextRender.hlsli using the bindless
// indices exposed here; once per frame Render() builds the draw args (resetting
// the counters) and ExecuteIndirect-rasterizes everything onto the backbuffer.

#include <wrl.h>
#include <d3d12.h>
#include <stdint.h>
#include "Graphics/GraphicsTypes.h"

class DebugTextRenderer
{
public:
    bool Initialize(ID3D12Device* device, ID3D12CommandQueue* cmdQueue, ID3D12RootSignature* mainRootSignature);

    // Builds indirect draw args from the producers' counters, then rasterizes all
    // recorded text/lines onto `rtv` (backbuffer, output resolution). Leaves the
    // render-data buffer in UAV state ready for next frame's producers.
    void Render(ID3D12GraphicsCommandList* cmdList, ID3D12RootSignature* mainRootSignature,
                D3D12_GPU_VIRTUAL_ADDRESS frameCBAddress,
                D3D12_CPU_DESCRIPTOR_HANDLE rtv, uint32_t width, uint32_t height);

    uint32_t GetRenderDataUAVIndex() const { return static_cast<uint32_t>(m_RenderData.uavIndex); }
    uint32_t GetGlyphSRVIndex()      const { return static_cast<uint32_t>(m_GlyphData.srvIndex); }
    float    GetFontSize()           const { return m_FontSize; }

private:
    bool CreateFontAtlas(ID3D12Device* device, ID3D12CommandQueue* cmdQueue);
    bool CreatePipelines(ID3D12Device* device, ID3D12RootSignature* mainRootSignature);

    // Must match the SharedTypes.h layout defines exactly
    struct Data
    {
        uint32_t TextCount;
        uint32_t LineCount;
        uint32_t _pad[2];
        DebugCharInstance TextInstances[DEBUG_TEXT_MAX_CHARS];
        DebugLineInstance LineInstances[DEBUG_TEXT_MAX_LINES];
    };

    GPUBuffer m_RenderData;      // ByteAddress (RAW): counters + instances
    GPUBuffer m_IndirectArgs;    // 2 × uint4 D3D12_DRAW_ARGUMENTS
    GPUBuffer m_GlyphData;       // StructuredBuffer<DebugGlyph>[128]
    GPUTexture m_FontAtlas;      // RGBA8 font atlas
    float     m_FontSize = 13.0f;

    Microsoft::WRL::ComPtr<ID3D12CommandSignature> m_DrawCommandSignature;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>    m_BuildArgsPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>    m_GlyphPSO;
    Microsoft::WRL::ComPtr<ID3D12PipelineState>    m_LinePSO;
};
