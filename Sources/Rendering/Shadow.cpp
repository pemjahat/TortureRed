#include "pch.h"

#include "Shadow.h"
#include "Core/Utility.h"
#include "Graphics/GraphicsHelper.h"

bool Shadow::CreateResources()
{
    if (!CreateTexture(m_ShadowMap, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, DXGI_FORMAT_R32_TYPELESS,
                        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                        nullptr, 1, 1, "Tex_ShadowMap"))
    {
        std::cerr << "Failed to create shadow map texture" << std::endl;
        return false;
    }

    return true;
}
