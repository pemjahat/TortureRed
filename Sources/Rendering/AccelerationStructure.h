#pragma once

#include <unordered_map>
#include "Graphics/GraphicsTypes.h"

// Forward declarations to avoid circular dependencies
struct GLTFPrimitive;
class Model;

// -----------------------------------------------------------------------------
// AccelerationStructure
//
// Owns the BLAS pool (one BLAS per unique GLTFPrimitive) and the single TLAS
// built from the model's draw-node instances. Used by PathTracing/RestirDI/
// RestirGI for ray queries (TLAS SRV bound at root param 4).
// -----------------------------------------------------------------------------
class AccelerationStructure
{
public:
    #include <functional>

    // `executeAndWait` closes the command list, submits it, and blocks until
    // the GPU has finished — must be called while cmdList is still open and
    // scratch buffers are still valid. Caller supplies this since only it
    // owns the command queue/fence (e.g. Renderer::ExecuteCommandList()).
    size_t Build(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, Model* model,
                 const std::function<void()>& executeAndWait);

    D3D12_GPU_VIRTUAL_ADDRESS GetTLASGPUAddress() const { return m_TLAS.gpuAddress; }

    // Releases the BLAS pool (call on shutdown, before GPU device teardown).
    void Reset() { m_BlasPool.clear(); }

private:
    std::unordered_map<const GLTFPrimitive*, GPUBuffer> m_BlasPool;
    GPUBuffer m_TLAS;
};
