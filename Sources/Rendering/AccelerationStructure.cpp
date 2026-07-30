#include "pch.h"

#include "AccelerationStructure.h"
#include "Core/Model.h"
#include "Core/Utility.h"
#include "Renderer.h"

size_t AccelerationStructure::Build(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, Model* model, Renderer* renderer)
{
    if (!model)
        return 0;

    // Keep temporary buffers alive until the caller submits/waits on the command list.
    GPUBuffer scratchBuffer;
    GPUBuffer tlasScratch;
    GPUBuffer instanceDescBuffer;

    Microsoft::WRL::ComPtr<ID3D12Device5> device5;
    CHECK_HR(device->QueryInterface(IID_PPV_ARGS(&device5)), "Failed to get ID3D12Device5");

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> cmdList4;
    CHECK_HR(cmdList->QueryInterface(IID_PPV_ARGS(&cmdList4)), "Failed to get ID3D12GraphicsCommandList4");

    // 1. Identify all unique primitives and build BLAS for each
    std::vector<const GLTFPrimitive*> modelPrims;
    model->GetAllPrimitives(modelPrims);

    struct BLASBuildInfo {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs;
        D3D12_RAYTRACING_GEOMETRY_DESC geom;
    };
    std::vector<BLASBuildInfo> buildInfos;
    std::vector<GLTFPrimitive*> primsToBuild;
    UINT64 maxScratchSize = 0;

    for (const auto* cp : modelPrims)
    {
        GLTFPrimitive* prim = const_cast<GLTFPrimitive*>(cp);
        BLASBuildInfo info = {};
        info.geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        info.geom.Triangles.VertexBuffer.StartAddress = model->GetGlobalVertexBufferAddress() + (prim->globalVertexOffset * sizeof(GLTFVertex));
        info.geom.Triangles.VertexBuffer.StrideInBytes = sizeof(GLTFVertex);
        info.geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        info.geom.Triangles.VertexCount = static_cast<UINT>(prim->vertices.size());
        info.geom.Triangles.IndexBuffer = model->GetGlobalIndexBufferAddress() + (prim->globalIndexOffset * sizeof(uint32_t));
        info.geom.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        info.geom.Triangles.IndexCount = static_cast<UINT>(prim->indices.size());

        if (prim->alphaMode == AlphaMode::Mask || prim->alphaMode == AlphaMode::Blend)
        {
            info.geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        }
        else
        {
            info.geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        }

        info.inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        info.inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        info.inputs.NumDescs = 1;
        info.inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        info.inputs.pGeometryDescs = &info.geom;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
        device5->GetRaytracingAccelerationStructurePrebuildInfo(&info.inputs, &prebuildInfo);

        maxScratchSize = (maxScratchSize > prebuildInfo.ScratchDataSizeInBytes) ? maxScratchSize : prebuildInfo.ScratchDataSizeInBytes;

        GPUBuffer blasBuffer;
        if (CreateBuffer(blasBuffer, prebuildInfo.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false, false, "AS_BLAS"))
        {
            m_BlasPool[prim] = std::move(blasBuffer);
            buildInfos.push_back(info);
            primsToBuild.push_back(prim);
        }
    }

    if (!primsToBuild.empty())
    {
        CreateBuffer(scratchBuffer, maxScratchSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, false, "AS_BLASScratch");

        for (size_t i = 0; i < primsToBuild.size(); ++i)
        {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
            buildDesc.Inputs = buildInfos[i].inputs;
            buildDesc.Inputs.pGeometryDescs = &buildInfos[i].geom; // Use pointer to internal geom
            buildDesc.ScratchAccelerationStructureData = scratchBuffer.gpuAddress;
            buildDesc.DestAccelerationStructureData = m_BlasPool[primsToBuild[i]].gpuAddress;

            cmdList4->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            D3D12_RESOURCE_BARRIER uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(scratchBuffer.resource.Get());
            cmdList->ResourceBarrier(1, &uavBarrier);
        }
    }

    // 2. Build TLAS for all draw node instances
    const auto& nodeData = model->GetDrawNodeData();
    std::vector<const GLTFPrimitive*> nodePrims;
    model->GetDrawNodePrimitives(nodePrims);

    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDescs;
    for (size_t i = 0; i < nodeData.size(); ++i)
    {
        D3D12_RAYTRACING_INSTANCE_DESC inst = {};
        const auto& world = nodeData[i].world;
        inst.Transform[0][0] = world._11; inst.Transform[0][1] = world._21; inst.Transform[0][2] = world._31; inst.Transform[0][3] = world._41;
        inst.Transform[1][0] = world._12; inst.Transform[1][1] = world._22; inst.Transform[1][2] = world._32; inst.Transform[1][3] = world._42;
        inst.Transform[2][0] = world._13; inst.Transform[2][1] = world._23; inst.Transform[2][2] = world._33; inst.Transform[2][3] = world._43;

        inst.InstanceID = static_cast<UINT>(i);
        inst.InstanceMask = 0xFF;
        inst.InstanceContributionToHitGroupIndex = 0;
        inst.Flags = D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
        inst.AccelerationStructure = m_BlasPool[nodePrims[i]].gpuAddress;
        instanceDescs.push_back(inst);
    }

    if (!instanceDescs.empty())
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
        tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        tlasInputs.NumDescs = static_cast<UINT>(instanceDescs.size());
        tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuildInfo = {};
        device5->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuildInfo);

        CreateBuffer(m_TLAS, tlasPrebuildInfo.ResultDataMaxSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, false, false, "AS_TLAS");

        CreateBuffer(tlasScratch, tlasPrebuildInfo.ScratchDataSizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, false, false, "AS_TLASScratch");

        CreateBuffer(instanceDescBuffer, instanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false, false, "AS_InstanceDescBuffer");
        memcpy(instanceDescBuffer.cpuPtr, instanceDescs.data(), instanceDescs.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
        tlasBuildDesc.Inputs = tlasInputs;
        tlasBuildDesc.Inputs.InstanceDescs = instanceDescBuffer.gpuAddress;
        tlasBuildDesc.ScratchAccelerationStructureData = tlasScratch.gpuAddress;
        tlasBuildDesc.DestAccelerationStructureData = m_TLAS.gpuAddress;

        cmdList4->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
        D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::UAV(m_TLAS.resource.Get());
        cmdList->ResourceBarrier(1, &barrier);
    }

     // Submit and block until GPU finishes, while scratchBuffer/tlasScratch/
    // instanceDescBuffer (stack locals) are still in scope — this makes it
    // safe for them to be destroyed right after this call returns.
    renderer->ExecuteCommandList();

    return instanceDescs.size();
}
