#include "Model.h"
#include <iostream>
#include <cgltf.h>
#define CGLTF_IMPLEMENTATION
#include <wrl/client.h>

Model::Model()
{
}

Model::~Model()
{
    // Cleanup GLTF data
    if (m_GltfModel.data)
    {
        cgltf_free(m_GltfModel.data);
        m_GltfModel.data = nullptr;
    }
}

bool Model::LoadGLTFModel(ID3D12Device* device, const std::string& filepath)
{
    cgltf_options options = {};
    cgltf_result result = cgltf_parse_file(&options, filepath.c_str(), &m_GltfModel.data);

    if (result != cgltf_result_success)
    {
        std::cerr << "Failed to parse GLTF file: " << filepath << std::endl;
        return false;
    }

    // Load buffer data - required for cgltf_accessor_read functions to work
    result = cgltf_load_buffers(&options, m_GltfModel.data, filepath.c_str());
    if (result != cgltf_result_success)
    {
        std::cerr << "Failed to load GLTF buffers: " << filepath << std::endl;
        cgltf_free(m_GltfModel.data);
        m_GltfModel.data = nullptr;
        return false;
    }

    result = cgltf_validate(m_GltfModel.data);
    if (result != cgltf_result_success)
    {
        std::cerr << "GLTF validation failed: " << filepath << std::endl;
        cgltf_free(m_GltfModel.data);
        m_GltfModel.data = nullptr;
        return false;
    }

    // Process meshes
    for (size_t i = 0; i < m_GltfModel.data->meshes_count; ++i)
    {
        cgltf_mesh* mesh = &m_GltfModel.data->meshes[i];

        for (size_t j = 0; j < mesh->primitives_count; ++j)
        {
            cgltf_primitive* primitive = &mesh->primitives[j];

            GLTFMesh gltfMesh;
            gltfMesh.vertices.clear();
            gltfMesh.indices.clear();

            // Process material
            if (primitive->material)
            {
                cgltf_material* material = primitive->material;
                if (material->has_pbr_metallic_roughness)
                {
                    // Base color factor
                    memcpy(gltfMesh.material.baseColorFactor,
                           material->pbr_metallic_roughness.base_color_factor,
                           sizeof(float) * 4);

                    // Metallic and roughness factors
                    gltfMesh.material.metallicFactor = material->pbr_metallic_roughness.metallic_factor;
                    gltfMesh.material.roughnessFactor = material->pbr_metallic_roughness.roughness_factor;
                }
            }

            // Process attributes
            cgltf_accessor* positionAccessor = nullptr;
            cgltf_accessor* normalAccessor = nullptr;
            cgltf_accessor* texCoordAccessor = nullptr;

            for (size_t k = 0; k < primitive->attributes_count; ++k)
            {
                cgltf_attribute* attribute = &primitive->attributes[k];
                if (attribute->type == cgltf_attribute_type_position)
                {
                    positionAccessor = attribute->data;
                }
                else if (attribute->type == cgltf_attribute_type_normal)
                {
                    normalAccessor = attribute->data;
                }
                else if (attribute->type == cgltf_attribute_type_texcoord && attribute->index == 0)
                {
                    texCoordAccessor = attribute->data;
                }
            }

            if (!positionAccessor)
            {
                std::cerr << "GLTF mesh missing position data" << std::endl;
                continue;
            }

            // Read vertices
            size_t vertexCount = positionAccessor->count;
            gltfMesh.vertices.resize(vertexCount);

            // Read positions
            for (size_t k = 0; k < vertexCount; ++k)
            {
                if (!cgltf_accessor_read_float(positionAccessor, k, gltfMesh.vertices[k].position, 3))
                {
                    std::cerr << "Failed to read position data from GLTF buffer" << std::endl;
                    return false;
                }
            }

            // Read normals (if available)
            if (normalAccessor)
            {
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    cgltf_accessor_read_float(normalAccessor, k,
                        gltfMesh.vertices[k].normal, 3);
                }
            }
            else
            {
                // Generate default normals
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    gltfMesh.vertices[k].normal[0] = 0.0f;
                    gltfMesh.vertices[k].normal[1] = 1.0f;
                    gltfMesh.vertices[k].normal[2] = 0.0f;
                }
            }

            // Read texture coordinates (if available)
            if (texCoordAccessor)
            {
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    cgltf_accessor_read_float(texCoordAccessor, k,
                        gltfMesh.vertices[k].texCoord, 2);
                }
            }
            else
            {
                // Generate default texture coordinates
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    gltfMesh.vertices[k].texCoord[0] = 0.0f;
                    gltfMesh.vertices[k].texCoord[1] = 0.0f;
                }
            }

            // Read indices
            if (primitive->indices)
            {
                size_t indexCount = primitive->indices->count;
                gltfMesh.indices.resize(indexCount);

                for (size_t k = 0; k < indexCount; ++k)
                {
                    gltfMesh.indices[k] = cgltf_accessor_read_index(primitive->indices, k);
                }
            }

            m_GltfModel.meshes.push_back(gltfMesh);
        }
    }

    std::cout << "Successfully loaded GLTF model: " << filepath << " (" << m_GltfModel.meshes.size() << " meshes)" << std::endl;

    // Create DirectX 12 resources for the loaded model
    CreateGLTFResources(device);

    return true;
}

void Model::CreateGLTFResources(ID3D12Device* device)
{
    for (auto& mesh : m_GltfModel.meshes)
    {
        if (mesh.vertices.empty())
            continue;

        // Create vertex buffer
        {
            const UINT vertexBufferSize = static_cast<UINT>(mesh.vertices.size() * sizeof(GLTFVertex));

            D3D12_HEAP_PROPERTIES heapProps = {};
            heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
            heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

            D3D12_RESOURCE_DESC resourceDesc = {};
            resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            resourceDesc.Alignment = 0;
            resourceDesc.Width = vertexBufferSize;
            resourceDesc.Height = 1;
            resourceDesc.DepthOrArraySize = 1;
            resourceDesc.MipLevels = 1;
            resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
            resourceDesc.SampleDesc.Count = 1;
            resourceDesc.SampleDesc.Quality = 0;
            resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            HRESULT hr = device->CreateCommittedResource(
                &heapProps,
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&mesh.vertexBuffer)
            );
            if (FAILED(hr))
            {
                std::cerr << "CreateCommittedResource for vertex buffer failed" << std::endl;
                return;
            }

            // Copy vertex data
            UINT8* pVertexDataBegin;
            D3D12_RANGE readRange = { 0, 0 };
            hr = mesh.vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
            if (FAILED(hr))
            {
                std::cerr << "Map vertex buffer failed" << std::endl;
                return;
            }
            memcpy(pVertexDataBegin, mesh.vertices.data(), vertexBufferSize);
            mesh.vertexBuffer->Unmap(0, nullptr);

            // Initialize vertex buffer view
            mesh.vertexBufferView.BufferLocation = mesh.vertexBuffer->GetGPUVirtualAddress();
            mesh.vertexBufferView.StrideInBytes = sizeof(GLTFVertex);
            mesh.vertexBufferView.SizeInBytes = vertexBufferSize;
        }

        // Create index buffer (if indices exist)
        if (!mesh.indices.empty())
        {
            const UINT indexBufferSize = static_cast<UINT>(mesh.indices.size() * sizeof(uint32_t));

            D3D12_HEAP_PROPERTIES heapProps = {};
            heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
            heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

            D3D12_RESOURCE_DESC resourceDesc = {};
            resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            resourceDesc.Alignment = 0;
            resourceDesc.Width = indexBufferSize;
            resourceDesc.Height = 1;
            resourceDesc.DepthOrArraySize = 1;
            resourceDesc.MipLevels = 1;
            resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
            resourceDesc.SampleDesc.Count = 1;
            resourceDesc.SampleDesc.Quality = 0;
            resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

            HRESULT hr = device->CreateCommittedResource(
                &heapProps,
                D3D12_HEAP_FLAG_NONE,
                &resourceDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(&mesh.indexBuffer)
            );
            if (FAILED(hr))
            {
                std::cerr << "CreateCommittedResource for index buffer failed" << std::endl;
                return;
            }

            // Copy index data
            UINT8* pIndexDataBegin;
            D3D12_RANGE readRange = { 0, 0 };
            hr = mesh.indexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pIndexDataBegin));
            if (FAILED(hr))
            {
                std::cerr << "Map index buffer failed" << std::endl;
                return;
            }
            memcpy(pIndexDataBegin, mesh.indices.data(), indexBufferSize);
            mesh.indexBuffer->Unmap(0, nullptr);

            // Initialize index buffer view
            mesh.indexBufferView.BufferLocation = mesh.indexBuffer->GetGPUVirtualAddress();
            mesh.indexBufferView.Format = DXGI_FORMAT_R32_UINT;
            mesh.indexBufferView.SizeInBytes = indexBufferSize;
        }
    }
}

void Model::Render(ID3D12GraphicsCommandList* commandList)
{
    for (const auto& mesh : m_GltfModel.meshes)
    {
        if (mesh.vertices.empty())
            continue;

        // Set vertex buffer
        commandList->IASetVertexBuffers(0, 1, &mesh.vertexBufferView);

        // Set index buffer if available
        if (!mesh.indices.empty())
        {
            commandList->IASetIndexBuffer(&mesh.indexBufferView);
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandList->DrawIndexedInstanced(static_cast<UINT>(mesh.indices.size()), 1, 0, 0, 0);
        }
        else
        {
            // Non-indexed rendering
            commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            commandList->DrawInstanced(static_cast<UINT>(mesh.vertices.size()), 1, 0, 0);
        }
    }
}