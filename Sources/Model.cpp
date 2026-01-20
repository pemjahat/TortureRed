#define NOMINMAX
#include "Model.h"
#include "Utility.h"
#include "Renderer.h"
#include <iostream>
#include <cgltf.h>
#define CGLTF_IMPLEMENTATION
#include <wrl/client.h>
#include <DirectXTex.h>
#include <cstdint>
#include <algorithm>

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

    // Set file directory
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos)
    {
        std::string dir = filepath.substr(0, lastSlash + 1);
        fileDirectory = std::wstring(dir.begin(), dir.end());
    }
    else
    {
        fileDirectory = L"";
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

    LoadTextures(device);

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

                    // Base color texture
                    if (material->pbr_metallic_roughness.base_color_texture.texture)
                    {
                        size_t texIndex = material->pbr_metallic_roughness.base_color_texture.texture - m_GltfModel.data->textures;
                        if (texIndex < m_GltfModel.textures.size())
                            gltfMesh.material.baseColorTexture = &m_GltfModel.textures[texIndex];
                    }
                }

                // Normal texture
                if (material->normal_texture.texture)
                {
                    size_t texIndex = material->normal_texture.texture - m_GltfModel.data->textures;
                    if (texIndex < m_GltfModel.textures.size())
                        gltfMesh.material.normalTexture = &m_GltfModel.textures[texIndex];
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

    BuildNodeHierarchy();
    LoadAnimations();
    if (!m_GltfModel.animations.empty())
        m_CurrentAnimation = &m_GltfModel.animations[0];

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

void Model::LoadTextures(ID3D12Device* device)
{
    m_GltfModel.textures.resize(m_GltfModel.data->textures_count);
    for (size_t i = 0; i < m_GltfModel.data->textures_count; ++i)
    {
        cgltf_texture* tex = &m_GltfModel.data->textures[i];
        cgltf_image* img = tex->image;
        GLTFTexture& gltfTex = m_GltfModel.textures[i];

        DirectX::ScratchImage image;
        if (img->uri)
        {
            // External image
            std::string uri = img->uri;
            std::string dirStr(this->fileDirectory.begin(), this->fileDirectory.end());
            std::string fullPath = dirStr + uri;
            std::wstring wuri(fullPath.begin(), fullPath.end());
            CHECK_HR(DirectX::LoadFromWICFile(wuri.c_str(), DirectX::WIC_FLAGS_NONE, nullptr, image), "Load external image failed");
            gltfTex.image = new DirectX::ScratchImage(std::move(image));
        }
        else if (img->buffer_view)
        {
            // Embedded image, assume PNG or use WIC
            cgltf_buffer_view* bv = img->buffer_view;
            unsigned char* data = (unsigned char*)bv->buffer->data + bv->offset;
            size_t size = bv->size;
            CHECK_HR(DirectX::LoadFromWICMemory(data, size, DirectX::WIC_FLAGS_NONE, nullptr, image), "Load embedded image failed");
            gltfTex.image = new DirectX::ScratchImage(std::move(image));
        }
        else
        {
            // Skip
            continue;
        }

        const DirectX::TexMetadata& metaData = image.GetMetadata();
        DXGI_FORMAT format = metaData.format;

        D3D12_RESOURCE_DESC textureDesc = {};
        textureDesc.MipLevels = UINT(metaData.mipLevels);
        textureDesc.Format = format;
        textureDesc.Width = UINT(metaData.width);
        textureDesc.Height = UINT(metaData.height);
        textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
        textureDesc.DepthOrArraySize = metaData.dimension == DirectX::TEX_DIMENSION_TEXTURE3D ? UINT(metaData.depth) : UINT(metaData.arraySize);
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Dimension = metaData.dimension == DirectX::TEX_DIMENSION_TEXTURE3D ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        textureDesc.Alignment = 0;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;
        
        CHECK_HR(device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &textureDesc,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&gltfTex.resource)), "CreateCommittedResource failed");

        // Upload data
        // For simplicity, assume no upload yet, or add upload code
    }
}

void Model::BuildNodeHierarchy()
{
    // Build node hierarchy
    m_GltfModel.nodes.resize(m_GltfModel.data->nodes_count);
    for (size_t i = 0; i < m_GltfModel.data->nodes_count; ++i)
    {
        cgltf_node* node = &m_GltfModel.data->nodes[i];
        GLTFNode& gltfNode = m_GltfModel.nodes[i];
        if (node->name)
            gltfNode.name = node->name;
        if (node->mesh)
        {
            // Assume meshes are in order, find index
            size_t meshIndex = node->mesh - m_GltfModel.data->meshes;
            if (meshIndex < m_GltfModel.meshes.size())
                gltfNode.mesh = &m_GltfModel.meshes[meshIndex];
        }
        // Set transform
        if (node->has_matrix)
        {
            memcpy(&gltfNode.transform, node->matrix, sizeof(float) * 16);
            // Decompose to TRS for animation
            DirectX::XMMATRIX mat = DirectX::XMLoadFloat4x4(&gltfNode.transform);
            DirectX::XMVECTOR scale, rotQuat, trans;
            DirectX::XMMatrixDecompose(&scale, &rotQuat, &trans, mat);
            DirectX::XMStoreFloat3(&gltfNode.scale, scale);
            DirectX::XMStoreFloat4(&gltfNode.rotation, rotQuat);
            DirectX::XMStoreFloat3(&gltfNode.translation, trans);
        }
        else
        {
            gltfNode.translation = DirectX::XMFLOAT3(node->translation[0], node->translation[1], node->translation[2]);
            gltfNode.rotation = DirectX::XMFLOAT4(node->rotation[0], node->rotation[1], node->rotation[2], node->rotation[3]);
            gltfNode.scale = DirectX::XMFLOAT3(node->scale[0], node->scale[1], node->scale[2]);
            // Compute matrix
            DirectX::XMMATRIX t = DirectX::XMMatrixTranslation(gltfNode.translation.x, gltfNode.translation.y, gltfNode.translation.z);
            DirectX::XMMATRIX r = DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(gltfNode.rotation.x, gltfNode.rotation.y, gltfNode.rotation.z, gltfNode.rotation.w));
            DirectX::XMMATRIX s = DirectX::XMMatrixScaling(gltfNode.scale.x, gltfNode.scale.y, gltfNode.scale.z);
            DirectX::XMMATRIX m = s * r * t;
            DirectX::XMStoreFloat4x4(&gltfNode.transform, m);
        }
        // Children
        gltfNode.children.resize(node->children_count);
        for (size_t j = 0; j < node->children_count; ++j)
        {
            size_t childIndex = node->children[j] - m_GltfModel.data->nodes;
            gltfNode.children[j] = &m_GltfModel.nodes[childIndex];
            m_GltfModel.nodes[childIndex].parent = &gltfNode;
        }
    }

    // Set root nodes
    cgltf_scene* scene = m_GltfModel.data->scene ? m_GltfModel.data->scene : (m_GltfModel.data->scenes_count > 0 ? &m_GltfModel.data->scenes[0] : nullptr);
    if (scene)
    {
        m_GltfModel.rootNodes.resize(scene->nodes_count);
        for (size_t i = 0; i < scene->nodes_count; ++i)
        {
            size_t nodeIndex = scene->nodes[i] - m_GltfModel.data->nodes;
            m_GltfModel.rootNodes[i] = &m_GltfModel.nodes[nodeIndex];
        }
    }
}

void Model::LoadAnimations()
{
    // Load animations
    m_GltfModel.animations.resize(m_GltfModel.data->animations_count);
    for (size_t i = 0; i < m_GltfModel.data->animations_count; ++i)
    {
        cgltf_animation* anim = &m_GltfModel.data->animations[i];
        GLTFAnimation& gltfAnim = m_GltfModel.animations[i];
        if (anim->name)
            gltfAnim.name = anim->name;
        gltfAnim.channels.resize(anim->channels_count);
        for (size_t j = 0; j < anim->channels_count; ++j)
        {
            cgltf_animation_channel* channel = &anim->channels[j];
            GLTFAnimationChannel& gltfChannel = gltfAnim.channels[j];
            // Target node
            size_t nodeIndex = channel->target_node - m_GltfModel.data->nodes;
            gltfChannel.targetNode = &m_GltfModel.nodes[nodeIndex];
            // Type
            if (channel->target_path == cgltf_animation_path_type_translation)
                gltfChannel.type = GLTFAnimationChannel::Translation;
            else if (channel->target_path == cgltf_animation_path_type_rotation)
                gltfChannel.type = GLTFAnimationChannel::Rotation;
            else if (channel->target_path == cgltf_animation_path_type_scale)
                gltfChannel.type = GLTFAnimationChannel::Scale;
            // Times
            cgltf_accessor* timeAccessor = channel->sampler->input;
            gltfChannel.times.resize(timeAccessor->count);
            if (timeAccessor->component_type == cgltf_component_type_r_32f) {
                for (size_t i = 0; i < timeAccessor->count; ++i) {
                    if (!cgltf_accessor_read_float(timeAccessor, i, &gltfChannel.times[i], 1)) {
                        std::cerr << "Failed to read animation time at index " << i << std::endl;
                        break;
                    }
                }
            } else {
                std::cerr << "Unsupported time accessor component type: " << timeAccessor->component_type << std::endl;
            }
            // Values
            cgltf_accessor* valueAccessor = channel->sampler->output;
            if (gltfChannel.type == GLTFAnimationChannel::Translation)
            {
                gltfChannel.translations.resize(valueAccessor->count);
                for (size_t k = 0; k < valueAccessor->count; ++k)
                {
                    float v[3];
                    cgltf_accessor_read_float(valueAccessor, k, v, 3);
                    gltfChannel.translations[k] = DirectX::XMFLOAT3(v[0], v[1], v[2]);
                }
            }
            else if (gltfChannel.type == GLTFAnimationChannel::Rotation)
            {
                gltfChannel.rotations.resize(valueAccessor->count);
                for (size_t k = 0; k < valueAccessor->count; ++k)
                {
                    float v[4];
                    cgltf_accessor_read_float(valueAccessor, k, v, 4);
                    gltfChannel.rotations[k] = DirectX::XMFLOAT4(v[0], v[1], v[2], v[3]);
                }
            }
            else if (gltfChannel.type == GLTFAnimationChannel::Scale)
            {
                gltfChannel.scales.resize(valueAccessor->count);
                for (size_t k = 0; k < valueAccessor->count; ++k)
                {
                    float v[3];
                    cgltf_accessor_read_float(valueAccessor, k, v, 3);
                    gltfChannel.scales[k] = DirectX::XMFLOAT3(v[0], v[1], v[2]);
                }
            }
        }
    }
}

void Model::UpdateAnimation(float deltaTime)
{
    if (!m_CurrentAnimation)
        return;

    m_AnimationTime += deltaTime;

    // For simplicity, loop the animation
    float animDuration = 0.0f;
    for (auto& channel : m_CurrentAnimation->channels)
    {
        if (!channel.times.empty())
            animDuration = std::max(animDuration, channel.times.back());
    }
    if (animDuration > 0.0f)
        m_AnimationTime = fmod(m_AnimationTime, animDuration);

    // Update each channel
    for (auto& channel : m_CurrentAnimation->channels)
    {
        if (channel.times.empty())
            continue;

        // Find the two keyframes
        size_t key0 = 0, key1 = 0;
        for (size_t i = 0; i < channel.times.size() - 1; ++i)
        {
            if (m_AnimationTime >= channel.times[i] && m_AnimationTime <= channel.times[i + 1])
            {
                key0 = i;
                key1 = i + 1;
                break;
            }
        }

        float t0 = channel.times[key0];
        float t1 = channel.times[key1];
        float factor = (m_AnimationTime - t0) / (t1 - t0);
        factor = std::clamp(factor, 0.0f, 1.0f);

        if (channel.type == GLTFAnimationChannel::Translation)
        {
            DirectX::XMFLOAT3 v0 = channel.translations[key0];
            DirectX::XMFLOAT3 v1 = channel.translations[key1];
            channel.targetNode->translation.x = v0.x + (v1.x - v0.x) * factor;
            channel.targetNode->translation.y = v0.y + (v1.y - v0.y) * factor;
            channel.targetNode->translation.z = v0.z + (v1.z - v0.z) * factor;
        }
        else if (channel.type == GLTFAnimationChannel::Rotation)
        {
            DirectX::XMFLOAT4 q0 = channel.rotations[key0];
            DirectX::XMFLOAT4 q1 = channel.rotations[key1];
            // Simple linear interpolation for testing
            channel.targetNode->rotation.x = q0.x + (q1.x - q0.x) * factor;
            channel.targetNode->rotation.y = q0.y + (q1.y - q0.y) * factor;
            channel.targetNode->rotation.z = q0.z + (q1.z - q0.z) * factor;
            channel.targetNode->rotation.w = q0.w + (q1.w - q0.w) * factor;
        }
        else if (channel.type == GLTFAnimationChannel::Scale)
        {
            DirectX::XMFLOAT3 v0 = channel.scales[key0];
            DirectX::XMFLOAT3 v1 = channel.scales[key1];
            channel.targetNode->scale.x = v0.x + (v1.x - v0.x) * factor;
            channel.targetNode->scale.y = v0.y + (v1.y - v0.y) * factor;
            channel.targetNode->scale.z = v0.z + (v1.z - v0.z) * factor;
        }

        // Recompute matrix
        DirectX::XMMATRIX t = DirectX::XMMatrixTranslation(channel.targetNode->translation.x, channel.targetNode->translation.y, channel.targetNode->translation.z);
        DirectX::XMMATRIX r = DirectX::XMMatrixRotationQuaternion(DirectX::XMVectorSet(channel.targetNode->rotation.x, channel.targetNode->rotation.y, channel.targetNode->rotation.z, channel.targetNode->rotation.w));
        DirectX::XMMATRIX s = DirectX::XMMatrixScaling(channel.targetNode->scale.x, channel.targetNode->scale.y, channel.targetNode->scale.z);
        DirectX::XMMATRIX m = s * r * t;
        DirectX::XMStoreFloat4x4(&channel.targetNode->transform, m);
    }
}

void Model::UploadTextures(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* cmdQueue, ID3D12CommandAllocator* cmdAllocator, ID3D12DescriptorHeap* srvHeap)
{
    // Reset the command list
    CHECK_HR(cmdList->Reset(cmdAllocator, nullptr), "Reset command list failed");

    srvHeapStart = srvHeap->GetGPUDescriptorHandleForHeapStart();
    srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> uploadBuffers;

    for (size_t i = 0; i < m_GltfModel.textures.size(); ++i)
    {
        auto& gltfTex = m_GltfModel.textures[i];
        if (!gltfTex.image)
            continue;

        gltfTex.srvIndex = UINT(i);  // Set index

        const DirectX::TexMetadata& metaData = gltfTex.image->GetMetadata();

        // Transition texture to COPY_DEST
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = gltfTex.resource.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        cmdList->ResourceBarrier(1, &barrier);

        const UINT numSubResources = UINT(metaData.mipLevels * metaData.arraySize);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)_alloca(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) * numSubResources);
        UINT* numRows = (UINT*)_alloca(sizeof(UINT) * numSubResources);
        UINT64* rowSizes = (UINT64*)_alloca(sizeof(UINT64) * numSubResources);

        UINT64 textureMemSize = 0;
        D3D12_RESOURCE_DESC textureDesc = gltfTex.resource->GetDesc();
        device->GetCopyableFootprints(&textureDesc, 0, numSubResources, 0, layouts, numRows, rowSizes, &textureMemSize);

        // Create upload buffer
        D3D12_HEAP_PROPERTIES uploadHeapProps = {};
        uploadHeapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC uploadDesc = {};
        uploadDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        uploadDesc.Width = textureMemSize;
        uploadDesc.Height = 1;
        uploadDesc.DepthOrArraySize = 1;
        uploadDesc.MipLevels = 1;
        uploadDesc.Format = DXGI_FORMAT_UNKNOWN;
        uploadDesc.SampleDesc.Count = 1;
        uploadDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        Microsoft::WRL::ComPtr<ID3D12Resource> uploadBuffer;
        CHECK_HR(device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &uploadDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuffer)), "Create upload buffer failed");

        uploadBuffers.push_back(uploadBuffer);

        // Copy data to upload buffer
        uint8_t* uploadMem;
        uploadBuffer->Map(0, nullptr, reinterpret_cast<void**>(&uploadMem));

        for (UINT arrayIdx = 0; arrayIdx < UINT(metaData.arraySize); ++arrayIdx)
        {
            for (UINT mipIdx = 0; mipIdx < UINT(metaData.mipLevels); ++mipIdx)
            {
                const UINT subResourceIdx = mipIdx + (arrayIdx * UINT(metaData.mipLevels));
                const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& subResourceLayout = layouts[subResourceIdx];
                uint8_t* dstSubResourceMem = uploadMem + subResourceLayout.Offset;
                const DirectX::Image* subImage = gltfTex.image->GetImage(mipIdx, arrayIdx, 0);
                for (UINT z = 0; z < subResourceLayout.Footprint.Depth; ++z)
                {
                    uint8_t* dst = dstSubResourceMem;
                    const uint8_t* src = subImage->pixels;
                    for (UINT y = 0; y < numRows[subResourceIdx]; ++y)
                    {
                        memcpy(dst, src, rowSizes[subResourceIdx]);
                        dst += subResourceLayout.Footprint.RowPitch;
                        src += subImage->rowPitch;
                    }
                }
            }
        }
        uploadBuffer->Unmap(0, nullptr);

        // Copy to texture
        for (UINT subResourceIdx = 0; subResourceIdx < numSubResources; ++subResourceIdx)
        {
            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource = gltfTex.resource.Get();
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = subResourceIdx;
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = uploadBuffer.Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = layouts[subResourceIdx];
            cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }

        // Transition to PIXEL_SHADER_RESOURCE
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        cmdList->ResourceBarrier(1, &barrier);

        // Create SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = metaData.format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = UINT(metaData.mipLevels);
        srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

        UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = srvHeap->GetCPUDescriptorHandleForHeapStart();
        srvHandle.ptr += gltfTex.srvIndex * descriptorSize;

        device->CreateShaderResourceView(gltfTex.resource.Get(), &srvDesc, srvHandle);

        gltfTex.srvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(srvHeapStart).Offset(gltfTex.srvIndex, srvDescriptorSize);

        // Clean up
        delete gltfTex.image;
        gltfTex.image = nullptr;
    }

    // Execute the upload commands
    CHECK_HR(cmdList->Close(), "Close command list failed");
    ID3D12CommandList* commandLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, commandLists);

    // Wait for completion
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence failed");
    HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    CHECK_HR(fence->SetEventOnCompletion(1, eventHandle), "Set event on completion failed");
    cmdQueue->Signal(fence.Get(), 1);
    WaitForSingleObject(eventHandle, INFINITE);
    CloseHandle(eventHandle);

    // The command list remains closed; BeginFrame will reset it
}

void Model::Render(ID3D12GraphicsCommandList* commandList, Renderer* renderer)
{
    // Bind Material CB
    commandList->SetGraphicsRootConstantBufferView(1, renderer->GetMaterialCB()->GetGPUVirtualAddress());

    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        RenderNode(commandList, rootNode, DirectX::XMMatrixIdentity(), renderer);
    }
}

void Model::RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer)
{
    DirectX::XMMATRIX world = DirectX::XMLoadFloat4x4(&node->transform) * parentTransform;

    if (node->mesh)
    {
        // Update Material CB
        MaterialConstants matCB;
        matCB.baseColorFactor.x = node->mesh->material.baseColorFactor[0];
        matCB.baseColorFactor.y = node->mesh->material.baseColorFactor[1];
        matCB.baseColorFactor.z = node->mesh->material.baseColorFactor[2];
        matCB.baseColorFactor.w = node->mesh->material.baseColorFactor[3];
        matCB.metallicFactor = node->mesh->material.metallicFactor;
        matCB.roughnessFactor = node->mesh->material.roughnessFactor;
        matCB.hasBaseColorTexture = node->mesh->material.baseColorTexture ? 1 : 0;
        renderer->UpdateMaterialCB(matCB);

        // Set world matrix as root constants
        float worldFloats[16];
        DirectX::XMStoreFloat4x4(reinterpret_cast<DirectX::XMFLOAT4X4*>(worldFloats), world);
        commandList->SetGraphicsRoot32BitConstants(2, 16, worldFloats, 0);

        commandList->SetGraphicsRootConstantBufferView(1, renderer->GetMaterialCB()->GetGPUVirtualAddress());

        // Set texture descriptor table if has texture
        if (node->mesh->material.baseColorTexture)
        {
            commandList->SetGraphicsRootDescriptorTable(3, node->mesh->material.baseColorTexture->srvHandle);
        }

        // Render the mesh
        commandList->IASetVertexBuffers(0, 1, &node->mesh->vertexBufferView);
        commandList->IASetIndexBuffer(&node->mesh->indexBufferView);
        commandList->DrawIndexedInstanced(static_cast<UINT>(node->mesh->indices.size()), 1, 0, 0, 0);
    }

    for (auto* child : node->children)
    {
        RenderNode(commandList, child, world, renderer);
    }
}