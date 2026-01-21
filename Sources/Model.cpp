#define NOMINMAX
#include "Model.h"
#include "Renderer.h"
#include "Utility.h"
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

bool Model::LoadGLTFModel(Renderer* renderer, const std::string& filepath)
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

    LoadTextures(renderer);

    m_GltfModel.meshes.reserve(m_GltfModel.data->meshes_count);

    // Process meshes
    for (size_t i = 0; i < m_GltfModel.data->meshes_count; ++i)
    {
        cgltf_mesh* mesh = &m_GltfModel.data->meshes[i];
        GLTFMesh gltfMesh;
        if (mesh->name) gltfMesh.name = mesh->name;

        for (size_t j = 0; j < mesh->primitives_count; ++j)
        {
            cgltf_primitive* primitive = &mesh->primitives[j];

            GLTFPrimitive gltfPrim;

            // Process material
            if (primitive->material)
            {
                cgltf_material* material = primitive->material;

                // Set alpha mode
                if (material->alpha_mode == cgltf_alpha_mode_opaque)
                     gltfPrim.alphaMode = AlphaMode::Opaque;
                else if (material->alpha_mode == cgltf_alpha_mode_mask)
                    gltfPrim.alphaMode = AlphaMode::Mask;
                else if (material->alpha_mode == cgltf_alpha_mode_blend)
                    gltfPrim.alphaMode = AlphaMode::Blend;

                if (material->has_pbr_metallic_roughness)
                {
                    // Base color factor
                    memcpy(gltfPrim.material.baseColorFactor,
                           material->pbr_metallic_roughness.base_color_factor,
                           sizeof(float) * 4);

                    // Metallic and roughness factors
                    gltfPrim.material.metallicFactor = material->pbr_metallic_roughness.metallic_factor;
                    gltfPrim.material.roughnessFactor = material->pbr_metallic_roughness.roughness_factor;

                    // Base color texture
                    if (material->pbr_metallic_roughness.base_color_texture.texture)
                    {
                        size_t texIndex = material->pbr_metallic_roughness.base_color_texture.texture - m_GltfModel.data->textures;
                        if (texIndex < m_GltfModel.textures.size())
                            gltfPrim.material.baseColorTexture = &m_GltfModel.textures[texIndex];
                    }
                }

                // Normal texture
                if (material->normal_texture.texture)
                {
                    size_t texIndex = material->normal_texture.texture - m_GltfModel.data->textures;
                    if (texIndex < m_GltfModel.textures.size())
                        gltfPrim.material.normalTexture = &m_GltfModel.textures[texIndex];
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
            gltfPrim.vertices.resize(vertexCount);

            // Read positions
            for (size_t k = 0; k < vertexCount; ++k)
            {
                if (!cgltf_accessor_read_float(positionAccessor, k, gltfPrim.vertices[k].position, 3))
                {
                    std::cerr << "Failed to read position data from GLTF buffer" << std::endl;
                    return false;
                }
            }

            // Compute AABB
            DirectX::XMFLOAT3 minPos, maxPos;
            if (positionAccessor->has_min && positionAccessor->has_max)
            {
                minPos.x = positionAccessor->min[0];
                minPos.y = positionAccessor->min[1];
                minPos.z = positionAccessor->min[2];
                maxPos.x = positionAccessor->max[0];
                maxPos.y = positionAccessor->max[1];
                maxPos.z = positionAccessor->max[2];
            }
            else
            {
                minPos = {FLT_MAX, FLT_MAX, FLT_MAX};
                maxPos = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (const auto& v : gltfPrim.vertices)
                {
                    minPos.x = std::min(minPos.x, v.position[0]);
                    minPos.y = std::min(minPos.y, v.position[1]);
                    minPos.z = std::min(minPos.z, v.position[2]);
                    maxPos.x = std::max(maxPos.x, v.position[0]);
                    maxPos.y = std::max(maxPos.y, v.position[1]);
                    maxPos.z = std::max(maxPos.z, v.position[2]);
                }
            }
            DirectX::XMFLOAT3 center = {
                (minPos.x + maxPos.x) * 0.5f,
                (minPos.y + maxPos.y) * 0.5f,
                (minPos.z + maxPos.z) * 0.5f
            };
            DirectX::XMFLOAT3 extents = {
                (maxPos.x - minPos.x) * 0.5f,
                (maxPos.y - minPos.y) * 0.5f,
                (maxPos.z - minPos.z) * 0.5f
            };
            gltfPrim.aabb = DirectX::BoundingBox(center, extents);

            // Read normals (if available)
            if (normalAccessor)
            {
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    cgltf_accessor_read_float(normalAccessor, k,
                        gltfPrim.vertices[k].normal, 3);
                }
            }
            else
            {
                // Generate default normals
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    gltfPrim.vertices[k].normal[0] = 0.0f;
                    gltfPrim.vertices[k].normal[1] = 1.0f;
                    gltfPrim.vertices[k].normal[2] = 0.0f;
                }
            }

            // Read texture coordinates (if available)
            if (texCoordAccessor)
            {
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    cgltf_accessor_read_float(texCoordAccessor, k,
                        gltfPrim.vertices[k].texCoord, 2);
                }
            }
            else
            {
                // Generate default texture coordinates
                for (size_t k = 0; k < vertexCount; ++k)
                {
                    gltfPrim.vertices[k].texCoord[0] = 0.0f;
                    gltfPrim.vertices[k].texCoord[1] = 0.0f;
                }
            }

            // Read indices
            if (primitive->indices)
            {
                size_t indexCount = primitive->indices->count;
                gltfPrim.indices.resize(indexCount);

                for (size_t k = 0; k < indexCount; ++k)
                {
                    gltfPrim.indices[k] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive->indices, k));
                }
            }

            gltfMesh.primitives.push_back(std::move(gltfPrim));
        }

        m_GltfModel.meshes.push_back(std::move(gltfMesh));
    }


    std::cout << "Successfully loaded GLTF model: " << filepath << " (" << m_GltfModel.meshes.size() << " meshes)" << std::endl;

    BuildNodeHierarchy();
    LoadAnimations();
    if (!m_GltfModel.animations.empty())
        m_CurrentAnimation = &m_GltfModel.animations[0];

    // Create DirectX 12 resources for the loaded model
    CreateGLTFResources(renderer);

    return true;
}

void Model::CreateGLTFResources(Renderer* renderer)
{
    for (auto& mesh : m_GltfModel.meshes)
    {
        for (auto& prim : mesh.primitives)
        {
            if (prim.vertices.empty())
                continue;

            // Create vertex buffer in default heap
            {
                const UINT vertexBufferSize = static_cast<UINT>(prim.vertices.size() * sizeof(GLTFVertex));

                if (!renderer->CreateBuffer(prim.vertexBuffer, vertexBufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST))
                {
                    std::cerr << "Failed to create vertex buffer" << std::endl;
                    return;
                }

                // Create staging buffer
                if (!renderer->CreateBuffer(prim.vertexStaging, vertexBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
                {
                    std::cerr << "Failed to create vertex staging buffer" << std::endl;
                    return;
                }

                // Copy vertex data to staging
                memcpy(prim.vertexStaging.cpuPtr, prim.vertices.data(), vertexBufferSize);

                // Initialize vertex buffer view
                prim.vertexBufferView.BufferLocation = prim.vertexBuffer.gpuAddress;
                prim.vertexBufferView.StrideInBytes = sizeof(GLTFVertex);
                prim.vertexBufferView.SizeInBytes = vertexBufferSize;
            }

            // Create index buffer (if indices exist)
            if (!prim.indices.empty())
            {
                const UINT indexBufferSize = static_cast<UINT>(prim.indices.size() * sizeof(uint32_t));

                if (!renderer->CreateBuffer(prim.indexBuffer, indexBufferSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST))
                {
                    std::cerr << "Failed to create index buffer" << std::endl;
                    return;
                }

                // Create staging buffer
                if (!renderer->CreateBuffer(prim.indexStaging, indexBufferSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
                {
                    std::cerr << "Failed to create index staging buffer" << std::endl;
                    return;
                }

                // Copy index data to staging
                memcpy(prim.indexStaging.cpuPtr, prim.indices.data(), indexBufferSize);

                // Initialize index buffer view
                prim.indexBufferView.BufferLocation = prim.indexBuffer.gpuAddress;
                prim.indexBufferView.Format = DXGI_FORMAT_R32_UINT;
                prim.indexBufferView.SizeInBytes = indexBufferSize;
            }
        }
    }
}

void Model::LoadTextures(Renderer* renderer)
{
    // First load images
    m_GltfModel.images.resize(m_GltfModel.data->images_count);
    for (size_t i = 0; i < m_GltfModel.data->images_count; ++i)
    {
        cgltf_image* img = &m_GltfModel.data->images[i];
        GLTFImage& gltfImg = m_GltfModel.images[i];

        DirectX::ScratchImage image;
        if (img->uri)
        {
            // External image
            std::string uri = img->uri;
            std::string dirStr(this->fileDirectory.begin(), this->fileDirectory.end());
            std::string fullPath = dirStr + uri;
            std::wstring wuri(fullPath.begin(), fullPath.end());
            CHECK_HR(DirectX::LoadFromWICFile(wuri.c_str(), DirectX::WIC_FLAGS_NONE, nullptr, image), "Load external image failed");
            gltfImg.image = new DirectX::ScratchImage(std::move(image));
        }
        else if (img->buffer_view)
        {
            // Embedded image, assume PNG or use WIC
            cgltf_buffer_view* bv = img->buffer_view;
            unsigned char* data = (unsigned char*)bv->buffer->data + bv->offset;
            size_t size = bv->size;
            CHECK_HR(DirectX::LoadFromWICMemory(data, size, DirectX::WIC_FLAGS_NONE, nullptr, image), "Load embedded image failed");
            gltfImg.image = new DirectX::ScratchImage(std::move(image));
        }
        else
        {
            continue;
        }

        const DirectX::TexMetadata& metaData = gltfImg.image->GetMetadata();
        DXGI_FORMAT format = metaData.format;

        if (!renderer->CreateTexture(gltfImg.texture,
            UINT(metaData.width),
            UINT(metaData.height),
            format,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr, // clearColor
            UINT(metaData.mipLevels)))
        {
            std::cerr << "Failed to create texture resource for image: " << i << std::endl;
            continue;
        }
    }

    // Now map textures to images
    m_GltfModel.textures.resize(m_GltfModel.data->textures_count);
    for (size_t i = 0; i < m_GltfModel.data->textures_count; ++i)
    {
        cgltf_texture* tex = &m_GltfModel.data->textures[i];
        if (tex->image)
        {
            size_t imageIndex = tex->image - m_GltfModel.data->images;
            m_GltfModel.textures[i].source = &m_GltfModel.images[imageIndex];
        }
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

    // Set debug counters
    m_TotalNodes = m_GltfModel.nodes.size();
    m_TotalRootNodes = m_GltfModel.rootNodes.size();

    // Compute world AABBs
    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        ComputeWorldAABBs(rootNode, DirectX::XMMatrixIdentity());
    }
}

void Model::ComputeWorldAABBs(GLTFNode* node, DirectX::XMMATRIX parentTransform)
{
    DirectX::XMMATRIX world = DirectX::XMLoadFloat4x4(&node->transform) * parentTransform;

    // Recurse FIRST (Post-order) so children's worldAabbs are calculated
    for (auto* child : node->children)
    {
        ComputeWorldAABBs(child, world);
    }

    // Now compute this node's world AABB including its meshes and all children
    bool initialized = false;
    if (node->mesh)
    {
        for (auto& prim : node->mesh->primitives)
        {
            DirectX::BoundingBox transformedAabb;
            prim.aabb.Transform(transformedAabb, world);
            if (!initialized)
            {
                node->worldAabb = transformedAabb;
                initialized = true;
            }
            else
            {
                DirectX::BoundingBox::CreateMerged(node->worldAabb, node->worldAabb, transformedAabb);
            }
        }
    }

    for (auto* child : node->children)
    {
        if (!initialized)
        {
            node->worldAabb = child->worldAabb;
            initialized = true;
        }
        else
        {
            DirectX::BoundingBox::CreateMerged(node->worldAabb, node->worldAabb, child->worldAabb);
        }
    }

    if (!initialized)
    {
        // If no mesh and no children, set a default AABB at the node's position
        DirectX::XMVECTOR scale, rot, trans;
        DirectX::XMMatrixDecompose(&scale, &rot, &trans, world);
        DirectX::XMFLOAT3 pos;
        DirectX::XMStoreFloat3(&pos, trans);
        node->worldAabb = DirectX::BoundingBox(pos, DirectX::XMFLOAT3(0.0f, 0.0f, 0.0f));
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
                for (size_t k = 0; k < timeAccessor->count; ++k) {
                    if (!cgltf_accessor_read_float(timeAccessor, k, &gltfChannel.times[k], 1)) {
                        std::cerr << "Failed to read animation time at index " << k << std::endl;
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

    // Recompute world AABBs after all local transforms are updated
    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        ComputeWorldAABBs(rootNode, DirectX::XMMatrixIdentity());
    }
}

void Model::UploadTextures(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList, ID3D12CommandQueue* cmdQueue, ID3D12CommandAllocator* cmdAllocator, Renderer* renderer)
{
    // Reset the command list
    CHECK_HR(cmdList->Reset(cmdAllocator, nullptr), "Reset command list failed");

    srvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> uploadBuffers;

    for (size_t i = 0; i < m_GltfModel.images.size(); ++i)
    {
        auto& gltfImg = m_GltfModel.images[i];
        if (!gltfImg.image || !gltfImg.texture.resource)
            continue;

        const DirectX::TexMetadata& metaData = gltfImg.image->GetMetadata();

        // Transition texture to COPY_DEST
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = gltfImg.texture.resource.Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        cmdList->ResourceBarrier(1, &barrier);

        const UINT numSubResources = UINT(metaData.mipLevels * metaData.arraySize);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)_alloca(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) * numSubResources);
        UINT* numRows = (UINT*)_alloca(sizeof(UINT) * numSubResources);
        UINT64* rowSizes = (UINT64*)_alloca(sizeof(UINT64) * numSubResources);

        UINT64 textureMemSize = 0;
        D3D12_RESOURCE_DESC textureDesc = gltfImg.texture.resource->GetDesc();
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
                const DirectX::Image* subImage = gltfImg.image->GetImage(mipIdx, arrayIdx, 0);
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
            dst.pResource = gltfImg.texture.resource.Get();
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

        // Clean up
        delete gltfImg.image;
        gltfImg.image = nullptr;
    }

    // Upload buffers
    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (auto& mesh : m_GltfModel.meshes)
    {
        for (auto& prim : mesh.primitives)
        {
            if (prim.vertexBuffer.resource)
            {
                // Copy vertex buffer
                cmdList->CopyBufferRegion(prim.vertexBuffer.resource.Get(), 0, prim.vertexStaging.resource.Get(), 0, prim.vertices.size() * sizeof(GLTFVertex));
                // Barrier to GENERIC_READ
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition.pResource = prim.vertexBuffer.resource.Get();
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                barriers.push_back(barrier);
            }
            if (prim.indexBuffer.resource)
            {
                // Copy index buffer
                cmdList->CopyBufferRegion(prim.indexBuffer.resource.Get(), 0, prim.indexStaging.resource.Get(), 0, prim.indices.size() * sizeof(uint32_t));
                // Barrier to GENERIC_READ
                D3D12_RESOURCE_BARRIER barrier = {};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition.pResource = prim.indexBuffer.resource.Get();
                barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_GENERIC_READ;
                barriers.push_back(barrier);
            }
        }
    }
    if (!barriers.empty())
    {
        cmdList->ResourceBarrier(static_cast<UINT>(barriers.size()), barriers.data());
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

void Model::Render(ID3D12GraphicsCommandList* commandList, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode)
{
    // Reset debug counter
    m_NodesSurviveFrustum = 0;

    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        RenderNode(commandList, rootNode, DirectX::XMMatrixIdentity(), renderer, frustum, mode);
    }
}

void Model::RenderNode(ID3D12GraphicsCommandList* commandList, GLTFNode* node, DirectX::XMMATRIX parentTransform, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode)
{
    // Frustum culling
    if (node->worldAabb.Intersects(frustum) == false)
    {
        return;
    }

    // Increment debug counter
    ++m_NodesSurviveFrustum;

    DirectX::XMMATRIX world = DirectX::XMLoadFloat4x4(&node->transform) * parentTransform;

    if (node->mesh)
    {
        for (auto& prim : node->mesh->primitives)
        {
            if (prim.alphaMode != mode)
                continue;

            // Set material constants as root constants
            struct {
                float baseColorFactor[4];
                float metallicFactor;
                float roughnessFactor;
                int baseColorTextureIndex;
                float padding;
            } materialCB;

            materialCB.baseColorFactor[0] = prim.material.baseColorFactor[0];
            materialCB.baseColorFactor[1] = prim.material.baseColorFactor[1];
            materialCB.baseColorFactor[2] = prim.material.baseColorFactor[2];
            materialCB.baseColorFactor[3] = prim.material.baseColorFactor[3];
            materialCB.metallicFactor = prim.material.metallicFactor;
            materialCB.roughnessFactor = prim.material.roughnessFactor;
            materialCB.baseColorTextureIndex = (prim.material.baseColorTexture && prim.material.baseColorTexture->source) ? (int)prim.material.baseColorTexture->source->texture.srvIndex : -1;
            materialCB.padding = 0.0f;
            commandList->SetGraphicsRoot32BitConstants(1, 8, &materialCB, 0);

            // Set world matrix as root constants
            float worldFloats[16];
            DirectX::XMStoreFloat4x4(reinterpret_cast<DirectX::XMFLOAT4X4*>(worldFloats), world);
            commandList->SetGraphicsRoot32BitConstants(2, 16, worldFloats, 0);

            // Render the mesh
            commandList->IASetVertexBuffers(0, 1, &prim.vertexBufferView);
            commandList->IASetIndexBuffer(&prim.indexBufferView);
            commandList->DrawIndexedInstanced(static_cast<UINT>(prim.indices.size()), 1, 0, 0, 0);
        }
    }

    for (auto* child : node->children)
    {
        RenderNode(commandList, child, world, renderer, frustum, mode);
    }
}