#include "pch.h"
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include "Model.h"
#include "Renderer.h"
#include "Utility.h"
#include "ResourceUploadBatch.h"
#include <cstdint>
#include <algorithm>
#include "GraphicsTypes.h"
#include "GraphicsHelper.h"
#include "MeshletCache.h"

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
    LoadMaterials();

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
                gltfPrim.materialIndex = static_cast<UINT>(material - m_GltfModel.data->materials);

                // Set alpha mode
                if (material->alpha_mode == cgltf_alpha_mode_opaque)
                     gltfPrim.alphaMode = AlphaMode::Opaque;
                else if (material->alpha_mode == cgltf_alpha_mode_mask)
                    gltfPrim.alphaMode = AlphaMode::Mask;
                else if (material->alpha_mode == cgltf_alpha_mode_blend)
                    gltfPrim.alphaMode = AlphaMode::Blend;
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

    // Generate meshlets for all primitives (with cache)
    for (auto& mesh : m_GltfModel.meshes)
    {
        for (auto& prim : mesh.primitives)
        {
            if (!prim.vertices.empty() && !prim.indices.empty())
            {
                BuildMeshlets(prim);
            }
        }
    }

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
    // Gather all vertices and indices into global buffers
    m_GlobalVertices.clear();
    m_GlobalIndices.clear();

    for (auto& mesh : m_GltfModel.meshes)
    {
        for (auto& prim : mesh.primitives)
        {
            prim.globalVertexOffset = static_cast<uint32_t>(m_GlobalVertices.size());
            prim.globalIndexOffset = static_cast<uint32_t>(m_GlobalIndices.size());

            m_GlobalVertices.insert(m_GlobalVertices.end(), prim.vertices.begin(), prim.vertices.end());
            m_GlobalIndices.insert(m_GlobalIndices.end(), prim.indices.begin(), prim.indices.end());
        }
    }

    // Create global vertex buffer
    if (!m_GlobalVertices.empty())
    {
        if (!CreateStructuredBuffer(m_GlobalVertexBuffer, sizeof(GLTFVertex), m_GlobalVertices.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON))
        {
            std::cerr << "Failed to create global vertex buffer" << std::endl;
            return;
        }
    }

    // Create global index buffer
    if (!m_GlobalIndices.empty())
    {
        if (!CreateStructuredBuffer(m_GlobalIndexBuffer, sizeof(uint32_t), m_GlobalIndices.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON))
        {
            std::cerr << "Failed to create global index buffer" << std::endl;
            return;
        }
    }

    // Pre-calculate node data for all node-primitive pairs
    m_DrawNodeData.clear();
    m_OpaqueCommands.clear();
    m_TransparentCommands.clear();

    for (uint32_t i = 0; i < static_cast<uint32_t>(m_GltfModel.nodes.size()); ++i)
    {
        GLTFNode& node = m_GltfModel.nodes[i];
        node.nodeDataOffset = static_cast<uint32_t>(m_DrawNodeData.size());
        if (node.mesh)
        {
            for (auto& prim : node.mesh->primitives)
            {
                DrawNodeData data;
                DirectX::XMStoreFloat4x4(&data.world, DirectX::XMMatrixIdentity());
                data.vertexOffset = prim.globalVertexOffset;
                data.indexOffset = prim.globalIndexOffset;
                data.materialID = prim.materialIndex;
                m_DrawNodeData.push_back(data);

                // Create indirect command for this primitive (Indexed)
                IndirectDrawCommand cmd;
                cmd.drawArgs.IndexCountPerInstance = static_cast<UINT>(prim.indices.size());
                cmd.drawArgs.InstanceCount = 1;
                cmd.drawArgs.StartIndexLocation = prim.globalIndexOffset;
                cmd.drawArgs.BaseVertexLocation = 0;
                cmd.drawArgs.StartInstanceLocation = static_cast<UINT>(m_DrawNodeData.size() - 1);
                
                if (prim.alphaMode == AlphaMode::Opaque || prim.alphaMode == AlphaMode::Mask)
                    m_OpaqueCommands.push_back(cmd);
                else
                    m_TransparentCommands.push_back(cmd);
            }
        }
    }

    // Create draw node buffer
    if (!m_DrawNodeData.empty())
    {
        if (!CreateStructuredBuffer(m_DrawNodeBuffer, sizeof(DrawNodeData), m_DrawNodeData.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
        {
            std::cerr << "Failed to create draw node buffer" << std::endl;
            return;
        }

        // Create opaque command buffer
        if (!m_OpaqueCommands.empty())
        {
            const UINT64 cmdSize = m_OpaqueCommands.size() * sizeof(IndirectDrawCommand);
            if (!CreateBuffer(m_OpaqueCommandBuffer, cmdSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false))
            {
                std::cerr << "Failed to create opaque indirect draw buffer" << std::endl;
                return;
            }
        }

        // Create transparent command buffer
        if (!m_TransparentCommands.empty())
        {
            const UINT64 cmdSize = m_TransparentCommands.size() * sizeof(IndirectDrawCommand);
            if (!CreateBuffer(m_TransparentCommandBuffer, cmdSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, false))
            {
                std::cerr << "Failed to create transparent indirect draw buffer" << std::endl;
                return;
            }
        }

        // Populate staging buffer immediately with initial transforms
        UpdateNodeBuffer();
    }

    // Create material buffer
    if (!m_MaterialConstants.empty())
    {
        if (!CreateStructuredBuffer(m_MaterialBuffer, sizeof(MaterialConstants), m_MaterialConstants.size(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON))
        {
            std::cerr << "Failed to create material buffer" << std::endl;
            return;
        }
    }

    // Create meshlet GPU buffers (global streams + MeshData + InstanceData)
    CreateMeshletResources(renderer);
}

void Model::LoadTextures(Renderer* renderer)
{
    // First load images
    m_GltfModel.images.resize(m_GltfModel.data->images_count);
    
    // Now map textures to images
    m_GltfModel.textures.resize(m_GltfModel.data->textures_count);
    for (size_t i = 0; i < m_GltfModel.data->textures_count; ++i)
    {
        cgltf_texture* tex = &m_GltfModel.data->textures[i];
        cgltf_image* img = tex->image;

        // Check for MSFT_texture_dds extension
        for (size_t j = 0; j < tex->extensions_count; ++j)
        {
            if (strcmp(tex->extensions[j].name, "MSFT_texture_dds") == 0)
            {
                // Parse the JSON string to find the "source" index
                // The JSON looks like: {"source": 1}
                const char* json = tex->extensions[j].data;
                const char* sourceStr = strstr(json, "\"source\"");
                if (sourceStr)
                {
                    const char* colon = strchr(sourceStr, ':');
                    if (colon)
                    {
                        int sourceIndex = atoi(colon + 1);
                        if (sourceIndex >= 0 && sourceIndex < m_GltfModel.data->images_count)
                        {
                            img = &m_GltfModel.data->images[sourceIndex];
                        }
                    }
                }
                break;
            }
        }

        if (!img)
            continue;

        size_t imageIndex = img - m_GltfModel.data->images;
        GLTFImage& gltfImg = m_GltfModel.images[imageIndex];
        m_GltfModel.textures[i].source = &gltfImg;

        // If image is already loaded, skip
        if (gltfImg.image != nullptr)
            continue;

        DirectX::ScratchImage image;
        if (img->uri)
        {
            // External image
            std::string uri = img->uri;
            std::string dirStr(this->fileDirectory.begin(), this->fileDirectory.end());
            std::string fullPath = dirStr + uri;
            std::wstring wuri(fullPath.begin(), fullPath.end());

            // Check if the file is a DDS texture
            bool isDDS = false;
            if (uri.length() >= 4)
            {
                std::string ext = uri.substr(uri.length() - 4);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".dds")
                {
                    isDDS = true;
                }
            }

            if (isDDS)
            {
                CHECK_HR(DirectX::LoadFromDDSFile(wuri.c_str(), DirectX::DDS_FLAGS_NONE, nullptr, image), "Load external DDS image failed");
            }
            else
            {
                CHECK_HR(DirectX::LoadFromWICFile(wuri.c_str(), DirectX::WIC_FLAGS_NONE, nullptr, image), "Load external image failed");
            }
            gltfImg.image = new DirectX::ScratchImage(std::move(image));
        }
        else if (img->buffer_view)
        {
            // Embedded image, assume PNG or use WIC
            cgltf_buffer_view* bv = img->buffer_view;
            unsigned char* data = (unsigned char*)bv->buffer->data + bv->offset;
            size_t size = bv->size;

            // Check for DDS magic number
            bool isDDS = false;
            if (size >= 4)
            {
                uint32_t magic = *reinterpret_cast<uint32_t*>(data);
                if (magic == 0x20534444) // 'DDS '
                {
                    isDDS = true;
                }
            }

            if (isDDS)
            {
                CHECK_HR(DirectX::LoadFromDDSMemory(data, size, DirectX::DDS_FLAGS_NONE, nullptr, image), "Load embedded DDS image failed");
            }
            else
            {
                CHECK_HR(DirectX::LoadFromWICMemory(data, size, DirectX::WIC_FLAGS_NONE, nullptr, image), "Load embedded image failed");
            }
            gltfImg.image = new DirectX::ScratchImage(std::move(image));
        }
        else
        {
            continue;
        }

        const DirectX::TexMetadata& metaData = gltfImg.image->GetMetadata();
        DXGI_FORMAT format = metaData.format;

        if (!CreateTexture(gltfImg.texture,
            UINT(metaData.width),
            UINT(metaData.height),
            format,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr, // clearColor
            UINT(metaData.mipLevels),
            UINT(metaData.arraySize)))
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

void Model::LoadMaterials()
{
    m_MaterialConstants.resize(m_GltfModel.data->materials_count);

    for (size_t i = 0; i < m_GltfModel.data->materials_count; ++i)
    {
        cgltf_material* material = &m_GltfModel.data->materials[i];
        MaterialConstants& mc = m_MaterialConstants[i];

        // Default values
        mc.baseColorFactor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
        mc.metallicFactor = 1.0f;
        mc.roughnessFactor = 1.0f;
        mc.baseColorTextureIndex = -1;
        mc.normalTextureIndex = -1;
        mc.metallicRoughnessTextureIndex = -1;
        mc.alphaMode = 0; // Opaque
        mc.alphaCutoff = 0.5f;

        if (material->alpha_mode == cgltf_alpha_mode_mask)
            mc.alphaMode = 1;
        else if (material->alpha_mode == cgltf_alpha_mode_blend)
            mc.alphaMode = 2;

        mc.alphaCutoff = material->alpha_cutoff;

        if (material->has_pbr_metallic_roughness)
        {
            mc.baseColorFactor = DirectX::XMFLOAT4(
                material->pbr_metallic_roughness.base_color_factor[0],
                material->pbr_metallic_roughness.base_color_factor[1],
                material->pbr_metallic_roughness.base_color_factor[2],
                material->pbr_metallic_roughness.base_color_factor[3]);
            mc.metallicFactor = material->pbr_metallic_roughness.metallic_factor;
            mc.roughnessFactor = material->pbr_metallic_roughness.roughness_factor;

            if (material->pbr_metallic_roughness.base_color_texture.texture)
            {
                size_t texIndex = material->pbr_metallic_roughness.base_color_texture.texture - m_GltfModel.data->textures;
                if (texIndex < m_GltfModel.textures.size())
                {
                    GLTFTexture* baseColorTexture = &m_GltfModel.textures[texIndex];
                    if (baseColorTexture->source)
                    {
                        mc.baseColorTextureIndex = baseColorTexture->source->texture.srvIndex;
                    }
                }
            }

            if (material->pbr_metallic_roughness.metallic_roughness_texture.texture)
            {
                size_t texIndex = material->pbr_metallic_roughness.metallic_roughness_texture.texture - m_GltfModel.data->textures;
                if (texIndex < m_GltfModel.textures.size())
                {
                    GLTFTexture* mrTexture = &m_GltfModel.textures[texIndex];
                    if (mrTexture->source)
                    {
                        mc.metallicRoughnessTextureIndex = mrTexture->source->texture.srvIndex;
                    }
                }
            }
        }

        if (material->normal_texture.texture)
        {
            size_t texIndex = material->normal_texture.texture - m_GltfModel.data->textures;
            if (texIndex < m_GltfModel.textures.size())
            {
                GLTFTexture* normalTexture = &m_GltfModel.textures[texIndex];
                if (normalTexture->source)
                {
                    mc.normalTextureIndex = normalTexture->source->texture.srvIndex;
                }
            }
        }
    }

    if (m_MaterialConstants.empty())
    {
        MaterialConstants mc;
        mc.baseColorFactor = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
        mc.metallicFactor = 1.0f;
        mc.roughnessFactor = 1.0f;
        mc.baseColorTextureIndex = -1;
        mc.normalTextureIndex = -1;
        mc.metallicRoughnessTextureIndex = -1;
        mc.alphaMode = 0;
        mc.alphaCutoff = 0.5f;
        m_MaterialConstants.push_back(mc);
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

void Model::UpdateNodeBuffer()
{
    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        UpdateNodeBufferRecursive(rootNode, DirectX::XMMatrixIdentity());
    }

    if (m_DrawNodeBuffer.cpuPtr)
    {
        memcpy(m_DrawNodeBuffer.cpuPtr, m_DrawNodeData.data(), m_DrawNodeData.size() * sizeof(DrawNodeData));
    }

    // Also update InstanceData transforms (meshlet path)
    if (m_InstanceDataBuffer.cpuPtr && !m_InstanceDataArray.empty())
    {
        memcpy(m_InstanceDataBuffer.cpuPtr, m_InstanceDataArray.data(), m_InstanceDataArray.size() * sizeof(InstanceData));
    }
}

void Model::UpdateNodeBufferRecursive(GLTFNode* node, DirectX::XMMATRIX parentTransform)
{
    DirectX::XMMATRIX world = DirectX::XMLoadFloat4x4(&node->transform) * parentTransform;

    if (node->mesh)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(node->mesh->primitives.size()); ++i)
        {
            uint32_t nodeDataIndex = node->nodeDataOffset + i;
            DirectX::XMStoreFloat4x4(&m_DrawNodeData[nodeDataIndex].world, world);

            // Also update InstanceData for meshlet path
            if (nodeDataIndex < static_cast<uint32_t>(m_InstanceDataArray.size()))
            {
                DirectX::XMStoreFloat4x4(&m_InstanceDataArray[nodeDataIndex].LocalToWorld, world);
            }
        }
    }

    for (auto* child : node->children)
    {
        UpdateNodeBufferRecursive(child, world);
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

    // Recompute world AABBs and update GPU node buffer after all local transforms are updated
    for (auto* rootNode : m_GltfModel.rootNodes)
    {
        ComputeWorldAABBs(rootNode, DirectX::XMMatrixIdentity());
    }
    
    UpdateNodeBuffer();
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

    // Execute the texture upload commands (which were recorded into cmdList)
    CHECK_HR(cmdList->Close(), "Close command list failed");
    ID3D12CommandList* commandLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, commandLists);

    // Wait for completion (for textures)
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create fence failed");
    HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    CHECK_HR(fence->SetEventOnCompletion(1, eventHandle), "Set event on completion failed");
    cmdQueue->Signal(fence.Get(), 1);
    WaitForSingleObject(eventHandle, INFINITE);
    CloseHandle(eventHandle);

    // The command list remains closed; BeginFrame will reset it
}

void Model::UploadBuffers(Renderer* renderer)
{
    ResourceUploadBatch batch(renderer);
    batch.Begin();

    if (m_MaterialBuffer.resource)
    {
        batch.Upload(m_MaterialBuffer, m_MaterialConstants.data(), m_MaterialConstants.size() * sizeof(MaterialConstants));
        batch.Transition(m_MaterialBuffer, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    if (m_OpaqueCommandBuffer.resource)
    {
        batch.Upload(m_OpaqueCommandBuffer, m_OpaqueCommands.data(), m_OpaqueCommands.size() * sizeof(IndirectDrawCommand));
        batch.Transition(m_OpaqueCommandBuffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    if (m_TransparentCommandBuffer.resource)
    {
        batch.Upload(m_TransparentCommandBuffer, m_TransparentCommands.data(), m_TransparentCommands.size() * sizeof(IndirectDrawCommand));
        batch.Transition(m_TransparentCommandBuffer, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    }

    if (m_GlobalVertexBuffer.resource)
    {
        batch.Upload(m_GlobalVertexBuffer, m_GlobalVertices.data(), m_GlobalVertices.size() * sizeof(GLTFVertex));
        batch.Transition(m_GlobalVertexBuffer, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    if (m_GlobalIndexBuffer.resource)
    {
        batch.Upload(m_GlobalIndexBuffer, m_GlobalIndices.data(), m_GlobalIndices.size() * sizeof(uint32_t));
        batch.Transition(m_GlobalIndexBuffer, D3D12_RESOURCE_STATE_INDEX_BUFFER);
    }

    // Meshlet global stream buffers — direct .data()/.size(), same pattern as
    // GlobalVertexBuffer / GlobalIndexBuffer.  Data built once in CreateMeshletResources().
    if (m_GlobalPositions.resource && !m_AllPositions.empty())
    {
        batch.Upload(m_GlobalPositions, m_AllPositions.data(), m_AllPositions.size() * sizeof(float3));
        batch.Transition(m_GlobalPositions, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalNormals.resource && !m_AllPackedNormals.empty())
    {
        batch.Upload(m_GlobalNormals, m_AllPackedNormals.data(), m_AllPackedNormals.size() * sizeof(uint32_t));
        batch.Transition(m_GlobalNormals, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalUVs.resource && !m_AllPackedUVs.empty())
    {
        batch.Upload(m_GlobalUVs, m_AllPackedUVs.data(), m_AllPackedUVs.size() * sizeof(uint32_t));
        batch.Transition(m_GlobalUVs, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalMeshlets.resource && !m_AllMeshlets.empty())
    {
        batch.Upload(m_GlobalMeshlets, m_AllMeshlets.data(), m_AllMeshlets.size() * sizeof(Meshlet));
        batch.Transition(m_GlobalMeshlets, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalMeshletVertices.resource && !m_AllMeshletVertices.empty())
    {
        batch.Upload(m_GlobalMeshletVertices, m_AllMeshletVertices.data(), m_AllMeshletVertices.size() * sizeof(uint32_t));
        batch.Transition(m_GlobalMeshletVertices, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalMeshletTriangles.resource && !m_AllMeshletTriangles.empty())
    {
        batch.Upload(m_GlobalMeshletTriangles, m_AllMeshletTriangles.data(), m_AllMeshletTriangles.size() * sizeof(MeshletTriangle));
        batch.Transition(m_GlobalMeshletTriangles, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_GlobalMeshletBounds.resource && !m_AllMeshletBounds.empty())
    {
        batch.Upload(m_GlobalMeshletBounds, m_AllMeshletBounds.data(), m_AllMeshletBounds.size() * sizeof(MeshletBounds));
        batch.Transition(m_GlobalMeshletBounds, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    if (m_MeshDataBuffer.resource && !m_MeshDataArray.empty())
    {
        batch.Upload(m_MeshDataBuffer, m_MeshDataArray.data(), m_MeshDataArray.size() * sizeof(MeshData));
        batch.Transition(m_MeshDataBuffer, D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    batch.End();
}

void Model::Render(ID3D12GraphicsCommandList* commandList, Renderer* renderer, const DirectX::BoundingFrustum& frustum, AlphaMode mode)
{
    // Reset debug counter
    m_NodesSurviveFrustum = 0;

    // Bind material buffer to root parameter 1
    if (m_MaterialBuffer.resource)
    {
        commandList->SetGraphicsRootShaderResourceView(1, m_MaterialBuffer.gpuAddress);
    }

    // Bind draw node buffer to root parameter 2
    if (m_DrawNodeBuffer.resource)
    {
        commandList->SetGraphicsRootShaderResourceView(2, m_DrawNodeBuffer.gpuAddress);
    }

    // Bind global vertex buffer to root parameter 6
    if (m_GlobalVertexBuffer.resource)
    {
        commandList->SetGraphicsRootShaderResourceView(6, m_GlobalVertexBuffer.gpuAddress);
    }

    // Bind global index buffer to IA
    if (m_GlobalIndexBuffer.resource)
    {
        D3D12_INDEX_BUFFER_VIEW ibv = {};
        ibv.BufferLocation = m_GlobalIndexBuffer.gpuAddress;
        ibv.SizeInBytes = static_cast<UINT>(m_GlobalIndexBuffer.size);
        ibv.Format = DXGI_FORMAT_R32_UINT;
        commandList->IASetIndexBuffer(&ibv);
    }

    // Unbind IA vertex buffers (using vertex pulling)
    commandList->IASetVertexBuffers(0, 0, nullptr);

    // Execute indirect draw
    ID3D12Resource* cmdBuffer = nullptr;
    UINT cmdCount = 0;

    if (mode == AlphaMode::Opaque || mode == AlphaMode::Mask)
    {
        cmdBuffer = m_OpaqueCommandBuffer.resource.Get();
        cmdCount = static_cast<UINT>(m_OpaqueCommands.size());
    }
    else
    {
        cmdBuffer = m_TransparentCommandBuffer.resource.Get();
        cmdCount = static_cast<UINT>(m_TransparentCommands.size());
    }

    if (cmdBuffer && cmdCount > 0)
    {
        commandList->ExecuteIndirect(
            renderer->GetCommandSignature(),
            cmdCount,
            cmdBuffer,
            0,
            nullptr,
            0);
    }
}

// RenderNode recursively (Keep it for debugging or future culling, but not used by ExecuteIndirect right now)
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
        for (uint32_t i = 0; i < static_cast<uint32_t>(node->mesh->primitives.size()); ++i)
        {
            auto& prim = node->mesh->primitives[i];
            if (prim.alphaMode != mode)
                continue;

            // Render the mesh using programmable vertex pulling
            // StartInstanceLocation serves as our index into m_DrawNodeBuffer
            uint32_t nodeDataIndex = node->nodeDataOffset + i;
            commandList->DrawIndexedInstanced(static_cast<UINT>(prim.indices.size()), 1, prim.globalIndexOffset, 0, nodeDataIndex);
        }
    }

    for (auto* child : node->children)
    {
        RenderNode(commandList, child, world, renderer, frustum, mode);
    }
}

void Model::GetAllPrimitives(std::vector<const GLTFPrimitive*>& primitives) const
{
    for (const auto& mesh : m_GltfModel.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            primitives.push_back(&primitive);
        }
    }
}

void Model::GetDrawNodePrimitives(std::vector<const GLTFPrimitive*>& primitives) const
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_GltfModel.nodes.size()); ++i)
    {
        const GLTFNode& node = m_GltfModel.nodes[i];
        if (node.mesh)
        {
            for (const auto& prim : node.mesh->primitives)
            {
                primitives.push_back(&prim);
            }
        }
    }
}

// =============================================================================
// Meshlet Generation (Phase 1 + Phase 2) with disk cache
// =============================================================================

// Helper: pack float3 normal into RGB10A2_SNORM for GPU storage
static uint32_t PackNormalRGB10A2_SNORM(float x, float y, float z)
{
    // Clamp to [-1, 1], then map to [0, 1023] (10-bit) with signed → unsigned offset
    auto pack10 = [](float v) -> uint32_t {
        int32_t s = static_cast<int32_t>(std::round(std::clamp(v, -1.0f, 1.0f) * 511.0f));
        return static_cast<uint32_t>(std::max(-512, std::min(511, s)) + 512) & 0x3FF;
    };
    auto pack2 = [](float v) -> uint32_t {
        int32_t s = static_cast<int32_t>(std::round(std::clamp(v, -1.0f, 1.0f) * 1.0f));
        return static_cast<uint32_t>(std::max(-2, std::min(1, s)) + 2) & 0x3;
    };
    return (pack10(x) << 0) | (pack10(y) << 10) | (pack10(z) << 20) | (pack2(1.0f) << 30);
}

// Helper: pack float2 UV into RG16_FLOAT
static uint32_t PackUVRG16_FLOAT(float u, float v)
{
    // Use half-precision float packing
    auto f32_to_f16 = [](float f) -> uint16_t {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t sign     = (bits >> 16) & 0x8000;
        int32_t  exponent = ((bits >> 23) & 0xFF) - 112;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exponent <= 0) { // subnormal
            mantissa = (mantissa | 0x800000) >> (1 - exponent);
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }
        if (exponent > 30) { // INF/NAN → INF
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
    };
    uint16_t hU = f32_to_f16(u);
    uint16_t hV = f32_to_f16(v);
    return (static_cast<uint32_t>(hV) << 16) | static_cast<uint32_t>(hU);
}

void Model::BuildMeshlets(GLTFPrimitive& prim)
{
    const size_t vertexCount = prim.vertices.size();
    const size_t indexCount  = prim.indices.size();
    if (vertexCount == 0 || indexCount == 0)
        return;

    // Path for the cache file
    // Note: BuildMeshlets is called per-primitive; we don't have the original GLTF path easily.
    // We'll just skip cache for now and always regenerate. The cache infrastructure is ready
    // but needs the file path plumbing from LoadGLTFModel.
    //
    // In a production setup, we'd pass the filepath and primitive index to BuildMeshlets.

    // --- Extract flat position array ---
    std::vector<float> positions(vertexCount * 3);
    for (size_t i = 0; i < vertexCount; ++i)
    {
        positions[i * 3 + 0] = prim.vertices[i].position[0];
        positions[i * 3 + 1] = prim.vertices[i].position[1];
        positions[i * 3 + 2] = prim.vertices[i].position[2];
    }

    // --- Phase 1: Mesh Optimization (always run before meshlet generation) ---
    // 1a: Vertex cache optimization
    meshopt_optimizeVertexCache(prim.indices.data(), prim.indices.data(),
                                indexCount, vertexCount);

    // 1b: Overdraw optimization
    meshopt_optimizeOverdraw(prim.indices.data(), prim.indices.data(),
                             indexCount, positions.data(), vertexCount,
                             sizeof(float) * 3, 1.05f);

    // 1c: Vertex fetch optimization — remap vertices for spatial locality
    std::vector<uint32_t> remap(vertexCount);
    size_t uniqueVertices = meshopt_optimizeVertexFetchRemap(
        &remap[0], prim.indices.data(), indexCount, vertexCount);

    // Apply remap to vertices (resize to unique count)
    {
        std::vector<GLTFVertex> remappedVertices(uniqueVertices);
        for (size_t i = 0; i < vertexCount; ++i)
        {
            if (remap[i] != ~0u)
            {
                remappedVertices[remap[i]] = prim.vertices[i];
            }
        }
        prim.vertices = std::move(remappedVertices);
    }

    // Remap indices
    meshopt_remapIndexBuffer(prim.indices.data(), prim.indices.data(),
                             indexCount, &remap[0]);

    // Re-build positions array after remap
    const size_t newVertexCount = prim.vertices.size();
    positions.resize(newVertexCount * 3);
    for (size_t i = 0; i < newVertexCount; ++i)
    {
        positions[i * 3 + 0] = prim.vertices[i].position[0];
        positions[i * 3 + 1] = prim.vertices[i].position[1];
        positions[i * 3 + 2] = prim.vertices[i].position[2];
    }

    // --- Phase 2: Meshlet Generation ---
    const size_t maxMeshlets = meshopt_buildMeshletsBound(
        indexCount, MESHLET_MAX_VERTICES, MESHLET_MAX_TRIANGLES);

    std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
    std::vector<uint32_t> meshletVertices(maxMeshlets * MESHLET_MAX_VERTICES);
    std::vector<uint8_t>  meshletTriangles(maxMeshlets * MESHLET_MAX_TRIANGLES * 3);

    size_t meshletCount = meshopt_buildMeshlets(
        meshlets.data(),
        meshletVertices.data(),
        meshletTriangles.data(),
        prim.indices.data(), indexCount,
        positions.data(), newVertexCount, sizeof(float) * 3,
        MESHLET_MAX_VERTICES, MESHLET_MAX_TRIANGLES,
        0.0f /* cone weight = 0 */);

    meshlets.resize(meshletCount);
    meshletVertices.resize(meshlets.back().vertex_offset + meshlets.back().vertex_count);
    meshletTriangles.resize((meshlets.back().triangle_offset + meshlets.back().triangle_count) * 3);

    // Per-meshlet optimization + AABB computation
    for (auto& m : meshlets)
    {
        meshopt_optimizeMeshlet(
            &meshletVertices[m.vertex_offset],
            &meshletTriangles[m.triangle_offset],
            m.triangle_count, m.vertex_count);
    }

    // --- Pack into GPU-ready Meshlet structs ---
    // Store in CPU-side vectors for later upload to global buffers
    auto& allMeshlets         = prim.meshlets;
    auto& allMeshletVertices  = prim.meshletVertices;
    auto& allMeshletTriangles = prim.meshletTriangles;
    auto& allMeshletBounds    = prim.meshletBounds;

    allMeshlets.clear();
    allMeshletVertices.clear();
    allMeshletTriangles.clear();
    allMeshletBounds.clear();

    // Copy meshlet vertices (indirection table) — these are uint32 indices
    allMeshletVertices.assign(meshletVertices.begin(), meshletVertices.end());

    for (auto& m : meshlets)
    {
        // GPU Meshlet header
        Meshlet gm;
        gm.VertexOffset   = m.vertex_offset;
        gm.TriangleOffset = m.triangle_offset;
        gm.VertexCount    = m.vertex_count;
        gm.TriangleCount  = m.triangle_count;
        allMeshlets.push_back(gm);

        // Pack triangles into MeshletTriangle struct
        for (uint32_t t = 0; t < m.triangle_count; ++t)
        {
            uint8_t* triBase = &meshletTriangles[(m.triangle_offset + t) * 3];
            MeshletTriangle mt;
            mt.V0 = triBase[0];
            mt.V1 = triBase[1];
            mt.V2 = triBase[2];
            allMeshletTriangles.push_back(mt);
        }

        // Compute AABB bounds
        float3 minBounds = { FLT_MAX, FLT_MAX, FLT_MAX };
        float3 maxBounds = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (uint32_t v = 0; v < m.vertex_count; ++v)
        {
            uint32_t globalIdx = meshletVertices[m.vertex_offset + v];
            float3 pos = { positions[globalIdx * 3 + 0],
                           positions[globalIdx * 3 + 1],
                           positions[globalIdx * 3 + 2] };
            minBounds.x = std::min(minBounds.x, pos.x);
            minBounds.y = std::min(minBounds.y, pos.y);
            minBounds.z = std::min(minBounds.z, pos.z);
            maxBounds.x = std::max(maxBounds.x, pos.x);
            maxBounds.y = std::max(maxBounds.y, pos.y);
            maxBounds.z = std::max(maxBounds.z, pos.z);
        }
        MeshletBounds bounds;
        bounds.LocalCenter.x  = (minBounds.x + maxBounds.x) * 0.5f;
        bounds.LocalCenter.y  = (minBounds.y + maxBounds.y) * 0.5f;
        bounds.LocalCenter.z  = (minBounds.z + maxBounds.z) * 0.5f;
        bounds.LocalExtents.x = (maxBounds.x - minBounds.x) * 0.5f;
        bounds.LocalExtents.y = (maxBounds.y - minBounds.y) * 0.5f;
        bounds.LocalExtents.z = (maxBounds.z - minBounds.z) * 0.5f;
        allMeshletBounds.push_back(bounds);
    }

    std::cout << "[Meshlet] Generated " << meshletCount << " meshlets from "
              << newVertexCount << " vertices, " << indexCount << " triangles" << std::endl;
}

// =============================================================================
// CreateMeshletResources: creates 7 global StructuredBuffer + MeshData + InstanceData
// =============================================================================
void Model::CreateMeshletResources(Renderer* renderer)
{
    // Accumulate into member vectors (persist across function calls, same pattern as
    // m_GlobalVertices / m_GlobalIndices).  Clear first — this is a build-once path.
    m_AllPositions.clear();
    m_AllPackedNormals.clear();
    m_AllPackedUVs.clear();
    m_AllMeshlets.clear();
    m_AllMeshletVertices.clear();
    m_AllMeshletTriangles.clear();
    m_AllMeshletBounds.clear();

    m_MeshDataArray.clear();
    m_InstanceDataArray.clear();
    m_TotalMeshletCount = 0;

    // Walk nodes and populate consolidated streams
    for (size_t nodeIdx = 0; nodeIdx < m_GltfModel.nodes.size(); ++nodeIdx)
    {
        GLTFNode& node = m_GltfModel.nodes[nodeIdx];
        if (!node.mesh)
            continue;

        for (auto& prim : node.mesh->primitives)
        {
            if (prim.vertices.empty())
                continue;

            MeshData md = {};
            md.MeshletCount    = static_cast<uint32_t>(prim.meshlets.size());
            md.MaterialIndex   = prim.materialIndex;

            // Source vertex streams (positions, normals, UVs)
            md.PositionOffset = static_cast<uint32_t>(m_AllPositions.size());
            md.NormalOffset   = static_cast<uint32_t>(m_AllPackedNormals.size());
            md.UVOffset       = static_cast<uint32_t>(m_AllPackedUVs.size());

            for (const auto& v : prim.vertices)
            {
                m_AllPositions.push_back({ v.position[0], v.position[1], v.position[2] });
                m_AllPackedNormals.push_back(PackNormalRGB10A2_SNORM(v.normal[0], v.normal[1], v.normal[2]));
                m_AllPackedUVs.push_back(PackUVRG16_FLOAT(v.texCoord[0], v.texCoord[1]));
            }

            // Meshlet streams
            md.MeshletOffset         = static_cast<uint32_t>(m_AllMeshlets.size());
            md.MeshletVertexOffset   = static_cast<uint32_t>(m_AllMeshletVertices.size());
            md.MeshletTriangleOffset = static_cast<uint32_t>(m_AllMeshletTriangles.size());
            md.MeshletBoundsOffset   = static_cast<uint32_t>(m_AllMeshletBounds.size());

            m_AllMeshlets.insert(m_AllMeshlets.end(), prim.meshlets.begin(), prim.meshlets.end());
            m_AllMeshletVertices.insert(m_AllMeshletVertices.end(), prim.meshletVertices.begin(), prim.meshletVertices.end());
            m_AllMeshletTriangles.insert(m_AllMeshletTriangles.end(), prim.meshletTriangles.begin(), prim.meshletTriangles.end());
            m_AllMeshletBounds.insert(m_AllMeshletBounds.end(), prim.meshletBounds.begin(), prim.meshletBounds.end());

            m_MeshDataArray.push_back(md);
            m_TotalMeshletCount += md.MeshletCount;

            // Per-instance data (matches DrawNodeData cardinality)
            InstanceData inst = {};
            DirectX::XMStoreFloat4x4(&inst.LocalToWorld, DirectX::XMMatrixIdentity());
            inst.MeshDataIndex = static_cast<uint32_t>(m_MeshDataArray.size() - 1);
            m_InstanceDataArray.push_back(inst);
        }
    }

    if (m_AllPositions.empty())
    {
        std::cout << "[Meshlet] No meshlet data to upload." << std::endl;
        return;
    }

    // Create 7 global StructuredBuffer<T>
    auto createSB = [](GPUBuffer& buf, UINT64 elemSize, UINT64 count, const char* name) -> bool {
        if (count == 0) return true;
        if (!CreateStructuredBuffer(buf, elemSize, count, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON)) {
            std::cerr << "[Meshlet] Failed to create " << name << std::endl;
            return false;
        }
        return true;
    };

    if (!createSB(m_GlobalPositions,        sizeof(float3),          m_AllPositions.size(),        "GlobalPositions")) return;
    if (!createSB(m_GlobalNormals,          sizeof(uint32_t),        m_AllPackedNormals.size(),    "GlobalNormals"))   return;
    if (!createSB(m_GlobalUVs,              sizeof(uint32_t),        m_AllPackedUVs.size(),        "GlobalUVs"))       return;
    if (!createSB(m_GlobalMeshlets,         sizeof(Meshlet),         m_AllMeshlets.size(),         "GlobalMeshlets"))  return;
    if (!createSB(m_GlobalMeshletVertices,  sizeof(uint32_t),        m_AllMeshletVertices.size(),  "MeshletVertices")) return;
    if (!createSB(m_GlobalMeshletTriangles, sizeof(MeshletTriangle), m_AllMeshletTriangles.size(), "MeshletTriangles"))return;
    if (!createSB(m_GlobalMeshletBounds,    sizeof(MeshletBounds),   m_AllMeshletBounds.size(),    "MeshletBounds"))  return;

    // Create MeshData and InstanceData buffers
    if (!m_MeshDataArray.empty()) {
        CreateStructuredBuffer(m_MeshDataBuffer, sizeof(MeshData), m_MeshDataArray.size(), 
                              D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    }
    if (!m_InstanceDataArray.empty()) {
        CreateStructuredBuffer(m_InstanceDataBuffer, sizeof(InstanceData), m_InstanceDataArray.size(),
                              D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    }

    m_MeshletReady = true;

    std::cout << "[Meshlet] Uploaded " << m_TotalMeshletCount << " total meshlets across "
              << m_MeshDataArray.size() << " primitives." << std::endl;
    std::cout << "[Meshlet]  Positions: " << m_AllPositions.size() << "  Normals: " << m_AllPackedNormals.size()
              << "  UVs: " << m_AllPackedUVs.size() << std::endl;
}