#include "pch.h"

#define JSMN_HEADER
#define JSMN_IMPLEMENTATION
#include <jsmn.h>

#include "Scene.h"
#include <sstream>
#include <cstring>

using namespace DirectX;

// Helper functions for JSMN parsing (based on cgltf implementation)
static int cgltf_json_strcmp(const jsmntok_t* tok, const uint8_t* json_chunk, const char* str)
{
    if (tok->type != JSMN_STRING) return 1;
    size_t str_len = strlen(str);
    size_t name_length = tok->end - tok->start;
    return (str_len == name_length) ? _strnicmp((const char*)json_chunk + tok->start, str, str_len) : 128;
}

static float cgltf_json_to_float(const jsmntok_t* tok, const uint8_t* json_chunk)
{
    if (tok->type != JSMN_PRIMITIVE) return 0.0f;
    char tmp[128];
    int size = (tok->end - tok->start) < (int)sizeof(tmp) ? (tok->end - tok->start) : (int)(sizeof(tmp) - 1);
    memcpy(tmp, (const char*)json_chunk + tok->start, size);
    tmp[size] = 0;
    return (float)atof(tmp);
}

static int cgltf_skip_json(const jsmntok_t* tokens, int i)
{
    // Use the token's byte range (start/end) for OBJECT and ARRAY to reliably
    // skip all children, since JSMN's size field can be undercounted when
    // JSMN_PARENT_LINKS is enabled and there are nested containers.
    int type = tokens[i].type;
    int end_pos = tokens[i].end; // byte position after this token in JSON string
    i++;
    if (type == JSMN_OBJECT || type == JSMN_ARRAY)
    {
        // All child tokens have start < end_pos of their parent container.
        // Tokens following this container have start >= end_pos.
        while (tokens[i].start < end_pos)
        {
            i++;
        }
    }
    return i;
}

static int cgltf_parse_json_float_array(const jsmntok_t* tokens, int i, const uint8_t* json_chunk, float* out_array, int size)
{
    if (tokens[i].type != JSMN_ARRAY || tokens[i].size != size) return -1;
    ++i;
    for (int j = 0; j < size; ++j)
    {
        out_array[j] = cgltf_json_to_float(tokens + i, json_chunk);
        ++i;
    }
    return i;
}

// Local helpers for string conversion since cgltf functions take uint8_t*
static std::string JsonString(const uint8_t* json, const jsmntok_t* token)
{
    return std::string((const char*)(json + token->start), token->end - token->start);
}

// Forward declaration
static int ParseGraphNode(Scene* scene, const uint8_t* json, const jsmntok_t* tokens, int nodeIndex);

Scene::Scene()
{
}

Scene::~Scene()
{
}

bool Scene::LoadScene(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open scene file: " << path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    m_JsonContent = buffer.str();

    return ParseJson(m_JsonContent.c_str(), m_JsonContent.length());
}

bool Scene::ParseJson(const char* json, size_t length)
{
    const uint8_t* json_ptr = (const uint8_t*)json;
    jsmn_parser parser;
    jsmn_init(&parser);

    // First pass to count tokens
    int tokenCount = jsmn_parse(&parser, json, length, nullptr, 0);
    if (tokenCount < 0)
    {
        std::cerr << "Failed to parse JSON: " << tokenCount << std::endl;
        return false;
    }

    std::vector<jsmntok_t> tokens(tokenCount);
    jsmn_init(&parser);
    jsmn_parse(&parser, json, length, tokens.data(), tokenCount);

    if (tokens[0].type != JSMN_OBJECT) return false;

    m_Lights.clear();
    m_ModelPath = "";

    // Iterate root object keys
    int i = 1;
    while (i < tokenCount)
    {
        if (cgltf_json_strcmp(&tokens[i], json_ptr, "models") == 0)
        {
            // models is array of strings
            int arrayIndex = i + 1;
            if (tokens[arrayIndex].type == JSMN_ARRAY && tokens[arrayIndex].size > 0)
            {
                // Take first model
                m_ModelPath = JsonString(json_ptr, &tokens[arrayIndex + 1]);
            }
            i = cgltf_skip_json(tokens.data(), arrayIndex);
        }
        else if (cgltf_json_strcmp(&tokens[i], json_ptr, "graph") == 0)
        {
             // generic graph parser
             int graphIndex = i + 1;
             int graphEnd = cgltf_skip_json(tokens.data(), graphIndex);
             
             int currentNodeIdx = graphIndex + 1;
             while (currentNodeIdx < graphEnd)
             {
                 currentNodeIdx = ParseGraphNode(this, json_ptr, tokens.data(), currentNodeIdx);
             }
             i = graphEnd;
        }
        else
        {
            // Unknown key, skip value
            i = cgltf_skip_json(tokens.data(), i + 1);
        }
    }

    return true;
}

static int ParseGraphNode(Scene* scene, const uint8_t* json, const jsmntok_t* tokens, int nodeIndex)
{
    // nodeIndex points to Object { ... }
    // NOTE: We cannot rely on tokens[nodeIndex].size for the key count because
    // JSMN with JSMN_PARENT_LINKS may undercount keys when nested arrays exist
    // inside an object that is itself inside an array. Instead, compute the end
    // of this node and iterate until we reach it.
    int nodeEnd = cgltf_skip_json(tokens, nodeIndex);
    int cur = nodeIndex + 1;
    
    // Properties to extract
    std::string type;
    std::string name;
    XMFLOAT3 translation = { 0, 0, 0 };
    XMFLOAT3 direction = { 0, -1, 0 }; // Default direction
    XMFLOAT3 color = { 1, 1, 1 };
    float intensity = 1.0f;
    float innerAngle = 0.0f;
    float outerAngle = 0.0f;
    
    while (cur < nodeEnd)
    {
        const jsmntok_t* keyToken = &tokens[cur];
        const jsmntok_t* valueToken = &tokens[cur + 1];

        if (cgltf_json_strcmp(keyToken, json, "type") == 0)
        {
            type = JsonString(json, valueToken);
        }
        else if (cgltf_json_strcmp(keyToken, json, "name") == 0)
        {
            name = JsonString(json, valueToken);
        }
        else if (cgltf_json_strcmp(keyToken, json, "translation") == 0)
        {
            cgltf_parse_json_float_array(tokens, cur + 1, json, &translation.x, 3);
        }
        else if (cgltf_json_strcmp(keyToken, json, "direction") == 0)
        {
            cgltf_parse_json_float_array(tokens, cur + 1, json, &direction.x, 3);
        }
        else if (cgltf_json_strcmp(keyToken, json, "color") == 0)
        {
            cgltf_parse_json_float_array(tokens, cur + 1, json, &color.x, 3);
        }
        else if (cgltf_json_strcmp(keyToken, json, "intensity") == 0)
        {
            intensity = cgltf_json_to_float(valueToken, json);
        }
        else if (cgltf_json_strcmp(keyToken, json, "irradiance") == 0)
        {
             intensity = cgltf_json_to_float(valueToken, json);
        }
        else if (cgltf_json_strcmp(keyToken, json, "innerAngle") == 0)
        {
            innerAngle = cgltf_json_to_float(valueToken, json);
        }
        else if (cgltf_json_strcmp(keyToken, json, "outerAngle") == 0)
        {
            outerAngle = cgltf_json_to_float(valueToken, json);
        }
        else if (cgltf_json_strcmp(keyToken, json, "children") == 0)
        {
            if (valueToken->type == JSMN_ARRAY)
            {
                int arrayEnd = cgltf_skip_json(tokens, cur + 1);
                int childIdx = cur + 2; // first element in the array
                while (childIdx < arrayEnd)
                {
                    childIdx = ParseGraphNode(scene, json, tokens, childIdx);
                }
            }
            else if (valueToken->type == JSMN_OBJECT)
            {
                 ParseGraphNode(scene, json, tokens, cur + 1);
            }
        }

        // Advance past the key token, then skip the value
        cur = cgltf_skip_json(tokens, cur + 1);
    }
    
    // Create Light if type matches
    if (type == "DirectionalLight" || type == "SpotLight")
    {
        LightConstants light = {};
        light.color = { color.x, color.y, color.z, 1.0f };
        light.intensity = intensity;
        
        XMVECTOR dir = XMVector3Normalize(XMLoadFloat3(&direction));
        XMStoreFloat4(&light.direction, dir);
        
        light.position = { translation.x, translation.y, translation.z, 1.0f };
        
        if (type == "DirectionalLight")
        {
            // Directional light viewProj build on Application side 
            // XMVECTOR lightPos = XMVectorScale(dir, -50.0f);
            //  XMMATRIX lightView = XMMatrixLookToLH(lightPos, dir, XMVectorSet(0, 1, 0, 0));
            //  XMMATRIX lightProj = XMMatrixOrthographicLH(100.0f, 100.0f, 0.1f, 200.0f);
            //  XMStoreFloat4x4(&light.viewProj, XMMatrixTranspose(lightView * lightProj));
        }
        else // SpotLight
        {
             float outerRad = XMConvertToRadians(outerAngle);
             float innerRad = XMConvertToRadians(innerAngle);
             light.direction.w = cos(outerRad);
             light.position.w = 1.0f;
             
             float cosInner = cos(innerRad);
             memcpy(&light.padding[0], &cosInner, sizeof(float)); 
             
             XMMATRIX lightView = XMMatrixLookAtRH(XMLoadFloat4(&light.position), XMLoadFloat4(&light.position) + dir, XMVectorSet(0, 1, 0, 0));
             XMMATRIX lightProj = XMMatrixPerspectiveFovRH(XMConvertToRadians(outerAngle * 2.0f), 1.0f, 0.1f, 100.0f);
             XMStoreFloat4x4(&light.viewProj, XMMatrixTranspose(lightView * lightProj));
        }
        
        scene->m_Lights.push_back(light);
    }
    
    return cur;
}
