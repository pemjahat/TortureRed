#pragma once

#include "GraphicsTypes.h"

class Scene
{
public:
    Scene();
    ~Scene();

    bool LoadScene(const std::string& path);

    std::string GetModelPath() const { return m_ModelPath; }
    std::vector<LightConstants>& GetLights() { return m_Lights; }

    std::vector<LightConstants> m_Lights;
    std::string m_ModelPath;

private:
    std::string m_JsonContent;
    
    // Helpers for JSON parsing
    bool ParseJson(const char* json, size_t length);
};
