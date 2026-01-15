#pragma once

#include <SDL.h>
#include <memory>

class Application
{
public:
    Application();
    ~Application();

    void Run();

private:
    void Initialize();
    void Shutdown();
    void ProcessEvents();
    void Update(float deltaTime);
    void Render();

    bool m_IsRunning;
    SDL_Window* m_Window;
    SDL_Renderer* m_Renderer;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};