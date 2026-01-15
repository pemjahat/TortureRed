#include "Application.h"
#include <SDL.h>
#include <cassert>
#include <iostream>

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const char* WINDOW_TITLE = "TortureRed";

Application::Application()
    : m_IsRunning(false)
    , m_Window(nullptr)
    , m_Renderer(nullptr)
{
}

Application::~Application()
{
    Shutdown();
}

void Application::Run()
{
    Initialize();

    m_IsRunning = true;
    Uint32 lastTime = SDL_GetTicks();

    while (m_IsRunning)
    {
        Uint32 currentTime = SDL_GetTicks();
        float deltaTime = (currentTime - lastTime) / 1000.0f;
        lastTime = currentTime;

        ProcessEvents();
        Update(deltaTime);
        Render();

        // Cap frame rate
        SDL_Delay(16); // ~60 FPS
    }
}

void Application::Initialize()
{
    // Initialize SDL
    assert(SDL_Init(SDL_INIT_VIDEO) == 0 && "SDL_Init failed");

    // Create window
    m_Window = SDL_CreateWindow(
        WINDOW_TITLE,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );
    assert(m_Window != nullptr && "SDL_CreateWindow failed");

    // Create renderer
    m_Renderer = SDL_CreateRenderer(m_Window, -1, SDL_RENDERER_ACCELERATED);
    assert(m_Renderer != nullptr && "SDL_CreateRenderer failed");

    std::cout << "TortureRed application initialized successfully!" << std::endl;
}

void Application::Shutdown()
{
    if (m_Renderer)
    {
        SDL_DestroyRenderer(m_Renderer);
        m_Renderer = nullptr;
    }

    if (m_Window)
    {
        SDL_DestroyWindow(m_Window);
        m_Window = nullptr;
    }

    SDL_Quit();
    std::cout << "Application shutdown complete." << std::endl;
}

void Application::ProcessEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
        case SDL_QUIT:
            m_IsRunning = false;
            break;

        case SDL_KEYDOWN:
            if (event.key.keysym.sym == SDLK_ESCAPE)
            {
                m_IsRunning = false;
            }
            break;

        default:
            break;
        }
    }
}

void Application::Update(float deltaTime)
{
    // Update game logic here
    // For now, just a placeholder
}

void Application::Render()
{
    // Clear screen with dark blue
    SDL_SetRenderDrawColor(m_Renderer, 25, 25, 112, 255);
    SDL_RenderClear(m_Renderer);

    // Render game objects here
    // For now, draw a simple white rectangle
    SDL_Rect rect = { 100, 100, 200, 150 };
    SDL_SetRenderDrawColor(m_Renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(m_Renderer, &rect);

    // Present the rendered frame
    SDL_RenderPresent(m_Renderer);
}