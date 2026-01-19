#pragma once

#include <DirectXMath.h>
using namespace DirectX;

// Camera class for 3D navigation
class Camera
{
public:
    Camera();
    ~Camera() = default;

    void Update(float deltaTime);
    void ProcessMouseMovement(float deltaX, float deltaY);
    void ProcessMouseWheel(float deltaWheel);
    void ProcessKeyboard(bool w, bool s, bool a, bool d);

    // Get view matrix
    DirectX::XMMATRIX GetViewMatrix() const;
    DirectX::XMFLOAT3 GetPosition() const { return m_Position; }

    // Camera control state
    void SetCameraMode(bool active) { m_CameraModeActive = active; }
    bool IsCameraModeActive() const { return m_CameraModeActive; }

private:
    // Camera position and orientation
    DirectX::XMFLOAT3 m_Position;
    float m_Yaw;      // Rotation around Y axis (left/right)
    float m_Pitch;    // Rotation around X axis (up/down)
    DirectX::XMFLOAT3 m_LookDirection;
    DirectX::XMFLOAT3 m_UpDirection;

    // Movement speeds
    float m_MoveSpeed;
    float m_RotationSpeed;
    float m_ZoomSpeed;

    // Camera mode state
    bool m_CameraModeActive;
    bool m_Keys[4]; // W, S, A, D

    // Update vectors based on current rotation
    DirectX::XMVECTOR GetForwardVector() const;
    DirectX::XMVECTOR GetRightVector() const;
};