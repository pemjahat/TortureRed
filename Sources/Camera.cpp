#include "Camera.h"
#include <algorithm>

// Camera implementation
Camera::Camera()
    : m_Position(0.0f, 0.0f, -10.0f)
    , m_Yaw(0.0f)
    , m_Pitch(0.0f)
    , m_LookDirection(0.0f, 0.0f, 1.0f)
    , m_UpDirection(0.0f, 1.0f, 0.0f)
    , m_MoveSpeed(15.0f)
    , m_RotationSpeed(0.0005f)
    , m_ZoomSpeed(2.0f)
    , m_CameraModeActive(false)
    , m_FovY(XM_PI / 3.0f)  // 60 degrees default
    , m_AspectRatio(16.0f / 9.0f)
    , m_NearZ(0.1f)
    , m_FarZ(1000.0f)
{
    m_Keys[0] = m_Keys[1] = m_Keys[2] = m_Keys[3] = false;
}

void Camera::Update(float deltaTime)
{
    // Handle keyboard movement (always available for first-person movement)
    XMVECTOR forward = GetForwardVector();
    XMVECTOR right = GetRightVector();

    XMVECTOR movement = XMVectorZero();

    if (m_Keys[0]) movement += forward; // W - forward
    if (m_Keys[1]) movement -= forward; // S - backward
    if (m_Keys[2]) movement -= right;   // A - left
    if (m_Keys[3]) movement += right;   // D - right

    if (XMVector3LengthSq(movement).m128_f32[0] > 0.0f)
    {
        movement = XMVector3Normalize(movement);
        movement *= m_MoveSpeed * deltaTime;

        m_Position.x += movement.m128_f32[0];
        m_Position.y += movement.m128_f32[1];
        m_Position.z += movement.m128_f32[2];
    }
}

void Camera::ProcessMouseMovement(float deltaX, float deltaY)
{
    if (!m_CameraModeActive) return;

    m_Yaw += deltaX * m_RotationSpeed;
    m_Pitch += deltaY * m_RotationSpeed;

    // Clamp pitch to avoid gimbal lock
    m_Pitch = std::max(-XM_PIDIV2 + 0.01f, std::min(XM_PIDIV2 - 0.01f, m_Pitch));

    // Update look direction based on yaw and pitch (up remains world up)
    XMMATRIX rotationY = XMMatrixRotationY(m_Yaw);
    XMMATRIX rotationX = XMMatrixRotationX(m_Pitch);
    XMMATRIX rotation = XMMatrixMultiply(rotationX, rotationY);

    XMVECTOR look = XMVector3TransformNormal(XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f), rotation);

    XMStoreFloat3(&m_LookDirection, look);
    // m_UpDirection remains (0,1,0) for FPS-style camera
}

void Camera::ProcessMouseWheel(float deltaWheel)
{
    XMVECTOR forward = GetForwardVector();
    forward *= deltaWheel * m_ZoomSpeed;

    m_Position.x += forward.m128_f32[0];
    m_Position.y += forward.m128_f32[1];
    m_Position.z += forward.m128_f32[2];
}

void Camera::ProcessKeyboard(bool w, bool s, bool a, bool d)
{
    m_Keys[0] = w; // W
    m_Keys[1] = s; // S
    m_Keys[2] = a; // A
    m_Keys[3] = d; // D
}

DirectX::XMMATRIX Camera::GetViewMatrix() const
{
    // Use XMMatrixLookToLH like SimpleCamera
    return XMMatrixLookToLH(XMLoadFloat3(&m_Position), XMLoadFloat3(&m_LookDirection), XMLoadFloat3(&m_UpDirection));
}

DirectX::XMVECTOR Camera::GetForwardVector() const
{
    return XMLoadFloat3(&m_LookDirection);
}

DirectX::XMVECTOR Camera::GetRightVector() const
{
    // Right vector is cross product of look and up directions
    XMVECTOR look = XMLoadFloat3(&m_LookDirection);
    XMVECTOR up = XMLoadFloat3(&m_UpDirection);
    return XMVector3Cross(look, up);
}

void Camera::SetProjectionParameters(float fovY, float aspectRatio, float nearZ, float farZ)
{
    m_FovY = fovY;
    m_AspectRatio = aspectRatio;
    m_NearZ = nearZ;
    m_FarZ = farZ;
}

DirectX::XMMATRIX Camera::GetProjMatrix() const
{
    return XMMatrixPerspectiveFovLH(m_FovY, m_AspectRatio, m_NearZ, m_FarZ);
}

DirectX::XMMATRIX Camera::GetInvViewMatrix() const
{
    XMMATRIX view = GetViewMatrix();
    return XMMatrixInverse(nullptr, view);
}