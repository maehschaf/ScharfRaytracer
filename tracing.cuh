#pragma once

#include <vector_types.h>
#include "helper_math.h"

struct Sphere;

struct Ray {
    __device__ Ray(float3 start, float3 direction) {
        this->origin = start;
        this->direction = normalize(direction);
    }

    __device__ const float3& getOrigin() const { return origin; }

    __device__ const float3& getDirection() const { return direction; }

    __device__ void setOrigin(const float3& origin) { this->origin = origin; }

    __device__ void setDirection(const float3& direction) { this->direction = normalize(direction); }

private:
    float3 origin;
    float3 direction;
};

struct Hit {
    //Whether we hit anything
    bool hit = false;
    //The Actor we hit
    const Sphere* actor = nullptr;
    //World space position that was hit
    float3 position = { 0, 0, 0 };
    //The normal of the collider at the hit position
    float3 normal = { 0, 0, 0 };
    //How far away is the object we hit
    float distance = 3.402823466e+38F;
};


struct Transform {
	float3 position = { 0,0,0 };
	float4 rotation = { 0,0,0,1 };
	float3 size = { 1,1,1 };
	float* matrix = nullptr;

	Transform() {

	}
};
