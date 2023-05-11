#pragma once
#include <vector_types.h>
#include "curand_kernel.h"
#include "scene.cuh"

struct Material
{
	bool emissive = true;
	float3 emissiveColor = { 0,0,0 };
	float3 diffuseColor = { 0,0,0 };
	float roughness = 0;

	__host__ __device__ Material(float3 emissiveColor) : emissiveColor(emissiveColor) {}

	__host__ __device__ Material(float3 diffuseColor, float roughness) : diffuseColor(diffuseColor), roughness(roughness), emissive(false) {}

	__device__ bool color(const Hit& hit, Ray& ray, float3& color, curandState* randomState) const {
		if (emissive) {
			color *= emissiveColor;
			return false;
		}
		ray.setOrigin(hit.position);
		color *= diffuseColor;
		if (curand_uniform(randomState) < roughness) {
			float3 random = make_float3(curand_uniform(randomState), curand_uniform(randomState), curand_uniform(randomState));
			ray.setDirection(hit.normal + 2 * (random - 0.5f));
			return true;
		}
		else {
			ray.setDirection(ray.getDirection() - 2 * dot(ray.getDirection(), hit.normal) * hit.normal); // Reflection vector
			return false;
		}
	}
};