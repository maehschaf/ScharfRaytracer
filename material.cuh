#pragma once
#include <vector_types.h>
#include "curand_kernel.h"
#include "scene.cuh"


__device__ float3 randomInUnitSphere(curandState* randomState);

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
			ray.setDirection(hit.normal + randomInUnitSphere(randomState));
			return true;
		}
		else {
			ray.setDirection(ray.getDirection() - 2 * dot(ray.getDirection(), hit.normal) * hit.normal); // Reflection vector
			return false;
		}
	}
};