#pragma once

#include "scene.cuh"
#include <iostream>

__device__ float3 randomInUnitSphere(curandState* randomState) {
	float3 random;
	do {
		random = make_float3(curand_uniform(randomState), curand_uniform(randomState), curand_uniform(randomState));
		random = 2 * (random - 0.5f);
	} while (random.x * random.x + random.y * random.y + random.z * random.z > 1); //TODO test performance of this
	return random;
}

__device__ Hit Scene::raytrace(const Ray& ray, const Sphere* ignore) const {
	Hit hit;
	Hit currentHit;
	for (int i = 0; i < objectsSize; i++) {
		if (d_objects[i].intersect(ray, currentHit) && currentHit.distance < hit.distance && currentHit.actor != ignore) {
			hit = currentHit;
		}
	}
	return hit;
}

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	{
		if (result) {
			std::cerr << "CUDA error: " << static_cast<unsigned int>(result) << " " << cudaGetErrorName(result) << ": " << cudaGetErrorString(result) << " at " <<
				file << ":" << line << " '" << func << "' \n";
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(99);
		}
	}
}