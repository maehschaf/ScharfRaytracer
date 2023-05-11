#pragma once

#include "camera.cuh"
#include <vector>
#include "sphere.cuh"


//Number of color channels
#define CHANNELS 3

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

/*
A representation of a scene viewed from a camera that copies its data to the GPU
*/
struct Scene {
	Camera camera;

	Sphere* d_objects = nullptr;
	size_t objectsSize = 0;

	Material background;

	__host__ Scene(std::vector<Sphere> objects, const Camera& camera, const Material& background) : camera(camera), background(background) {
		copyToDevice(objects);
	}

	__host__ void copyToDevice(std::vector<Sphere> objects) {
		objectsSize = objects.size();
		Sphere* data = objects.data();
		checkCudaErrors(cudaMalloc((void**)&d_objects, sizeof(*data) * objectsSize));
		checkCudaErrors(cudaMemcpy(d_objects, data, sizeof(*data) * objectsSize, cudaMemcpyHostToDevice));
	}

	__device__ Hit raytrace(const Ray& ray, const Sphere* ignore) const;

	__host__ ~Scene() {
		checkCudaErrors(cudaFree(d_objects));
	}
};

struct ViewRender
{
	int width = 1920;
	int height = 1080;
	int tileSizeX = 16;
	int tileSizeY = 16;
	int maxBounces = 3;
	int samples = 20;

	float* frameBuffer = nullptr;
	size_t frameBufferSize = 0;

	__host__ ViewRender(int width, int height, int tileSizeX, int tileSizeY) : width(width), height(height), tileSizeX(tileSizeX), tileSizeY(tileSizeY) {
		this->frameBufferSize = CHANNELS * height * width * sizeof(float);
		checkCudaErrors(cudaMallocManaged((void**)&frameBuffer, frameBufferSize));
	}

	__host__ ~ViewRender() {
		checkCudaErrors(cudaFree(frameBuffer));
	}
};