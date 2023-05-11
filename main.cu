#pragma once

#include "device_launch_parameters.h"
#include "scene.cuh"
#include "curand_kernel.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void renderScene(Scene* scene, ViewRender* view) {
	//Get the pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x >= view->width) || (y >= view->height)) return;
	int pixelIndex = (y * view->width + x) * CHANNELS;

	//Create a random state
	curandState randomState;
	curand_init(23984, pixelIndex, 0, &randomState);

	float3 finalColor = scene->background.emissiveColor;
	//Cast our ray
	Ray ray = scene->camera.getRay(static_cast<float>(x) / view->width, static_cast<float>(y) / view->height, static_cast<float>(view->height) / view->width);
	Hit hit;
	for (int bounce = 0; bounce < view->maxBounces; bounce++) {
		hit = scene->raytrace(ray, hit.actor);
		if (!hit.hit) break;
		bool continueSampling = hit.actor->material.color(hit, ray, finalColor, &randomState);

		Ray sampleRay(ray);
		for (int sample = 0; sample < view->samples && continueSampling; sample++) {
			Hit sampleHit = scene->raytrace(sampleRay, hit.actor);
			if (!sampleHit.hit) continue;
			sampleHit.actor->material.color(sampleHit, sampleRay, finalColor, &randomState);
		}
	}

	//Set the final color
	//finalColor = make_float3((float) x / view->width, (float) y / view->height, 0);
	view->frameBuffer[pixelIndex + 0] = clamp(finalColor.x, 0.f, 1.f);
	view->frameBuffer[pixelIndex + 1] = clamp(finalColor.y, 0.f, 1.f);
	view->frameBuffer[pixelIndex + 2] = clamp(finalColor.z, 0.f, 1.f);
}

__global__ void initScene(Scene* scene, ViewRender* view) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {


	}
}

int main() {

	ViewRender* d_view;
	checkCudaErrors(cudaMallocManaged((void**)&d_view, sizeof(ViewRender)));
	//Render settings!
	new (d_view) ViewRender(1920, 1080, 16, 16); //be carefull with placement new....
	d_view->maxBounces = 2;
	d_view->samples = 1000;

	Camera camera = Camera({ 0,0,1.f }, { 1,1,1 });

	Sphere sun = Sphere({ -10, 5, 10 }, Material({ 10,10,10 }), 2.f);
	Sphere earth = Sphere({ 0, 0, -2000.f }, Material({ .2f, .9f, .2f }, 1.f), 2000.f);
	Sphere a = Sphere({ 1,1,1 }, Material({ .8f, .2f, .2f }, 1.f), 0.2f);
	Sphere b = Sphere({ 1, 1.4f, 1 }, Material({ .2f, .2f, .8f }, 1.f), 0.2f);
	Sphere c = Sphere({ 2,3,1 }, Material({ 1.f, 1.f, 1.f }, 1.f), 1.f);

	//Background Material
	Material background(make_float3(0.3f, 0.3f, .6f) * 2.f);

	Scene* d_scene;
	checkCudaErrors(cudaMallocManaged((void**)&d_scene, sizeof(Scene)));
	//Scene setup
	new (d_scene) Scene({ sun, earth, a,b,c }, camera, background);

	dim3 blocks(d_view->width / d_view->tileSizeX + 1, d_view->height / d_view->tileSizeY + 1);
	dim3 threads(d_view->tileSizeX, d_view->tileSizeY);

	renderScene << <blocks, threads >> > (d_scene, d_view);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//Convert float based rgb in frame buffer to only 8bit rgb for the out image
	size_t outPixelsSize = d_view->width * d_view->height * CHANNELS;

	unsigned char* outPixels = (unsigned char*)malloc(outPixelsSize);
	for (size_t i = 0; i < outPixelsSize; i++) {
		outPixels[i] = int(d_view->frameBuffer[i] * 255);
	}

	stbi_write_jpg("out.jpg", d_view->width, d_view->height, CHANNELS, outPixels, 100);


	//Cleanup
	free(outPixels);
	d_view->~ViewRender();
	checkCudaErrors(cudaFree(d_view));
	d_scene->~Scene();
	checkCudaErrors(cudaFree(d_scene));
}