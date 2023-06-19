#pragma once

#include "device_launch_parameters.h"
#include "scene.cuh"
#include "curand_kernel.h"
#include <ctime>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

__global__ void initFramebuffer(ViewRender* view) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		for (size_t i = 0; i < view->pixelCount; i++) {
			view->frameBuffer[i] = 0.f;
		}
	}
}

__global__ void renderScene(Scene* scene, ViewRender* view) {
	//Get the pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int sample = blockIdx.z * blockDim.z + threadIdx.z;
	if ((x >= view->width) || (y >= view->height) || sample >= view->samples) return;
	int pixelIndex = (y * view->width + x) * CHANNELS;

	//Create a random state
	curandState randomState;
	curand_init(23984, pixelIndex, 0, &randomState);

	//The sun
	float sunStrength = 14;
	float size = 0.05f;
	float3 sunColor = make_float3(sunStrength);
	float3 sunDir = normalize(make_float3(-1, .5f, .5));
	float sharpness = 2.f;

	float3 finalColor = make_float3(0, 0, 0);
	for (int sample = 0; sample < view->samples; sample++) {

		//What color should be the starting point?
		float3 sampleColor = make_float3(1);
		//Cast our ray
		Ray ray = scene->camera.getRay((x + curand_uniform(&randomState) * 2 - 0.5f) / view->width, (y + curand_uniform(&randomState) * 2 - 0.5f) / view->height, static_cast<float>(view->height) / view->width);
		Hit hit;
		int bounce = 0;
		for (; bounce < view->maxBounces; bounce++) {
			hit = scene->raytrace(ray, hit.actor);
			if (!hit.hit) {
				//the world background
				float sunAngle = clamp((powf(dot(ray.getDirection(), sunDir), sharpness) - (1 - size)) * (1 / size), 0.f, 1.f);
				sampleColor *= (1 - sunAngle) * scene->background.emissiveColor + sunAngle * sunColor;
				break;
			}
			hit.actor->material.color(hit, ray, sampleColor, &randomState);
			if (hit.actor->material.emissive) break;
		}
		//Failed to hit a light source
		if (bounce >= view->maxBounces) sampleColor *= 0;

	sampleColor /= view->samples;

	//Set the final color
	//finalColor = make_float3((float) x / view->width, (float) y / view->height, 0);
	view->frameBuffer[pixelIndex + 0] += sampleColor.x;
	view->frameBuffer[pixelIndex + 1] += sampleColor.y;
	view->frameBuffer[pixelIndex + 2] += sampleColor.z;
}

int main() {

	std::clock_t startTime = std::clock();

	ViewRender* d_view;
	checkCudaErrors(cudaMallocManaged((void**)&d_view, sizeof(ViewRender)));
	//Render settings!
	new (d_view) ViewRender(1920, 1080, 16, 16); //be carefull with placement new....
	d_view->maxBounces = 4;
	d_view->samples = 40;

	Camera camera = Camera({ 0,0,0.3f }, { 1,1,0.2f });

	Sphere lamp = Sphere({ -3, 3, 1 }, Material(make_float3(20, .5f, .5f)), .5f);
	Sphere earth = Sphere({ 0, 0, -2000.f }, Material({ .2f, .9f, .2f }, 1.f), 2000.f);
	Sphere a = Sphere({ 1, 1, 0.2f }, Material({ .8f, .2f, .2f }, 1.f), 0.2f);
	Sphere b = Sphere({ 1, 1.4f, 0.2f }, Material({ .2f, .2f, .8f }, 1.f), 0.2f);
	Sphere c = Sphere({ 5, 4, 1 }, Material({ 1.f, 1.f, 1.f }, 1.f), 1.f);
	Sphere mirror = Sphere({ 2.4f, 2, 0.6f }, Material({ 1.f, 1.f, 1.f }, 0.f), .2f);

	//Background Material
	Material background(make_float3(0.6f, 0.8f, 1.0f) * 0.8f);
	//Material background(make_float3(0));

	Scene* d_scene;
	checkCudaErrors(cudaMallocManaged((void**)&d_scene, sizeof(Scene)));
	//Scene setup
	new (d_scene) Scene({ earth, a,b,c, mirror, lamp }, camera, background);

	std::cout << "Samples: " << d_view->samples << " Max Bounces: " << d_view->maxBounces << " Resolution: " << d_view->width << "x" << d_view->height << std::endl;
	std::cout << "Scene Setup time: " << (std::clock() - startTime) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	startTime = std::clock();

	dim3 blocks(d_view->width / d_view->tileSizeX + 1, d_view->height / d_view->tileSizeY + 1, d_view->samples);
	dim3 threads(d_view->tileSizeX, d_view->tileSizeY, 1);

	initFramebuffer << <1, 1 >> > (d_view);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	renderScene << <blocks, threads >> > (d_scene, d_view);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "Render time: " << (std::clock() - startTime) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	startTime = std::clock();

	//Convert float based rgb in frame buffer to only 8bit rgb for the out image
	unsigned char* outPixels = (unsigned char*)malloc(d_view->pixelCount);
	for (size_t i = 0; i < d_view->pixelCount; i++) {
		outPixels[i] = int(clamp(d_view->frameBuffer[i], 0.f, 1.f) * 255);
	}

	stbi_write_jpg("out.jpg", d_view->width, d_view->height, CHANNELS, outPixels, 100);

	std::cout << "Image output time: " << (std::clock() - startTime) / (double)CLOCKS_PER_SEC << "s" << std::endl;

	//Cleanup
	free(outPixels);
	d_view->~ViewRender();
	checkCudaErrors(cudaFree(d_view));
	d_scene->~Scene();
	checkCudaErrors(cudaFree(d_scene));
}