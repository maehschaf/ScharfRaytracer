#pragma once

#include "tracing.cuh"

struct Camera {

	float3 position;
	float fov = 90.f;
	float focalLength = 1.f;

protected:
	float3 forward; //points forward
	float3 vertical; //points upwards from the view of the camera
	float3 horizontal; //Points to the right horizontally
	float xfov;

public:
	__host__ Camera(const float3& position, const float3& target) : position(position) {
		forward = normalize(target - position);
		float3 fakeUp = make_float3(forward.x, forward.y, forward.z + 1);
		horizontal = normalize(cross(forward, fakeUp)); //Crappy way to get a horizontal vector.. not safe! it will flip!
		vertical = normalize(cross(horizontal, forward)); //Normalize might be unneccesarry..
		xfov = sinf(fov / 2.f);
	}

	//Get a ray going through the given screen position
	//x and y will be from 0 to 1 starting in the upper left edge
	//yMult is height/width
	__device__ Ray getRay(float x, float y, float yMult) const {
		//Calculate a spot on our "screen plane"
		float3 spot = forward * focalLength + position;
		spot += horizontal * xfov * (x - .5f) * focalLength;
		spot -= vertical * xfov * yMult * (y - .5f) * focalLength;
		return Ray(position, spot - position);
	}
};