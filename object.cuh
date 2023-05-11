#pragma once

#include "tracing.cuh"

struct Object {

	Transform transform;

	__host__ __device__ Object(const Transform& transform) : transform(transform) {

	}
};