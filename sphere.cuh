#pragma once

#include "tracing.cuh"
#include "material.cuh"

struct Sphere {

	float3 position;
	Material material;
	float radius = 1.f;

	__host__ __device__ Sphere(const float3& position, const Material& material, float radius) : position(position), material(material), radius(radius) {

	}

    __device__ bool intersect(const Ray& ray, Hit& hit) const {
        hit.hit = false;

        //The position of the sphere
        const float3& spherePos = position;

        //Position of our sphere relative to the ray origin
        float3 originToPos = spherePos - ray.getOrigin();
        //(dot is used as lengthSquared here)
        float originToPosLengthSquared = dot(originToPos, originToPos);

        //The distance of the closest approach to the center of the sphere along ray
        float closestApproach = dot(originToPos, ray.getDirection());

        if (originToPosLengthSquared >= radius * radius) { // Ray origin is outside the sphere
            if (closestApproach < 0) { //Ray is pointing away from the sphere - we can never hit the sphere
                return false;
            }
        }

        //The distance from the closest approach to the center of the sphere to the surface of the sphere
        float halfChordDistanceSquared = (radius * radius) - originToPosLengthSquared + (closestApproach * closestApproach);

        if (halfChordDistanceSquared < 0) { //We miss the sphere - our closestApproach is outside the sphere, therefore it has a negative distance to the surface
            return false;
        }

        //If we get here we have hit the sphere!
        float distance = 0;
        if (originToPosLengthSquared >= radius * radius) { // Ray origin is outside the sphere
            distance = closestApproach - sqrt(halfChordDistanceSquared);
        }
        else { // Ray origin is inside the sphere
            distance = closestApproach + sqrt(halfChordDistanceSquared);
        }

        //we now have the intersection distance! - now we just need to fill in the hit information
        hit.hit = true;
        hit.actor = this;
        hit.distance = distance;
        //Direction is always normalized thanks tho Rays constructor
        hit.position = ray.getOrigin() + ray.getDirection() * distance;
        hit.normal = (hit.position - spherePos) / radius; //Simply the normal of a point on a sphere

        return true;
    }
};