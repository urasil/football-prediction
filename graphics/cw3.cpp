//#define SOLUTION_LIGHT
//#define SOLUTION_BOUNCE
//#define SOLUTION_THROUGHPUT
//#define SOLUTION_AA
//#define SOLUTION_MB
//#define SOLUTION_VR


precision highp float;

#define M_PI 3.14159265359

struct Material {
	#ifdef SOLUTION_LIGHT
    vec3 emission;
	#endif
	vec3 diffuse;
	vec3 specular;
	float glossiness;
};

struct Sphere {
	vec3 position;
#ifdef SOLUTION_MB
#endif
	float radius;
	Material material;
};

struct Plane {
	vec3 normal;
	float d;
	Material material;
};

const int sphereCount = 4;
const int planeCount = 4;
const int emittingSphereCount = 2;
#ifdef SOLUTION_BOUNCE
#else
const int maxPathLength = 1;
#endif 

struct Scene {
	Sphere[sphereCount] spheres;
	Plane[planeCount] planes;
};

struct Ray {
	vec3 origin;
	vec3 direction;
};

// Contains all information pertaining to a ray/object intersection
struct HitInfo {
	bool hit;
	float t;
	vec3 position;
	vec3 normal;
	Material material;
};

// Contains info to sample a direction and this directions probability
struct DirectionSample {
	vec3 direction;
	float probability;
};

HitInfo getEmptyHit() {
	Material emptyMaterial;
	#ifdef SOLUTION_LIGHT
	#endif
	emptyMaterial.diffuse = vec3(0.0);
	emptyMaterial.specular = vec3(0.0);
	emptyMaterial.glossiness = 1.0;
	return HitInfo(false, 0.0, vec3(0.0), vec3(0.0), emptyMaterial);
}

// Sorts the two t values such that t1 is smaller than t2
void sortT(inout float t1, inout float t2) {
	// Make t1 the smaller t
	if(t2 < t1)  {
		float temp = t1;
		t1 = t2;
		t2 = temp;
	}
}

// Tests if t is in an interval
bool isTInInterval(const float t, const float tMin, const float tMax) {
	return t > tMin && t < tMax;
}

// Get the smallest t in an interval
bool getSmallestTInInterval(float t0, float t1, const float tMin, const float tMax, inout float smallestTInInterval) {

	sortT(t0, t1);

	// As t0 is smaller, test this first
	if(isTInInterval(t0, tMin, tMax)) {
		smallestTInInterval = t0;
		return true;
	}

	// If t0 was not in the interval, still t1 could be
	if(isTInInterval(t1, tMin, tMax)) {
		smallestTInInterval = t1;
		return true;
	}

	// None was
	return false;
}

// Converts a random integer in 15 bits to a float in (0, 1)
float randomIntegerToRandomFloat(int i) {
	return float(i) / 32768.0;
}

// Returns a random integer for every pixel and dimension that remains the same in all iterations
int pixelIntegerSeed(const int dimensionIndex) {
	vec3 p = vec3(gl_FragCoord.xy, dimensionIndex);
	vec3 r = vec3(23.14069263277926, 2.665144142690225,7.358926345 );
	return int(32768.0 * fract(cos(dot(p,r)) * 123456.0));
}

// Returns a random float for every pixel that remains the same in all iterations
float pixelSeed(const int dimensionIndex) {
	return randomIntegerToRandomFloat(pixelIntegerSeed(dimensionIndex));
}

// The global random seed of this iteration
// It will be set to a new random value in each step
uniform int globalSeed;
int randomSeed;
void initRandomSequence() {
	randomSeed = globalSeed + pixelIntegerSeed(0);
}

// Computes integer  x modulo y not available in most WEBGL SL implementations
int mod(const int x, const int y) {
	return int(float(x) - floor(float(x) / float(y)) * float(y));
}

// Returns the next integer in a pseudo-random sequence
int rand() {
	randomSeed = randomSeed * 1103515245 + 12345;
	return mod(randomSeed / 65536, 32768);
}

float uniformRandomImproved(vec2 co){
    float a = 12.9898;
    float b = 78.233;
    float c = 43758.5453;
    float dt= dot(co.xy ,vec2(a,b));
    float sn= mod(dt,3.14);
    return fract(sin(sn) * c);
}

// Returns the next float in this pixels pseudo-random sequence
float uniformRandom() {
	return randomIntegerToRandomFloat(rand());
}

// This is the index of the sample controlled by the framework.
// It increments by one in every call of this shader
uniform int baseSampleIndex;

// Returns a well-distributed number in (0,1) for the dimension dimensionIndex
float sample(const int dimensionIndex) {
	// combining 2 PRNGs to avoid the patterns in the C-standard LCG
	return uniformRandomImproved(vec2(uniformRandom(), uniformRandom()));
}

// This is a helper function to sample two-dimensionaly in dimension dimensionIndex
vec2 sample2(const int dimensionIndex) {
	return vec2(sample(dimensionIndex + 0), sample(dimensionIndex + 1));
}

vec3 sample3(const int dimensionIndex) {
	return vec3(sample(dimensionIndex + 0), sample(dimensionIndex + 1), sample(dimensionIndex + 2));
}

// This is a register of all dimensions that we will want to sample.
// Thanks to Iliyan Georgiev from Solid Angle for explaining proper housekeeping of sample dimensions in ranomdized Quasi-Monte Carlo
//
// There are infinitely many path sampling dimensions.
// These start at PATH_SAMPLE_DIMENSION.
// The 2D sample pair for vertex i is at PATH_SAMPLE_DIMENSION + PATH_SAMPLE_DIMENSION_MULTIPLIER * i + 0
#define ANTI_ALIAS_SAMPLE_DIMENSION 0
#define TIME_SAMPLE_DIMENSION 1
#define PATH_SAMPLE_DIMENSION 3

// This is 2 for two dimensions and 2 as we use it for two purposese: NEE and path connection
#define PATH_SAMPLE_DIMENSION_MULTIPLIER (2 * 2)

vec3 getEmission(const Material material) {
	#ifdef SOLUTION_LIGHT
    return material.emission;
	#else
	// This is wrong. It just returns the diffuse color so that you see something to be sure it is working.
	return material.diffuse;
	#endif
}

vec3 getReflectance(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	#else
	return vec3(1.0);
	#endif
}

vec3 getGeometricTerm(const Material material, const vec3 normal, const vec3 inDirection, const vec3 outDirection) {
	#ifdef SOLUTION_THROUGHPUT
	#else
	return vec3(1.0);
	#endif
}

vec3 sphericalToEuclidean(float theta, float phi) {
	float x = sin(theta) * cos(phi);
	float y = sin(theta) * sin(phi);
	float z = cos(theta);
	return vec3(x, y, z);	
}

vec3 getRandomDirection(const int dimensionIndex) {
	#ifdef SOLUTION_BOUNCE
    // TEST FLAG
    bool test = true;
    float sample0, sample1 = sample2(dimensionIndex);

    float theta = acos(2.0 sample0 - 1.0)
    float phi = 2.0 * M_PI * sample1;
    vec3 direction = sphericalToEuclidean(theta, phi);

    if(test){
        if (abs(length(direction) - 1.0) < 1e-6) gl_FragColor = vec3(1,0,0);
        else gl_FragColor = vec3(0,1,0);
    }

    return direction;

	#else
	// Put your code to compute a random direction in 3D in the #ifdef above
	return vec3(0);
	#endif
}

HitInfo intersectSphere(const Ray ray, Sphere sphere, const float tMin, const float tMax) {

#ifdef SOLUTION_MB
#endif
	
	vec3 to_sphere = ray.origin - sphere.position;

	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(ray.direction, to_sphere);
	float c = dot(to_sphere, to_sphere) - sphere.radius * sphere.radius;
	float D = b * b - 4.0 * a * c;
	if (D > 0.0)
	{
		float t0 = (-b - sqrt(D)) / (2.0 * a);
		float t1 = (-b + sqrt(D)) / (2.0 * a);

		float smallestTInInterval;
		if(!getSmallestTInInterval(t0, t1, tMin, tMax, smallestTInInterval)) {
			return getEmptyHit();
		}

		vec3 hitPosition = ray.origin + smallestTInInterval * ray.direction;

		vec3 normal =
			length(ray.origin - sphere.position) < sphere.radius + 0.001?
			-normalize(hitPosition - sphere.position) :
		normalize(hitPosition - sphere.position);

		return HitInfo(
			true,
			smallestTInInterval,
			hitPosition,
			normal,
			sphere.material);
	}
	return getEmptyHit();
}

HitInfo intersectPlane(Ray ray, Plane plane) {
	float t = -(dot(ray.origin, plane.normal) + plane.d) / dot(ray.direction, plane.normal);
	vec3 hitPosition = ray.origin + t * ray.direction;
	return HitInfo(
		true,
		t,
		hitPosition,
		normalize(plane.normal),
		plane.material);
	return getEmptyHit();
}

float lengthSquared(const vec3 x) {
	return dot(x, x);
}

HitInfo intersectScene(Scene scene, Ray ray, const float tMin, const float tMax)
{
	HitInfo best_hit_info;
	best_hit_info.t = tMax;
	best_hit_info.hit = false;

	for (int i = 0; i < sphereCount; ++i) {
		Sphere sphere = scene.spheres[i];
		HitInfo hit_info = intersectSphere(ray, sphere, tMin, tMax);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	for (int i = 0; i < planeCount; ++i) {
		Plane plane = scene.planes[i];
		HitInfo hit_info = intersectPlane(ray, plane);

		if(	hit_info.hit &&
		   hit_info.t < best_hit_info.t &&
		   hit_info.t > tMin)
		{
			best_hit_info = hit_info;
		}
	}

	return best_hit_info;
}

mat3 transpose(mat3 m) {
	return mat3(
		m[0][0], m[1][0], m[2][0],
		m[0][1], m[1][1], m[2][1],
		m[0][2], m[1][2], m[2][2]
	);
}

// This function creates a matrix to transform from global space into a local space oriented around the provided vector.
mat3 makeLocalFrame(const vec3 vector) {
	#ifdef SOLUTION_VR
	#else
	return mat3(1.0);
	#endif
}

float luminance(const vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

#define EPSILON (1e-6)  // for avoiding numeric issues caused by floating point accuracy

DirectionSample sampleDirection(const vec3 normal, const vec3 inDirection, const vec3 diffuse, const vec3 specular, const float n, const int dimensionIndex) {
	DirectionSample result;
		
	#ifdef SOLUTION_VR
	#else
	// Depending on the technique: put your variance reduction code in the #ifdef above 
	result.direction = getRandomDirection(dimensionIndex);	
	result.probability = 1.0 / (4.0 * M_PI);
	#endif
	return result;
}

vec3 samplePath(const Scene scene, const Ray initialRay) {

	// Initial result is black
	vec3 result = vec3(0);

	Ray incomingRay = initialRay;
	vec3 throughput = vec3(1.0);
	for(int i = 0; i < maxPathLength; i++) {
		HitInfo hitInfo = intersectScene(scene, incomingRay, 0.001, 10000.0);

		if(!hitInfo.hit) return result;

		result += throughput * getEmission(hitInfo.material);

		Ray outgoingRay;
		DirectionSample directionSample;
		#ifdef SOLUTION_BOUNCE
        vec3 newDirection = randomDirection(PATH_SAMPLE_DIMENSION + 2 * i);
        Ray outgoingRay;
        outgoingRay.origin = hitInfo.position; 
        outgoingRay.direction = newDirection;
		#else
			// Put your code to compute the next ray in the #ifdef above
		#endif

		#ifdef SOLUTION_THROUGHPUT
		#else
		// Compute the proper throughput in the #ifdef above 
		throughput *= 0.1;
		#endif

		// div by probability of sampled direction 
		throughput /= directionSample.probability; 
	
		#ifdef SOLUTION_BOUNCE
        incomingRay = outgoingRay;
		#else
		// Put some handling of the next and the current ray in the #ifdef above
		#endif
	}
	return result;
}

uniform ivec2 resolution;
Ray getFragCoordRay(const vec2 fragCoord) {

	float sensorDistance = 1.0;
	vec3 origin = vec3(0, 0, sensorDistance);
	vec2 sensorMin = vec2(-1, -0.5);
	vec2 sensorMax = vec2(1, 0.5);
	vec2 pixelSize = (sensorMax - sensorMin) / vec2(resolution);
	vec3 direction = normalize(vec3(sensorMin + pixelSize * fragCoord, -sensorDistance));

	float apertureSize = 0.0;
	float focalPlane = 100.0;
	vec3 sensorPosition = origin + focalPlane * direction;
	origin.xy += -vec2(0.5);
	direction = normalize(sensorPosition - origin);

	return Ray(origin, direction);
}

vec3 colorForFragment(const Scene scene, const vec2 fragCoord) {
	initRandomSequence();

	#ifdef SOLUTION_AA
	#else  	
	// Put your anti-aliasing code in the #ifdef above
	vec2 sampleCoord = fragCoord;
	#endif
	return samplePath(scene, getFragCoordRay(sampleCoord));
}

void loadScene1(inout Scene scene) {

	scene.spheres[0].position = vec3(7, -2, -12);
	scene.spheres[0].radius = 2.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
    scene.spheres[0].material.emission = 15.0 * vec3(0.9, 0.9, 0.5);
#endif
	scene.spheres[0].material.diffuse = vec3(0.5);
	scene.spheres[0].material.specular = vec3(0.5);
	scene.spheres[0].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
#endif
	
	scene.spheres[1].position = vec3(-8, 4, -13);
	scene.spheres[1].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
    scene.spheres[1].material.emission = 15.0 * vec3(0.8, 0.3, 0.1);
#endif
	scene.spheres[1].material.diffuse = vec3(0.5);
	scene.spheres[1].material.specular = vec3(0.5);
	scene.spheres[1].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
#endif
	
	scene.spheres[2].position = vec3(-2, -2, -12);
	scene.spheres[2].radius = 3.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
#endif  
	scene.spheres[2].material.diffuse = vec3(0.2, 0.5, 0.8);
	scene.spheres[2].material.specular = vec3(0.8);
	scene.spheres[2].material.glossiness = 40.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
#endif
	
	scene.spheres[3].position = vec3(3, -3.5, -14);
	scene.spheres[3].radius = 1.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT  
#endif  
	scene.spheres[3].material.diffuse = vec3(0.9, 0.8, 0.8);
	scene.spheres[3].material.specular = vec3(1.0);
	scene.spheres[3].material.glossiness = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_MB
#endif
	
	scene.planes[0].normal = vec3(0, 1, 0);
	scene.planes[0].d = 4.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
#endif
	scene.planes[0].material.diffuse = vec3(0.8);
	scene.planes[0].material.specular = vec3(0.0);
	scene.planes[0].material.glossiness = 50.0;    

	scene.planes[1].normal = vec3(0, 0, 1);
	scene.planes[1].d = 18.5;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
#endif
	scene.planes[1].material.diffuse = vec3(0.9, 0.6, 0.3);
	scene.planes[1].material.specular = vec3(0.02);
	scene.planes[1].material.glossiness = 3000.0;

	scene.planes[2].normal = vec3(1, 0,0);
	scene.planes[2].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
#endif
	
	scene.planes[2].material.diffuse = vec3(0.2);
	scene.planes[2].material.specular = vec3(0.1);
	scene.planes[2].material.glossiness = 100.0; 

	scene.planes[3].normal = vec3(-1, 0,0);
	scene.planes[3].d = 10.0;
	// Set the value of the missing property in the ifdef below 
#ifdef SOLUTION_LIGHT    
#endif
	
	scene.planes[3].material.diffuse = vec3(0.2);
	scene.planes[3].material.specular = vec3(0.1);
	scene.planes[3].material.glossiness = 100.0; 
}


void main() {
	// Setup scene
	Scene scene;
	loadScene1(scene);

	// compute color for fragment
	gl_FragColor.rgb = colorForFragment(scene, gl_FragCoord.xy);
	gl_FragColor.a = 1.0;
}