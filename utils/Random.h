#ifndef RANDOM_H
#define RANDOM_H
#include <random>

/** Return random float between 0.0 and 1.0 */
inline float randomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

//https://stackoverflow.com/questions/39369424/generate-random-pastel-colour
inline float randomDarkColor() {
    return (85 * randomFloat() + 30) / 255;
}

#endif // RANDOM_H
