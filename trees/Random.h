#ifndef RANDOM_H
#define RANDOM_H
#include <random>

/** Return random float between 0.0 and 1.0 */
inline float randomFloat() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

#endif // RANDOM_H
