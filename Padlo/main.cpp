#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

#define BASE 10
#define EXP 1

unsigned long long times10pow(unsigned long long arg, int exp) {
    if (exp == 0)
        return arg;
    else if (exp > 0)
        return times10pow((arg << 3) + (arg << 1), exp - 1);
}

int main(int argc, char ** argv)
{
    const int base = BASE;
    int exp = EXP;
    unsigned long long spacerCount = 0ull;
    int sqrt2[10] = { 1, 4, 1, 4, 2, 1, 3, 5, 6, 2 };
    int diminishBy[5] = { 2, 1, 2, 1, 1 };
    int dimIndex = 0;

    unsigned long long fullLength = pow(base, exp);

    printf("Length: %llu\n", fullLength);

    unsigned long long maxHeight = 0ull;
    for(int i = 0; i <= exp; i++) {
        maxHeight += times10pow(sqrt2[i], exp - i);
    }

    printf("Max Height: %llu\n", maxHeight);

    maxHeight++;
    while (fullLength > 0) {
        spacerCount += maxHeight;

        maxHeight -= diminishBy[dimIndex];

        fullLength--;
        if (++dimIndex > 4)
            dimIndex = 0;
    }

    printf("Remaining height: %llu\n", maxHeight);
    printf("Result: %llu\n", ++spacerCount);

    return 0;
}
