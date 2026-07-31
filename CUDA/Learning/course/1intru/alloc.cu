#include <__clang_cuda_runtime_wrapper.h>
#include <cstdlib>
#define N 512

__global__ void add(int *a, int *b, int *c) {
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
}

void random_ints(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }
}

int main(void) {
    int *a, *b, *c;       // host copy
    int *d_a, *d_b, *d_c; // device copy
    int size = N * sizeof(int);

    // allocate space for device copies
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // allocate space for host copies
    a = (int *)malloc(size);
    random_ints(a, N);
    b = (int *)malloc(size);
    random_ints(b, N);
    c = (int *)malloc(size);
}
