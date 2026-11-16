// Memory-bandwidth probe for memcalc's CPU device entries.
// Decode streams weights from memory, so the READ kernel is the relevant ceiling;
// COPY and TRIAD (STREAM-style, counting reads + writes) are reported for comparison.
// Build: gcc -O3 -march=native -fopenmp membw.c -o membw   Run: ./membw [MiB per array] [reps]
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static double now(void) { return omp_get_wtime(); }

int main(int argc, char **argv) {
    size_t mib = argc > 1 ? strtoul(argv[1], NULL, 10) : 1024;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    size_t n = mib * 1024 * 1024 / sizeof(double);
    double *a = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double)), *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

    double best_read = 1e30, best_copy = 1e30, best_triad = 1e30, sink = 0;
    for (int r = 0; r < reps; r++) {
        double t = now(), s = 0;
#pragma omp parallel for simd reduction(+ : s)
        for (size_t i = 0; i < n; i++) s += a[i];
        t = now() - t; sink += s; if (t < best_read) best_read = t;

        t = now();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) c[i] = a[i];
        t = now() - t; if (t < best_copy) best_copy = t;

        t = now();
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        t = now() - t; if (t < best_triad) best_triad = t;
    }
    double bytes = (double)n * sizeof(double);
    printf("threads=%d array=%zu MiB reps=%d\n", omp_get_max_threads(), mib, reps);
    printf("read  %.2f GB/s\n", bytes / best_read / 1e9);
    printf("copy  %.2f GB/s\n", 2 * bytes / best_copy / 1e9);
    printf("triad %.2f GB/s\n", 3 * bytes / best_triad / 1e9);
    return sink == 42.0;  // keep the read loop from being optimised away
}
