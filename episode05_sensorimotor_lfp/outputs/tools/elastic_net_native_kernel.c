/* One fixed cyclic sweep only. Python owns deadlines, KKT and iteration caps.
 * Build on allocated compute with:
 * gcc -O3 -std=c99 -fPIC -shared -ffp-contract=off
 *     elastic_net_native_kernel.c -o elastic_net_native_kernel.so
 * No fast-math, threading, allocation, I/O or convergence rule.
 */
#include <math.h>
#include <stddef.h>

int ep05_elastic_net_sweep(size_t electrodes, const double *gram,
                          const double *cross, double penalty, double scale,
                          double *coefficients) {
    if (electrodes == 0 || gram == NULL || cross == NULL || coefficients == NULL
            || !isfinite(penalty) || penalty <= 0.0
            || !isfinite(scale) || scale < 0.0) {
        return 1;
    }
    const double threshold = penalty * scale;
    for (size_t e = 0; e < electrodes; ++e) {
        const double diagonal = gram[e * electrodes + e];
        const double denominator = diagonal + penalty;
        if (!isfinite(denominator) || denominator <= 0.0) {
            return 2;
        }
        /* Native X and Y are separate scalar problems; neither reads the
         * other coordinate. Preserve physical order and the original
         * C - dot(G_row, W) + G_diagonal * W expression, including diagonal
         * in the dot product before adding its old contribution back.
         */
        for (size_t k = 0; k < 2; ++k) {
            double product = 0.0;
            for (size_t j = 0; j < electrodes; ++j) {
                product += gram[e * electrodes + j] * coefficients[2 * j + k];
            }
            const double partial = cross[2 * e + k] - product
                + diagonal * coefficients[2 * e + k];
            double magnitude = fabs(partial) - threshold;
            if (magnitude < 0.0) {
                magnitude = 0.0;
            }
            const double sign = (partial > 0.0) - (partial < 0.0);
            const double value = sign * magnitude / denominator;
            if (!isfinite(partial) || !isfinite(value)) {
                return 3;
            }
            coefficients[2 * e + k] = value;
        }
    }
    return 0;
}
