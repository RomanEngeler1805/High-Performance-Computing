// File       : gemm.ispc
// Created    : Wed Oct 17 2018 09:43:19 AM (+0200)
// Description: ISPC GEMM body
// Copyright 2018 ETH Zurich. All Rights Reserved.
#include "common.h"

/**
 * @brief General matrix-matrix multiplication kernel (GEMM). Computes C = AB.
 * ISPC implementation used for SSE2 and AVX2 extended ISA
 *
 * @param A Matrix dimension p x r
 * @param B Matrix dimension r x q
 * @param C Matrix dimension p x q
 * @param p Dimensional parameter
 * @param r Dimensional parameter
 * @param q Dimensional parameter
 */
/* TODO: Missing piece */
#ifdef _ISPC_SSE2_
void gemm_sse2(
#else
void gemm_avx2(
#endif /* _ISPC_SSE2_ */
        const uniform Real* uniform const A,
        const uniform Real* uniform const B,
        uniform Real* uniform const C,
        const uniform int p,
        const uniform int r,
        const uniform int q)
{
    ///////////////////////////////////////////////////////////////////////////
    // TODO: Write your ISPC implementation of a matrix-matrix multiplication
    // here.  Try to use a general type for floating point numbers, such that
    // you can easily compile this kernel for float (single precision, 32bit)
    // and double (double precision, 64bit).  Check out the 'common.h' header
    // for a hint.  Note: A working code can be implemented with ~15 lines of
    // ISPC code.
    //
    // We use two different function names for the same function, such that we
    // can compile for two different targets (SSE2 and AVX2) and link both of
    // them to our application code in 'gemm.cpp'.
    ///////////////////////////////////////////////////////////////////////////

    int VTILE = 16, HTILE = 128;
    Real* tile[VTILE][HTILE];

    for (uniform int i = 0; i < p; i+= VTILE)
    for (uniform int j = 0; j < q; j+= HTILE)
    {
        // initialize tile with all zeros
        for (uniform int tv = 0; tv < VTILE; ++tv){
            foreach(th=0 ... HTILE){
                tile[tv][th] = (Real)0.;
            }
        }

        // inner product
        for (uniform k = 0; k< r; k++){
            for (uniform int tv = 0; tv < VTILE; ++tv){
                foreach(th=0 ... HTILE){
                    tile[tv][th] += A[(i+ tv)* r + (k)]* B[k* q+ (j+ th)];
                }
            }
        }

        // add up C
        for (uniform int tv = 0; tv < VTILE; ++tv){
            foreach(th=0 ... HTILE){
                C[(i+ tv)* q+ j+ th] = tile[tv][th];
            }
        }
    }

}
