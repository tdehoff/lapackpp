// Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

#if LAPACK_VERSION >= 30400  // >= 3.4.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_tpmqrt_device_work( Params& params, bool run )
{
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t l = params.l();
    int64_t nb = params.nb();
    int64_t device = params.device();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.msg();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // skip invalid sizes
    if (k < nb || nb < 1) {
        params.msg() = "skipping: k >= nb >= 1";
        return;
    }
    if (side == lapack::Side::Left  && std::min( m, k ) < l) {
        params.msg() = "skipping: side=left requires min( m, k ) >= l >= 0";
        return;
    }
    if (side == lapack::Side::Right && std::min( n, k ) < l) {
        params.msg() = "skipping: side=right requires min( n, k ) >= l >= 0";
        return;
    }


    // ---------- setup
    // Householder vectors W, Q = \product_i (I - Wi Ti Wi^T)
    // V1, Bhat1 are rectangular
    // V2, Bhat2 are upper trapezoidal
    // left, C = op(Q)*C:
    //      tpmqrt                                  tpqrt
    //      C = [ A ] } k       W = [ I  ] } k      [ What0 ] } k
    //          [ B ] } m           [ V1 ] } m-l    [ Vhat1 ] } m-l
    //                              [ V2 ] } l      [ Vhat2 ] } l
    //            n cols              k cols          k cols
    //
    // right, C = C * op(Q):
    //      tpmqrt                                  tpqrt
    //      C = [ A, B ] } m    W = [ I  ] } k      [ What0 ] } k
    //            k, n cols         [ V1 ] } n-l    [ Vhat1 ] } n-l
    //                              [ V2 ] } l      [ Vhat2 ] } l
    //                                k cols          k cols
    //
    // In tpqrt, What is A, Vhat is B.
    //
    int64_t Vm = (side == blas::Side::Left ? m : n);
    int64_t Am = (side == blas::Side::Left ? k : m);
    int64_t An = (side == blas::Side::Left ? n : k);
    int64_t ldv = roundup( blas::max( 1, Vm ), align );
    int64_t ldt = roundup( blas::max( 1, nb ), align );
    int64_t lda = roundup( blas::max( 1, Am ), align );
    int64_t ldb = roundup( blas::max( 1, m  ), align );
    int64_t ldw = roundup( blas::max( 1, k  ), align );
    size_t size_V  = (size_t) ldv * k;   // m-by-k (Left) or n-by-k (Right)
    size_t size_T  = (size_t) ldt * k;   // nb-by-k
    size_t size_A  = (size_t) lda * An;  // k-by-n (Left) or m-by-k (Right)
    size_t size_B  = (size_t) ldb * n;   // m-by-n
    size_t size_W0 = (size_t) ldw * k;   // k-by-k

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > T( size_T );
    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > W0( size_W0 );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, T.size(), &T[0] );
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    lapack::larnv( idist, iseed, W0.size(), &W0[0] );
    A_ref = A_tst;
    B_ref = B_tst;

    // Get data for Householder reflectors in V and T by factoring matrix
    //     D = [ What0 ] = QR.
    //         [ Vhat  ]
    // When applying Q = I - W T W^H with
    //     W = [ I ],
    //         [ V ]
    // the top block, corresponding to W0, is identity, so isn't used in tpmqrt.
    // Using random data, without factoring, can lead to nan in tpmqrt.
    int64_t info = lapack::tpqrt( Vm, k, l, nb, &W0[0], ldw, &V[0], ldv, &T[0], ldt );
    if (info != 0) {
        fprintf( stderr, "lapack::tpqrt returned error %lld\n", llong( info ) );
    }

    if (verbose > 1) {
        printf( "V =" ); print_matrix( Vm, k,  &V[0], ldv );
        printf( "T =" ); print_matrix( nb, k,  &T[0], ldt );
        printf( "A =" ); print_matrix( Am, An, &A_tst[0], lda );
        printf( "B =" ); print_matrix( m,  n,  &B_tst[0], ldb );
    }

    // Allocate and copy to GPU
    lapack::Queue queue( device );
    scalar_t* dV = blas::device_malloc< scalar_t >( size_V, queue );
    scalar_t* dT = blas::device_malloc< scalar_t >( size_T, queue );
    scalar_t* dA_tst = blas::device_malloc< scalar_t >( size_A, queue );
    scalar_t* dB_tst = blas::device_malloc< scalar_t >( size_B, queue );
    blas::device_copy_matrix( Vm, k, V.data(), ldv, dV, ldv, queue );
    blas::device_copy_matrix( nb,  k,  T.data(), ldt, dT, ldt, queue );
    blas::device_copy_matrix( Am, An, A_tst.data(), lda, dA_tst, lda, queue );
    blas::device_copy_matrix( m,  n,  B_tst.data(), ldb, dB_tst, ldb, queue );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::tpmqrt(
        side, trans, m, n, k, l, nb, dV, ldv, dT, ldt, dA_tst, lda, dB_tst, ldb, queue );
    queue.sync();
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tpqrt returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );  // under-estimate
    params.gflops() = gflop / time;

    // Copy result back to CPU
    blas::device_copy_matrix( Am, An, dA_tst, lda, A_tst.data(), lda, queue );
    blas::device_copy_matrix( m,  n,  dB_tst, ldb, B_tst.data(), ldb, queue );
    queue.sync();
    blas::device_free( dV, queue );
    blas::device_free( dT, queue );
    blas::device_free( dA_tst, queue );
    blas::device_free( dB_tst, queue );

    if (verbose > 1) {
        printf( "Aout =" ); print_matrix( Am, An, &A_tst[0], lda );
        printf( "Bout =" ); print_matrix( m,  n,  &B_tst[0], ldb );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_tpmqrt( to_char( side ), to_char( trans ), m, n, k, l, nb, &V[0], ldv, &T[0], ldt, &A_ref[0], lda, &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tpmqrt returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose > 1) {
            printf( "Aref =" ); print_matrix( Am, An, &A_ref[0], lda );
            printf( "Bref =" ); print_matrix( m,  n,  &B_ref[0], ldb );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_tpmqrt_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_tpmqrt_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tpmqrt_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tpmqrt_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tpmqrt_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tpmqrt_device( Params& params, bool run )
{
    fprintf( stderr, "tpqrt requires LAPACK >= 3.4.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.4.0
