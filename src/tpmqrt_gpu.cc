// Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/device.hh"

namespace lapack {

using blas::max, blas::min;

// -----------------------------------------------------------------------------
/// Applies a complex orthogonal matrix Q obtained from a
/// "triangular-pentagonal" complex block reflector H to a general
/// complex matrix C, which consists of two blocks A and B, as follows:
///
/// - side = Left,  trans = NoTrans:   $Q C$
/// - side = Right, trans = NoTrans:   $C Q$
/// - side = Left,  trans = ConjTrans: $Q^H C$
/// - side = Right, trans = ConjTrans: $C Q^H$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.4.0
///
/// @param[in] side
///     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
///     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   No transpose,        apply $Q$;
///     - lapack::Op::Trans:     Transpose,           apply $Q^T$ (real only);
///     - lapack::Op::ConjTrans: Conjugate-transpose, apply $Q^H$.
///
/// @param[in] m
///     The number of rows of the matrix B. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix B. n >= 0.
///
/// @param[in] k
///     The number of elementary reflectors whose product defines
///     the matrix Q.
///
/// @param[in] l
///     The order of the trapezoidal part of V.
///     k >= l >= 0. See Further Details.
///
/// @param[in] nb
///     The block size used for the storage of T. k >= nb >= 1.
///     This must be the same value of nb used to generate T
///     in `lapack::tpqrt`.
///
/// @param[in] V
///     The m-by-k matrix V, stored in an lda-by-k array.
///     The i-th column must contain the vector which defines the
///     elementary reflector H(i), for i = 1,2,...,k, as returned by
///     `lapack::tpqrt` in B. See Further Details.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     If side = Left, ldv >= max(1,m);
///     if side = Right, ldv >= max(1,n).
///
/// @param[in] T
///     The nb-by-k matrix T, stored in an ldt-by-k array.
///     The upper triangular factors of the block reflectors
///     as returned by `lapack::tpqrt`, stored as a nb-by-k matrix.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= nb.
///
/// @param[in,out] A
///     If side = Left,  the k-by-n matrix A, stored in an lda-by-n array;
///     if side = Right, the m-by-k matrix A, stored in an lda-by-k array.
///     On exit, A is overwritten by the corresponding block of
///     $Q C$ or $Q^H C$ or $C Q$ or $C Q^H$. See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     If side = Left,  lda >= max(1,k);
///     If side = Right, lda >= max(1,m).
///
/// @param[in,out] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the m-by-n matrix B.
///     On exit, B is overwritten by the corresponding block of
///     $Q C$ or $Q^H C$ or $C Q$ or $C Q^H$. See Further Details.
///
/// @param[in] ldb
///     The leading dimension of the array B.
///     ldb >= max(1,m).
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The columns of the pentagonal matrix V contain the elementary reflectors
/// H(1), H(2), ..., H(k); V is composed of a rectangular block V1 and a
/// trapezoidal block V2:
/// \[
///     V = \begin{bmatrix}
///             V1
///         \\  V2
///     \end{bmatrix}.
/// \]
/// The size of the trapezoidal block V2 is determined by the parameter l,
/// where 0 <= l <= k; V2 is upper trapezoidal, consisting of the first l
/// rows of a k-by-k upper triangular matrix. If l=k, V2 is upper triangular;
/// if l=0, there is no trapezoidal block, hence V = V1 is rectangular.
///
/// If side = Left:
/// \[
///     C = \begin{bmatrix}
///             A
///         \\  B
///     \end{bmatrix},
/// \]
/// where A is k-by-n, B is m-by-n and V is m-by-k.
///
/// If side = Right:
/// \[
///     C = \begin{bmatrix}  A  &  B  \end{bmatrix},
/// \]
/// where A is m-by-k, B is m-by-n and V is n-by-k.
///
/// The unitary matrix Q is formed from V and T.
///
/// @ingroup tpqrt
template <typename scalar_t>
int64_t tpmqrt(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k, int64_t l, int64_t nb,
    scalar_t const* dV, int64_t lddv,
    scalar_t const* dT, int64_t lddt,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    lapack::Queue& queue )
{
    #define dA(i_, j_) ( dA + i_ + (j_)*ldda )
    #define dB(i_, j_) ( dB + i_ + (j_)*lddb )
    #define dV(i_, j_) ( dV + i_ + (j_)*lddv )
    #define dT(i_, j_) ( dT + i_ + (j_)*lddt )
    #define work(i_, j_) ( work + i_ + (j_)*ldwork )

    // Test the input arguments
    bool left, right;
    bool tran, notran;

    left = (side == Side::Left);
    right = ! left;
    tran = (trans == Op::ConjTrans) || (trans == Op::Trans);
    notran = ! tran;

    // Allocate workspace

    // ConjTrans for complex, Trans for real
    Op op_trans = Op::Trans;
    if (blas::is_complex_v<scalar_t>) {
        op_trans = Op::ConjTrans;
    }

    int64_t ldvq, ldaq;
    if (left) {
        ldvq = max( 1, m );
        ldaq = max( 1, k );
    }
    else if (right) {
        ldvq = max( 1, n );
        ldaq = max( 1, m );
    }

    int64_t info = 0;
    if (! left && ! right) {
        info = -1;
    }
    else if (! tran && ! notran) {
        info = -2;
    }
    else if (m < 0) {
        info = -3;
    }
    else if (n < 0) {
        info = -4;
    }
    else if (k < 0) {
        info = -5;
    }
    else if (l < 0 || l > k) {
        info = -6;
    }
    else if (nb < 1 || (nb > k && k > 0)) {
        info = -7;
    }
    else if (lddv < ldvq) {
        info = -9;
    }
    else if (lddt < nb) {
        info = -11;
    }
    else if (ldda < ldaq) {
        info = -13;
    }
    else if (lddb < max( 1, m )) {
        info = -15;
    }

    if (info != 0) {
        return info;
    }

    // Quick return if possible
    if (m == 0 || n == 0 || k == 0)
        return info;

    if (left && tran) {
        for (int i = 0; i < k; i += nb) {
            int64_t ib = min( nb, k - i );             // width of block
            int64_t mb = min ( m - l + i + ib, m);     // height of block
            int64_t lb = max ( mb - m + l - i, 0);     // height of trapezoidal part

            tprfb( side, op_trans, Direction::Forward, StoreV::Columnwise,
                   mb, n, ib, lb, dV(0, i), lddv, dT(0, i), lddt,
                   dA(i, 0), ldda, dB, lddb, queue );
        }
    }
    else if (right && notran) {
        for (int i = 0; i < k; i += nb) {
            int64_t ib = min( nb, k - i );             // width of block
            int64_t mb = min ( n - l + i + ib, n);     // height of block
            int64_t lb = max ( mb - n + l - i, 0);     // height of trapezoidal part

            tprfb( side, trans, Direction::Forward, StoreV::Columnwise,
                   m, mb, ib, lb, dV(0, i), lddv, dT(0, i), lddt,
                   dA(0, i), ldda, dB, lddb, queue );
        }
    }
    else if (left && notran) {
        int64_t kf = ( (k - 1) / nb ) * nb;
        for (int i = kf; i >= 0; i -= nb) {
            int64_t ib = min( nb, k - i );             // width of block
            int64_t mb = min ( m - l + i + ib, m);     // height of block
            int64_t lb = max ( mb - m + l - i, 0);     // height of trapezoidal part

            tprfb( side, trans, Direction::Forward, StoreV::Columnwise,
                   mb, n, ib, lb, dV(0, i), lddv, dT(0, i), lddt,
                   dA(i, 0), ldda, dB, lddb, queue );
        }
    }
    else if (right && tran) {
        int64_t kf = ( (k - 1) / nb ) * nb;
        for (int i = kf; i >= 0; i -= nb) {
            int64_t ib = min( nb, k - i );             // width of block
            int64_t mb = min ( n - l + i + ib, n);     // height of block
            int64_t lb = max ( mb - n + l - i, 0);     // height of trapezoidal part

            tprfb( side, op_trans, Direction::Forward, StoreV::Columnwise,
                   m, mb, ib, lb, dV(0, i), lddv, dT(0, i), lddt,
                   dA(0, i), ldda,  dB, lddb, queue);
        }
    }

    return info;
}

template int64_t tpmqrt(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k, int64_t l, int64_t nb,
    float const* dV, int64_t lddv,
    float const* dT, int64_t lddt,
    float* dA, int64_t ldda,
    float* dB, int64_t lddb,
    lapack::Queue& queue );

template int64_t tpmqrt(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k, int64_t l, int64_t nb,
    double const* dV, int64_t lddv,
    double const* dT, int64_t lddt,
    double* dA, int64_t ldda,
    double* dB, int64_t lddb,
    lapack::Queue& queue );

template int64_t tpmqrt(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k, int64_t l, int64_t nb,
    std::complex<float> const* dV, int64_t lddv,
    std::complex<float> const* dT, int64_t lddt,
    std::complex<float>* dA, int64_t ldda,
    std::complex<float>* dB, int64_t lddb,
    lapack::Queue& queue );

template int64_t tpmqrt(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k, int64_t l, int64_t nb,
    std::complex<double> const* dV, int64_t lddv,
    std::complex<double> const* dT, int64_t lddt,
    std::complex<double>* dA, int64_t ldda,
    std::complex<double>* dB, int64_t lddb,
    lapack::Queue& queue );

}  // namespace lapack
