// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fitted walls, the fit: the cut of a
// triangle mesh by a cubic grid, decided exactly, and the air, open area and wall area of
// every cell that a face reaches.
//
// Input. The mesh as given (Mesh: the vertex and face arrays the BVH was built from), each
// face's material, the grid as the doubles of its faces and centres along each axis (Grid),
// the source point, the width of the matched layer in cells, and a thread count.
//
// The lattice Q. The doubles are taken once onto a lattice of 64-bit integers: with E the
// largest half-extent of the faces that reach the grid and of the grid's own box, grown by 2h,
// and p = ceil(log2 E) + 1,
// a coordinate x becomes round((x - X0) 2^(61 - p)), ties to even, formed exactly; X0 is a
// multiple of 2^(p - 2). A face whose vertex lies 2^24 cells or more from that middle is
// refused (R0), as is a grid too far from its own origin for its cell. Every later decision
// is made on Q and is exact; doubles only filter, measure and route.
//
// Planes and points. Every plane is integer (n, d) on Q, n primitive with its first nonzero
// component positive, one id per plane: the faces' planes (ids in the order of their tuples),
// one coverage plane (B - A) x e_axis per face edge, and the grid planes. A point is a lattice
// point (a mesh vertex, a grid corner) or the meet of three planes. A point on the line of a
// welded mesh edge whose end lies on its third plane is that end, a lattice point. The sign
// of a point against a plane is side(): the plane is one of the point's own (0); else a float
// filter in cells from the frame's corner with a certified bound (a lattice point's error is
// its rounding; a meet's is bounded by the way Table::point() makes it: from exact integer
// numerators and denominators where the meet lies on a welded edge's line or on two axis
// planes, else by Cramer's rule in double-double with a bound on every rounding, accepted where
// that bound is at most 2^-60 cells times the larger of 1 and the point's largest coordinate,
// and the point made from its exact coordinates otherwise); else incidence by the mesh (a
// welded vertex on a plane through it, a point on a welded edge's line and a plane through
// both its ends: 0); else the exact sign, in 256-bit integers for a lattice point and by
// exact Cramer in Big (at most 640 bits: |n| < 2^127, |d| < 2^254) for a meet.
//
// Cubes. In each cell that a face's part of positive area meets, the cell is cut by a binary
// tree: a leaf is split by a face plane exactly when some face on it has positive area inside
// the leaf; planes are taken in the order of their ids. New points on an edge are made once,
// by the edge's planes, and caps are chained by adjacency. Each face plane's part in a leaf is
// covered by the faces on it (their coverage half-planes, in key order of the faces), which
// gives its open part and its walls with the material of each side. Faces lying exactly on a
// grid plane (membranes) are cut in the squares between cells instead. Regions are the leaves;
// their volumes are exact up to the measure of the approximations.
//
// Squares. Each square between two cells that a cut or a membrane touches is cut by the
// two cells' trees and covered by its membranes: its pieces join the regions on its two
// sides (open) or not (wall). The total covered area, each face's covered part counted once
// (faces lying exactly on one another each count their own area, in cubes and squares
// alike), against the total area of the faces' parts in cut cubes and squares (membranes
// included) is reported as Stats::balance, a measure (totals only, not per face or per
// side). Each facet between two leaves of a cube and each square, against its pieces' bare
// area plus the walls given to each of its sides, is reported as Stats::partition, a second
// measure (totals only, the worse side). Every build throws std::logic_error where either
// passes 1e-9 relative, as it does where one of the self-checks K1 to K8 fails.
//
// Candidates and measures on Q. Which cubes a face, a zone or a pair of
// faces may meet, and which faces a way may cross, are read on the lattice: integer boxes
// against the grid planes G, and triangles in each cube's own frame (q - G_i) / (G_i+1 - G_i),
// never from X0 + 2^-s q - origin in doubles, which rounds by ulp(|x|) / h cells far from the
// origin. Every cell is measured between its own planes: an uncut air cell holds
// widths(0)[i] widths(1)[j] widths(2)[k] cells^3 (1 to within 1e-15 near the origin), and
// Stats::air_volume is the air's total on Q.
//
// Air. Components by union-find over the cut regions, the uncut cells (as x-runs) and the
// open pieces of squares. The air is the component that holds the source; a cell is air when
// the region holding its centre is in it (a point exactly on a plane is taken on its + side).
// Refused, with a message: the source in a cell on the grid's edge; the source in a closed
// space that holds no cell centre; the model reaching into the matched layer; faces in a cube
// next to a cell of the layer that holds air (code kRefusedNextToLayer: the layer reads its
// neighbours whole, so each of them owns its whole cube and face towards it); a part of the
// air open only to the layer (T-4). Each region of the air is owned by one air cell, the
// nearest by the exact open flag of the squares between them (1e-12 band, ties to the lowest
// cell); the cell's air volume, open areas (by face) and wall area are the sums of what it
// owns. Counts of what is left out are in Stats.
//
// The grid's edge. Every air cell next to an edge cell (sharing a face with one) is a row; such
// a row that owns nothing cut holds its own volume. A row's wall area towards the grid's edge
// (its edge column) holds the open pieces in the edge plane between an edge cell and a region
// the row owns, in the row's own cube or in a neighbour's cube whose centre is solid, and the
// whole faces its cell shares with edge cells where no square was cut (w_b w_c each);
// Stats::edge_area is their total.
// open_sides() says which sides of the grid are open, decided from the cells' kinds and the
// squares' exact open flags only. A side is open when an air cell lies next to an edge cell on
// that side, when an open piece of a square in that side's edge plane joins a region of the
// source's air to the edge cell, or when a cell of the matched layer lies within the layer's
// width of that side. A closed piece (a wall in the edge plane) and air the source does not
// reach (an edge pocket) open no side.
//
// Refusals (i) and (ii). (i): the mesh's mod-2 boundary, line by line, as odd intervals; each
// interval's zone (the interval grown by kGap = 0.01 h in each coordinate, its facets exact
// lattice planes) cuts the cubes it meets again (pass B); leaves inside zones are not nodes of
// a second union-find, and a part of the air outside the zones that reaches the source only
// through them is refused (code kRefusedSlit), as is a source inside a zone. (ii): pairs of
// faces at under 7.18 degrees (exact) whose projections overlap (exact 2D test) and that lie
// within 7/4 kGap of each other over the overlap (exact), each with a covered part facing the
// air on the side towards the other, are refused (kRefusedThinLayer). A refusal does not
// throw from the constructor: refused() and refusal() give it, check() throws
// std::invalid_argument with the message; every cell is then kSolid and the rows are empty.
// The wave solver adds one condition of its own, require_whole_widths() (code kRefusedWidths):
// every cell's width on the lattice within 2^-24 of a whole cell.
//
// Points (T-1). locate() takes a point onto Q with its centre bracket and exact ties;
// way() decides whether the segment between two lattice points, both moved by
// (eps, eps^2, eps^3), crosses a face (box filter, then the plane and edge signs in 256 bits,
// none zero after the perturbation); on_face() is the exact on-face flag. The solver's
// stencil on these (fdtd.hpp, reading_cells_fitted) gives cut_points().
//
// Arithmetic. The fit uses IEEE-754 binary64 addition, subtraction, multiplication, division
// and square root, and exact integer arithmetic (U128, I128, I256, Big), and nothing else that
// feeds a result: no long double, no fused multiply-add, no library function whose rounding the
// standard leaves open. Its results are then the same bits in every build whose doubles are
// evaluated in double, rounded to nearest, with no contraction of a product and a sum and no
// reassociation (the package builds with -ffp-contract=off, and /fp:precise for MSVC). In any
// other build CutCells's constructor throws std::runtime_error ("fitted walls: ..."): a
// constant known at compile time and a probe run at every construction find such builds
// (require_arithmetic), and every thread of the fit checks the rounding mode.
//
// Test-only build flags: ACOUSTIC_SPACES_CUT_CHECK computes the exact sign beside every
// filtered or incident one and counts disagreements; ACOUSTIC_SPACES_CUT_EXACT_POINTS solves
// every meet from its exact coordinates; ACOUSTIC_SPACES_PORTABLE_PRIMITIVES takes the portable
// code of the wide integers' primitives in place of the compiler's.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "acoustic_spaces/ray_tracer.hpp"
#include "acoustic_spaces/vec3.hpp"

namespace acoustic_spaces {

// ---- the exact kernel -------------------------------------------------------------------

namespace cut_detail {

// ---- the primitives of the wide integers ------------------------------------------------
//
// U128 and I128 below are classes in every build, with the values and results of 128-bit
// integers: modular arithmetic on 128 bits, two's complement for I128, division truncated
// towards zero, an arithmetic right shift for I128, and conversion to double rounded to
// nearest, ties to even. They rest on three primitives: the full product of two 64-bit numbers
// and the counts of leading and of trailing zero bits. Each primitive takes the compiler's
// instruction where the compiler offers one (GCC and Clang: the product of unsigned __int128,
// __builtin_clzll and __builtin_ctzll; MSVC on x64 and ARM64: _umul128 or __umulh,
// _BitScanReverse64 and _BitScanForward64) and portable code otherwise, or everywhere when
// ACOUSTIC_SPACES_PORTABLE_PRIMITIVES is defined. The two give the same results. The portable
// code is compiled in every build under its own name, so that a test can compare it with the
// primitive at run time (fit_selftest). Clang on Windows (clang-cl) defines _MSC_VER as well as
// __SIZEOF_INT128__ and __clang__; every branch below tests the latter two first, so it takes
// Clang's code. None of these functions is constexpr, because the intrinsics are not.

// The low 64 bits of a b; its high 64 bits go to hi. Four products of 32 bits.
inline std::uint64_t mul64_portable(std::uint64_t a, std::uint64_t b, std::uint64_t &hi) {
    const std::uint64_t m = 0xffffffffu;
    const std::uint64_t a0 = a & m, a1 = a >> 32, b0 = b & m, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & m) + (p10 & m);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (p00 & m) | (mid << 32);
}
// The number of trailing zero bits of x, which is not 0: the lowest set bit of x times a de
// Bruijn sequence, whose top six bits are distinct for the 64 powers of two.
inline int ctz64_portable(std::uint64_t x) {
    static const unsigned char kIndex[64] = {0,  1,  48, 2,  57, 49, 28, 3,  61, 58, 50, 42, 38, 29, 17, 4,
                                             62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,
                                             63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11,
                                             46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9,  13, 8,  7,  6};
    return kIndex[((x & (0 - x)) * 0x03f79d71b4cb0a89ull) >> 58];
}
// The number of leading zero bits of x, which is not 0: a binary search in six steps.
inline int clz64_portable(std::uint64_t x) {
    int n = 0;
    if (!(x >> 32)) n += 32, x <<= 32;
    if (!(x >> 48)) n += 16, x <<= 16;
    if (!(x >> 56)) n += 8, x <<= 8;
    if (!(x >> 60)) n += 4, x <<= 4;
    if (!(x >> 62)) n += 2, x <<= 2;
    if (!(x >> 63)) n += 1;
    return n;
}

inline std::uint64_t mul64(std::uint64_t a, std::uint64_t b, std::uint64_t &hi) {
#if defined(ACOUSTIC_SPACES_PORTABLE_PRIMITIVES)
    return mul64_portable(a, b, hi);
#elif defined(__SIZEOF_INT128__)
    // __extension__ keeps -Wpedantic quiet about GCC's and Clang's 128-bit integers.
    __extension__ const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    hi = static_cast<std::uint64_t>(p >> 64);
    return static_cast<std::uint64_t>(p);
#elif defined(_MSC_VER) && defined(_M_X64)
    return _umul128(a, b, &hi);
#elif defined(_MSC_VER) && defined(_M_ARM64)
    hi = __umulh(a, b);
    return a * b;
#else
    return mul64_portable(a, b, hi);
#endif
}
inline int ctz64(std::uint64_t x) {
#if defined(ACOUSTIC_SPACES_PORTABLE_PRIMITIVES)
    return ctz64_portable(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long i = 0;
    _BitScanForward64(&i, x);
    return static_cast<int>(i);
#else
    return ctz64_portable(x);
#endif
}
inline int clz64(std::uint64_t x) {
#if defined(ACOUSTIC_SPACES_PORTABLE_PRIMITIVES)
    return clz64_portable(x);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long i = 0;
    _BitScanReverse64(&i, x);
    return 63 - static_cast<int>(i);
#else
    return clz64_portable(x);
#endif
}

// x 2^e rounded to nearest, ties to even, for a 64-bit x with a sticky bit that stands for
// nonzero bits below it: the top 53 bits are rounded in integers, and the result (at most 2^53)
// is converted exactly through int64. The compiler's conversion never sees an unsigned 64-bit
// value or one of 2^53 or more, whose rounding C++ leaves to the implementation.
inline double round_u64(std::uint64_t x, bool sticky, int e) {
    if (x == 0) return 0.0;
    const int lz = clz64(x);
    const std::uint64_t m = x << lz;
    std::uint64_t keep = m >> 11;
    const std::uint64_t rest = m & 0x7FF;
    if (rest > 0x400 || (rest == 0x400 && (sticky || (keep & 1)))) ++keep;
    return std::ldexp(static_cast<double>(static_cast<std::int64_t>(keep)), e + 11 - lz);
}

class U128 {
public:
    std::uint64_t lo = 0, hi = 0;
    U128() = default;
    template <class T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, int> = 0>
    U128(T v)  // NOLINT: implicit, as the built-in integer is
        : lo(static_cast<std::uint64_t>(v)), hi(negative(v) ? ~std::uint64_t{0} : 0) {}
    static U128 of(std::uint64_t h, std::uint64_t l) {
        U128 r;
        r.hi = h;
        r.lo = l;
        return r;
    }
    template <class T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, int> = 0>
    explicit operator T() const {
        return static_cast<T>(lo);
    }
    explicit operator bool() const { return (lo | hi) != 0; }
    // Rounded to nearest, ties to even: the top 64 bits, with a sticky bit for the rest.
    explicit operator double() const {
        if (hi == 0) return lo < (std::uint64_t{1} << 53) ? static_cast<double>(static_cast<std::int64_t>(lo)) : round_u64(lo, false, 0);
        const int s = 64 - clz64(hi);
        const std::uint64_t top = s == 64 ? hi : (hi << (64 - s)) | (lo >> s);
        const bool st = s == 64 ? lo != 0 : (lo << (64 - s)) != 0;
        return round_u64(top, st, s);
    }

    friend U128 operator+(const U128 &a, const U128 &b) {
        const std::uint64_t l = a.lo + b.lo;
        return of(a.hi + b.hi + (l < a.lo ? 1u : 0u), l);
    }
    friend U128 operator-(const U128 &a, const U128 &b) { return of(a.hi - b.hi - (a.lo < b.lo ? 1u : 0u), a.lo - b.lo); }
    friend U128 operator-(const U128 &a) { return U128() - a; }
    friend U128 operator~(const U128 &a) { return of(~a.hi, ~a.lo); }
    friend U128 operator*(const U128 &a, const U128 &b) {
        U128 r = mul64(a.lo, b.lo);
        r.hi += a.lo * b.hi + a.hi * b.lo;
        return r;
    }
    friend U128 operator&(const U128 &a, const U128 &b) { return of(a.hi & b.hi, a.lo & b.lo); }
    friend U128 operator|(const U128 &a, const U128 &b) { return of(a.hi | b.hi, a.lo | b.lo); }
    friend U128 operator^(const U128 &a, const U128 &b) { return of(a.hi ^ b.hi, a.lo ^ b.lo); }
    friend U128 operator<<(const U128 &a, int k) {
        if (k <= 0) return a;
        if (k >= 128) return U128();
        if (k >= 64) return of(a.lo << (k - 64), 0);
        return of((a.hi << k) | (a.lo >> (64 - k)), a.lo << k);
    }
    friend U128 operator>>(const U128 &a, int k) {
        if (k <= 0) return a;
        if (k >= 128) return U128();
        if (k >= 64) return of(0, a.hi >> (k - 64));
        return of(a.hi >> k, (a.lo >> k) | (a.hi << (64 - k)));
    }
    friend U128 operator/(const U128 &a, const U128 &b) {
        U128 q, r;
        divmod(a, b, q, r);
        return q;
    }
    friend U128 operator%(const U128 &a, const U128 &b) {
        U128 q, r;
        divmod(a, b, q, r);
        return r;
    }
    friend bool operator==(const U128 &a, const U128 &b) { return a.lo == b.lo && a.hi == b.hi; }
    friend bool operator!=(const U128 &a, const U128 &b) { return !(a == b); }
    friend bool operator<(const U128 &a, const U128 &b) { return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo; }
    friend bool operator>(const U128 &a, const U128 &b) { return b < a; }
    friend bool operator<=(const U128 &a, const U128 &b) { return !(b < a); }
    friend bool operator>=(const U128 &a, const U128 &b) { return !(a < b); }
    U128 &operator+=(const U128 &b) { return *this = *this + b; }
    U128 &operator-=(const U128 &b) { return *this = *this - b; }
    U128 &operator*=(const U128 &b) { return *this = *this * b; }
    U128 &operator/=(const U128 &b) { return *this = *this / b; }
    U128 &operator%=(const U128 &b) { return *this = *this % b; }
    U128 &operator&=(const U128 &b) { return *this = *this & b; }
    U128 &operator|=(const U128 &b) { return *this = *this | b; }
    U128 &operator^=(const U128 &b) { return *this = *this ^ b; }
    U128 &operator<<=(int k) { return *this = *this << k; }
    U128 &operator>>=(int k) { return *this = *this >> k; }

    // The full product of two 64-bit numbers.
    static U128 mul64(std::uint64_t a, std::uint64_t b) {
        std::uint64_t h = 0;
        const std::uint64_t l = cut_detail::mul64(a, b, h);
        return of(h, l);
    }
    // a = q b + r, 0 <= r < b; b is not 0 (as for the built-in division). Shift and subtract,
    // with a fast path where both operands fit 64 bits.
    static void divmod(const U128 &a, const U128 &b, U128 &q, U128 &r) {
        if (a.hi == 0 && b.hi == 0) {
            q = U128(a.lo / b.lo);
            r = U128(a.lo % b.lo);
            return;
        }
        q = U128();
        r = U128();
        for (int i = 127; i >= 0; --i) {
            r = (r << 1) | ((a >> i) & U128(1));
            if (r >= b) {
                r -= b;
                q |= U128(1) << i;
            }
        }
    }

private:
    template <class T>
    static bool negative(T v) {
        if constexpr (std::is_signed<T>::value) return v < 0;
        else return false;
    }
};

class I128 {
public:
    U128 u;  // the bits, two's complement
    I128() = default;
    template <class T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, int> = 0>
    I128(T v) : u(v) {}  // NOLINT: implicit, as the built-in integer is
    explicit I128(const U128 &x) : u(x) {}
    // Truncated towards zero, as the built-in conversion (|v| < 2^127).
    explicit I128(double v) {
        if (std::abs(v) < 9.0e18) {
            u = U128(static_cast<std::int64_t>(v));
            return;
        }
        int e = 0;
        const double m = std::frexp(v, &e);  // v = m 2^e, 1/2 <= |m| < 1, e > 53 here
        u = (I128(static_cast<std::int64_t>(std::ldexp(m, 53))) << (e - 53)).u;
    }
    template <class T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, int> = 0>
    explicit operator T() const {
        return static_cast<T>(u.lo);
    }
    explicit operator U128() const { return u; }
    explicit operator bool() const { return static_cast<bool>(u); }
    explicit operator double() const { return negative() ? -static_cast<double>(U128() - u) : static_cast<double>(u); }
    bool negative() const { return (u.hi >> 63) != 0; }

    friend I128 operator+(const I128 &a, const I128 &b) { return I128(a.u + b.u); }
    friend I128 operator-(const I128 &a, const I128 &b) { return I128(a.u - b.u); }
    friend I128 operator-(const I128 &a) { return I128(U128() - a.u); }
    friend I128 operator~(const I128 &a) { return I128(~a.u); }
    friend I128 operator*(const I128 &a, const I128 &b) { return I128(a.u * b.u); }
    friend I128 operator&(const I128 &a, const I128 &b) { return I128(a.u & b.u); }
    friend I128 operator|(const I128 &a, const I128 &b) { return I128(a.u | b.u); }
    friend I128 operator^(const I128 &a, const I128 &b) { return I128(a.u ^ b.u); }
    friend I128 operator<<(const I128 &a, int k) { return I128(a.u << k); }
    friend I128 operator>>(const I128 &a, int k) { return a.negative() ? I128(~(~a.u >> k)) : I128(a.u >> k); }
    friend I128 operator/(const I128 &a, const I128 &b) {
        U128 q, r;
        U128::divmod(a.mag(), b.mag(), q, r);
        return a.negative() != b.negative() ? -I128(q) : I128(q);
    }
    friend I128 operator%(const I128 &a, const I128 &b) {
        U128 q, r;
        U128::divmod(a.mag(), b.mag(), q, r);
        return a.negative() ? -I128(r) : I128(r);
    }
    friend bool operator==(const I128 &a, const I128 &b) { return a.u == b.u; }
    friend bool operator!=(const I128 &a, const I128 &b) { return a.u != b.u; }
    friend bool operator<(const I128 &a, const I128 &b) { return a.negative() != b.negative() ? a.negative() : a.u < b.u; }
    friend bool operator>(const I128 &a, const I128 &b) { return b < a; }
    friend bool operator<=(const I128 &a, const I128 &b) { return !(b < a); }
    friend bool operator>=(const I128 &a, const I128 &b) { return !(a < b); }
    I128 &operator+=(const I128 &b) { return *this = *this + b; }
    I128 &operator-=(const I128 &b) { return *this = *this - b; }
    I128 &operator*=(const I128 &b) { return *this = *this * b; }
    I128 &operator/=(const I128 &b) { return *this = *this / b; }
    I128 &operator%=(const I128 &b) { return *this = *this % b; }
    I128 &operator<<=(int k) { return *this = *this << k; }
    I128 &operator>>=(int k) { return *this = *this >> k; }

private:
    U128 mag() const { return negative() ? U128() - u : u; }
};

// A signed integer of up to kLimbs limbs of 64 bits, held as sign and magnitude. The fit's
// predicates need at most 640 bits (the bounds are in the head comment); a sum or product
// that would not fit throws std::logic_error, which would be a defect of those bounds.

// Double-double: hi + lo with |lo| <= ulp(hi) / 2, from IEEE double operations only.
struct DD {
    double hi = 0.0, lo = 0.0;
    DD() = default;
    DD(double h) : hi(h), lo(0.0) {}  // NOLINT
    DD(double h, double l) : hi(h), lo(l) {}
    static DD two_sum(double a, double b) {
        const double s = a + b, bb = s - a;
        return {s, (a - (s - bb)) + (b - bb)};
    }
    static DD quick(double a, double b) {
        const double s = a + b;
        return {s, b - (s - a)};
    }
    static void split(double a, double &h, double &l) {
        const double t = 134217729.0 * a;
        h = t - (t - a);
        l = a - h;
    }
    static DD two_prod(double a, double b) {
        const double p = a * b;
        double ah, al, bh, bl;
        split(a, ah, al);
        split(b, bh, bl);
        return {p, ((ah * bh - p) + ah * bl + al * bh) + al * bl};
    }
    friend DD operator+(const DD &a, const DD &b) {
        DD s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
        s.lo += t.hi;
        s = quick(s.hi, s.lo);
        s.lo += t.lo;
        return quick(s.hi, s.lo);
    }
    friend DD operator-(const DD &a) { return {-a.hi, -a.lo}; }
    friend DD operator-(const DD &a, const DD &b) { return a + (-b); }
    friend DD operator*(const DD &a, const DD &b) {
        DD p = two_prod(a.hi, b.hi);
        p.lo += a.hi * b.lo + a.lo * b.hi;
        return quick(p.hi, p.lo);
    }
    // DWTimesFP1 (Joldes, Muller and Popescu 2017, Algorithm 7): relative error below 1.5 u^2 + 4 u^3.
    static DD times_fp(const DD &x, double y) {
        const DD c = two_prod(x.hi, y);
        const double cl2 = x.lo * y;
        const DD t = quick(c.hi, cl2);
        const double tl2 = t.lo + c.lo;
        return quick(t.hi, tl2);
    }
    // DWDivDW2 (the same paper, Algorithm 17): relative error below 15 u^2 + 56 u^3.
    friend DD operator/(const DD &x, const DD &y) {
        const double th = x.hi / y.hi;
        const DD r = times_fp(y, th);
        const double ph = x.hi - r.hi;
        const double dl = x.lo - r.lo;
        const double d = ph + dl;
        const double tl = d / y.hi;
        return quick(th, tl);
    }
    friend bool operator<(const DD &a, const DD &b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
    friend bool operator>(const DD &a, const DD &b) { return b < a; }
    friend bool operator<=(const DD &a, const DD &b) { return !(b < a); }
    explicit operator double() const { return hi + lo; }
};
inline DD dd_abs(const DD &a) { return a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0) ? -a : a; }
inline DD dd_sqrt(const DD &a) {
    if (a.hi <= 0.0) return DD(0.0);
    const double x = std::sqrt(a.hi);
    const DD r = a - DD::two_prod(x, x);
    return DD::quick(x, r.hi / (2.0 * x));
}
inline DD dd_of_u64(std::uint64_t v) {
    return DD::two_sum(static_cast<double>(v >> 32) * 4294967296.0, static_cast<double>(v & 0xffffffffu));
}

// ---- the arithmetic the fit needs -------------------------------------------------------
//
// The fit's results are the same bits in every build whose doubles are IEEE-754 binary64, evaluated in double,
// rounded to nearest, with no product and sum contracted into a fused multiply-add and no reassociation. A build
// that is not of this kind gets no fit: CutCells's constructor throws std::runtime_error. kArithmeticRefusal is
// the reason known at compile time (nullptr where there is none); arithmetic_probe() finds at run time what the
// macros cannot show.
static_assert(std::numeric_limits<double>::is_iec559, "acoustic_spaces: doubles must be IEEE-754 binary64");
#if defined(__FAST_MATH__)
inline constexpr const char *kArithmeticRefusal =
    "this build was compiled with -ffast-math (or -Ofast), which lets the compiler reorder and fuse "
    "floating-point operations";
#elif defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
inline constexpr const char *kArithmeticRefusal =
    "this build evaluates doubles in a wider precision (FLT_EVAL_METHOD is not 0, as with x87 arithmetic "
    "without SSE2)";
#elif defined(_MSC_VER) && !defined(__clang__) && (defined(_M_FP_FAST) || defined(_M_FP_CONTRACT))
inline constexpr const char *kArithmeticRefusal =
    "this build was compiled with /fp:fast or /fp:contract, which let the compiler reorder and fuse "
    "floating-point operations";
#else
inline constexpr const char *kArithmeticRefusal = nullptr;
#endif

// One test of the probe: its name, the doubles it gave, and the doubles IEEE-754 arithmetic gives.
struct ProbeResult {
    const char *name = nullptr;
    double got[3] = {0.0, 0.0, 0.0};
    double want[3] = {0.0, 0.0, 0.0};
};

// Runs the fit's own error-free routines, the expression forms that a contracting or reassociating compiler
// changes, and the compiler's conversion of 64-bit integers of 2^53 and more to double at ties, on inputs read
// through volatile variables, so that no compiler can fold them. Returns the first test whose result differs from
// IEEE-754 arithmetic (name nullptr when every test passes).
inline ProbeResult arithmetic_probe() {
    // Every test reads its inputs afresh from the volatile variables, so that no product or sum is shared between
    // tests (a compiler fuses a product only where every use of it is a sum, so one shared product could hide the
    // contraction of another test), and stores its results into volatile variables before they are compared, so
    // that each test is computed in straight-line code as written (a compiler that moves a division into the
    // branch that reads it no longer sees the three quotients by one divisor together).
    volatile double va = 0x1.00000004p+0, vb = 0x1.fffffff8p-1, vc = -1.0, vone = 1.0, vtiny = 0x1p-60;
    volatile double vx1 = 5.0, vx2 = 7.0, vx3 = 11.0, vy = 3.0, vtwo = 2.0;
    ProbeResult r;
    auto check = [&r](const char *name, std::initializer_list<double> got, std::initializer_list<double> want) {
        if (r.name != nullptr) return;
        std::size_t i = 0;
        bool same = true;
        for (auto g = got.begin(), w = want.begin(); g != got.end(); ++g, ++w, ++i) {
            r.got[i] = *g;
            r.want[i] = *w;
            // Equal as doubles and of the same sign (a fused result can differ from 0 in sign only).
            same = same && *g == *w && std::signbit(*g) == std::signbit(*w);
        }
        if (!same) r.name = name;
    };
    // a b = 1 - 2^-60, which rounds to 1: a * b + c is 0 when the product is rounded, -2^-60 when fused.
    {
        const double a = va, b = vb, c = vc;
        volatile double e = a * b + c;
        check("a product and a sum in one expression", {e}, {0.0});
    }
    {
        const double a = va, b = vb, c = vc;
        const double p = a * b;
        volatile double e = p + c;
        check("a product and a sum in two statements", {e}, {0.0});
    }
    {
        double h = 0.0, l = 0.0;
        DD::split(va, h, l);
        volatile double oh = h, ol = l;
        check("the split", {oh, ol}, {1.0, 0x1p-30});
    }
    {
        const DD tp = DD::two_prod(va, vb);
        volatile double oh = tp.hi, ol = tp.lo;
        check("two_prod", {oh, ol}, {1.0, -0x1p-60});
    }
    {
        const DD ts = DD::two_sum(vone, vtiny), tr = DD::two_sum(vtiny, vone);
        volatile double oh = ts.hi, ol = ts.lo, orl = tr.lo;
        check("two_sum", {oh, ol, orl}, {1.0, 0x1p-60, 0x1p-60});
    }
    {
        const DD qs = DD::quick(vone, vtiny);
        volatile double oh = qs.hi, ol = qs.lo;
        check("quick", {oh, ol}, {1.0, 0x1p-60});
    }
    {
        const double y = vy;
        volatile double q1 = vx1 / y, q2 = vx2 / y, q3 = vx3 / y;
        check("three quotients by one divisor", {q1, q2, q3},
              {0x1.aaaaaaaaaaaabp+0, 0x1.2aaaaaaaaaaabp+1, 0x1.d555555555555p+1});
    }
    {
        volatile double r2 = std::sqrt(static_cast<double>(vtwo));
        check("a square root", {r2}, {0x1.6a09e667f3bcdp+0});
    }
    // The fit converts 64-bit integers of 2^53 and more to double with the compiler's conversion (the lattice
    // coordinates of Table::lattice() and of the axis planes in Table::point()). C++ leaves the rounding of such
    // a conversion to the implementation; the fit needs it rounded to nearest, ties to even, as IEEE-754's
    // conversion is. Each value below lies halfway between two doubles, or just past halfway.
    {
        volatile std::int64_t t1 = (std::int64_t{1} << 53) + 1, t2 = (std::int64_t{1} << 53) + 3,
                              t3 = -((std::int64_t{1} << 62) + (std::int64_t{1} << 9));
        volatile double c1 = static_cast<double>(t1), c2 = static_cast<double>(t2), c3 = static_cast<double>(t3);
        check("64-bit integers to double at ties", {c1, c2, c3}, {0x1p53, 0x1.0000000000002p53, -0x1p62});
    }
    {
        volatile std::int64_t t1 = (std::int64_t{1} << 62) + 3 * (std::int64_t{1} << 9),
                              t2 = -((std::int64_t{1} << 53) + 3), t3 = (std::int64_t{1} << 62) + (std::int64_t{1} << 9) + 1;
        volatile double c1 = static_cast<double>(t1), c2 = static_cast<double>(t2), c3 = static_cast<double>(t3);
        check("64-bit integers to double at and past ties", {c1, c2, c3},
              {0x1.0000000000002p62, -0x1.0000000000002p53, 0x1.0000000000001p62});
    }
    return r;
}

// The refusal of a build or a floating-point environment that the fit cannot run in, as CutCells's constructor
// throws it.
[[noreturn]] inline void refuse_arithmetic(const std::string &why) {
    throw std::runtime_error("fitted walls: " + why +
                             ". The fit's results would differ from those of other platforms. Build the package "
                             "with -ffp-contract=off and without -ffast-math, -Ofast or "
                             "-funsafe-math-optimizations (GCC, Clang), or with /fp:precise (MSVC).");
}

inline std::string hex_double(double x) {
    char s[40];
    std::snprintf(s, sizeof s, "%a", x);
    return s;
}

// Throws the refusal unless the rounding mode is to nearest; `where` names the thread.
inline void require_rounding(const char *where) {
    const int mode = std::fegetround();
    if (mode != FE_TONEAREST)
        throw std::runtime_error(std::string("fitted walls: the floating-point rounding mode in ") + where +
                                 " is not round-to-nearest (fegetround() returned " + std::to_string(mode) +
                                 "); set FE_TONEAREST before the fit.");
}

// Throws the refusal unless the build and the environment are those the fit needs.
inline void require_arithmetic() {
    if (kArithmeticRefusal != nullptr) refuse_arithmetic(kArithmeticRefusal);
    const ProbeResult r = arithmetic_probe();
    if (r.name != nullptr) {
        std::string got, want;
        for (int i = 0; i < 3; ++i) {
            got += (i ? " " : "") + hex_double(r.got[i]);
            want += (i ? " " : "") + hex_double(r.want[i]);
        }
        refuse_arithmetic(std::string("this build's floating-point arithmetic is not IEEE-754 arithmetic as written: it "
                                      "contracts or reassociates operations, or rounds a conversion otherwise (the "
                                      "probe's test '") + r.name + "' gave " + got + " where IEEE-754 arithmetic gives " +
                          want + ")");
    }
    require_rounding("the calling thread");
}

class Big {
public:
    static constexpr int kLimbs = 12;
    Big() = default;
    Big(I128 x) {  // NOLINT: implicit on purpose
        const U128 m = x < 0 ? U128{0} - static_cast<U128>(x) : static_cast<U128>(x);
        w_[0] = static_cast<std::uint64_t>(m);
        w_[1] = static_cast<std::uint64_t>(m >> 64);
        n_ = 2;
        neg_ = x < 0;
        trim();
    }
    // From a built-in integer, as from I128 (needed where I128 is a class).
    template <class T, std::enable_if_t<std::is_integral<T>::value && !std::is_same<T, bool>::value, int> = 0>
    Big(T x) : Big(I128(x)) {}  // NOLINT: implicit on purpose
    int sign() const { return n_ == 0 ? 0 : (neg_ ? -1 : 1); }
    friend Big operator-(Big a) {
        if (a.n_) a.neg_ = !a.neg_;
        return a;
    }
    friend Big operator+(const Big &a, const Big &b) {
        Big r;
        if (a.neg_ == b.neg_) {
            add_mag(a, b, r);
            r.neg_ = a.neg_;
        } else if (cmp_mag(a, b) >= 0) {
            sub_mag(a, b, r);
            r.neg_ = a.neg_;
        } else {
            sub_mag(b, a, r);
            r.neg_ = b.neg_;
        }
        r.trim();
        return r;
    }
    friend Big operator-(const Big &a, const Big &b) { return a + (-b); }
    friend Big operator*(const Big &a, const Big &b) {
        Big r;
        if (a.n_ == 0 || b.n_ == 0) return r;
        if (a.n_ + b.n_ > kLimbs) throw std::logic_error("fitted walls: an exact product exceeds its bound");
        for (int i = 0; i < a.n_ + b.n_; ++i) r.w_[i] = 0;
        for (int i = 0; i < a.n_; ++i) {
            U128 c = 0;
            for (int j = 0; j < b.n_; ++j) {
                c += U128::mul64(a.w_[i], b.w_[j]) + r.w_[i + j];
                r.w_[i + j] = static_cast<std::uint64_t>(c);
                c >>= 64;
            }
            r.w_[i + b.n_] = static_cast<std::uint64_t>(c);
        }
        r.n_ = a.n_ + b.n_;
        r.neg_ = a.neg_ != b.neg_;
        r.trim();
        return r;
    }
    static int compare(const Big &a, const Big &b) {
        if (a.sign() != b.sign()) return a.sign() < b.sign() ? -1 : 1;
        const int m = cmp_mag(a, b);
        return a.neg_ ? -m : m;
    }
    // The value as 256 bits, two's complement (it must fit).
    // The value of four limbs in two's complement.
    static Big of_limbs(const std::uint64_t in[4]) {
        Big r;
        std::uint64_t m[4] = {in[0], in[1], in[2], in[3]};
        r.neg_ = (in[3] >> 63) != 0;
        if (r.neg_) {
            U128 c = 1;
            for (int i = 0; i < 4; ++i) {
                c += static_cast<std::uint64_t>(~m[i]);
                m[i] = static_cast<std::uint64_t>(c);
                c >>= 64;
            }
        }
        for (int i = 0; i < 4; ++i) r.w_[i] = m[i];
        r.n_ = 4;
        r.trim();
        return r;
    }
    // The nearest double, ties to even: the top 64 bits, the rest as a sticky bit.
    double to_double() const {
        if (n_ == 0) return 0.0;
        const int t = n_ - 1;
        const int lz = clz64(w_[t]);
        std::uint64_t m = w_[t] << lz;
        bool sticky = false;
        if (t > 0) {
            if (lz) m |= w_[t - 1] >> (64 - lz);
            sticky = lz ? (w_[t - 1] << lz) != 0 : w_[t - 1] != 0;
            for (int i = t - 2; i >= 0; --i) sticky = sticky || w_[i] != 0;
        }
        const double r = round_u64(m, sticky, 64 * t - lz);
        return neg_ ? -r : r;
    }

private:
    std::uint64_t w_[kLimbs] = {};
    int n_ = 0;
    bool neg_ = false;
    void trim() {
        while (n_ > 0 && w_[n_ - 1] == 0) --n_;
        if (n_ == 0) neg_ = false;
    }
    static int cmp_mag(const Big &a, const Big &b) {
        if (a.n_ != b.n_) return a.n_ < b.n_ ? -1 : 1;
        for (int i = a.n_ - 1; i >= 0; --i)
            if (a.w_[i] != b.w_[i]) return a.w_[i] < b.w_[i] ? -1 : 1;
        return 0;
    }
    static void add_mag(const Big &a, const Big &b, Big &r) {
        const int n = std::max(a.n_, b.n_);
        U128 c = 0;
        for (int i = 0; i < n; ++i) {
            c += static_cast<U128>(i < a.n_ ? a.w_[i] : 0) + (i < b.n_ ? b.w_[i] : 0);
            r.w_[i] = static_cast<std::uint64_t>(c);
            c >>= 64;
        }
        r.n_ = n;
        if (c) {
            if (n == kLimbs) throw std::logic_error("fitted walls: an exact sum exceeds its bound");
            r.w_[n] = static_cast<std::uint64_t>(c);
            r.n_ = n + 1;
        }
    }
    static void sub_mag(const Big &a, const Big &b, Big &r) {  // |a| >= |b|
        std::uint64_t borrow = 0;
        for (int i = 0; i < a.n_; ++i) {
            const std::uint64_t x = a.w_[i], y = i < b.n_ ? b.w_[i] : 0;
            const std::uint64_t d = x - y - borrow;
            borrow = (x < y || (x == y && borrow)) ? 1 : 0;
            r.w_[i] = d;
        }
        r.n_ = a.n_;
    }
};

// gcd of magnitudes, by the binary method (shifts and subtractions, no 128-bit division).
inline int ctz128(U128 x) {
    const auto lo = static_cast<std::uint64_t>(x);
    return lo ? ctz64(lo) : 64 + ctz64(static_cast<std::uint64_t>(x >> 64));
}
inline U128 gcd(U128 a, U128 b) {
    if (a == 0) return b;
    if (b == 0) return a;
    const int k = std::min(ctz128(a), ctz128(b));
    a >>= ctz128(a);
    while (b) {
        b >>= ctz128(b);
        if (a > b) std::swap(a, b);
        b -= a;
    }
    return a << k;
}
inline U128 mag(I128 x) { return x < 0 ? U128{0} - static_cast<U128>(x) : static_cast<U128>(x); }
// x / g for a g > 0 that divides x: x shifted right by g's trailing zero bits (exact, since
// 2^k divides x), times the inverse of g's odd part modulo 2^128 (Newton's iteration, which
// doubles the correct low bits at each step from 3: o o = 1 modulo 8 for an odd o).
inline I128 exact_div(I128 x, U128 g) {
    const int k = ctz128(g);
    const U128 o = g >> k;
    U128 inv = o;
    for (int i = 0; i < 6; ++i) inv = inv * (U128(2) - o * inv);
    return static_cast<I128>(static_cast<U128>(x >> k) * inv);
}
// The number of bits of n (0 for 0).
inline int bit_width(std::size_t n) {
    int b = 0;
    for (; n != 0; n >>= 1) ++b;
    return b;
}

inline std::uint64_t gcd64(std::uint64_t a, std::uint64_t b) {
    while (b) {
        const std::uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
// gcd of three magnitudes (in 64 bits where they fit).
inline U128 gcd3(I128 a, I128 b, I128 c) {
    const U128 x = mag(a), y = mag(b), z = mag(c);
    if ((x | y | z) >> 64) return gcd(gcd(x, y), z);
    return gcd64(gcd64(static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(y)), static_cast<std::uint64_t>(z));
}

// The integer value of a double that holds one.
inline I128 to_i128(double v) {
    if (std::abs(v) < 9.0e18) return static_cast<I128>(static_cast<std::int64_t>(v));
    int e = 0;
    const double m = std::frexp(v, &e);  // v = m 2^e, 1/2 <= |m| < 1, e > 53 here
    const auto mi = static_cast<std::int64_t>(std::ldexp(m, 53));
    return static_cast<I128>(mi) * (static_cast<I128>(1) << (e - 53));
}

// A signed integer of 256 bits, two's complement: the fast form of the predicates of a lattice
// point against a plane (n.q - d, below 2^189) and of a plane's offset from a lattice corner.
struct I256 {
    std::uint64_t w[4] = {0, 0, 0, 0};
    static I256 of(I128 x) {
        I256 r;
        r.w[0] = static_cast<std::uint64_t>(x);
        r.w[1] = static_cast<std::uint64_t>(static_cast<U128>(x) >> 64);
        const std::uint64_t fill = x < 0 ? ~0ull : 0ull;
        r.w[2] = r.w[3] = fill;
        return r;
    }
    // a b for |a| < 2^127, |b| < 2^63
    static I256 mul(I128 a, std::int64_t b) {
        const bool neg = (a < 0) != (b < 0);
        const U128 m = mag(a);
        const std::uint64_t bm = b < 0 ? 0ull - static_cast<std::uint64_t>(b) : static_cast<std::uint64_t>(b);
        const U128 lo = U128::mul64(static_cast<std::uint64_t>(m), bm);
        const U128 hi = U128::mul64(static_cast<std::uint64_t>(m >> 64), bm);
        I256 r;
        r.w[0] = static_cast<std::uint64_t>(lo);
        const U128 mid = (lo >> 64) + static_cast<std::uint64_t>(hi);
        r.w[1] = static_cast<std::uint64_t>(mid);
        const U128 top = (hi >> 64) + (mid >> 64);
        r.w[2] = static_cast<std::uint64_t>(top);
        r.w[3] = static_cast<std::uint64_t>(top >> 64);
        return neg ? -r : r;
    }
    friend I256 operator-(const I256 &a) {
        I256 r;
        U128 c = 1;
        for (int i = 0; i < 4; ++i) {
            c += static_cast<std::uint64_t>(~a.w[i]);
            r.w[i] = static_cast<std::uint64_t>(c);
            c >>= 64;
        }
        return r;
    }
    friend I256 operator+(const I256 &a, const I256 &b) {
        I256 r;
        U128 c = 0;
        for (int i = 0; i < 4; ++i) {
            c += static_cast<U128>(a.w[i]) + b.w[i];
            r.w[i] = static_cast<std::uint64_t>(c);
            c >>= 64;
        }
        return r;
    }
    friend I256 operator-(const I256 &a, const I256 &b) { return a + (-b); }
    static I256 mul64s(const I256 &a, std::int64_t b) {
        const bool na = (a.w[3] >> 63) != 0, nb = b < 0;
        const I256 m = na ? -a : a;
        const std::uint64_t bm = nb ? 0ull - static_cast<std::uint64_t>(b) : static_cast<std::uint64_t>(b);
        I256 r;
        U128 c = 0;
        for (int i = 0; i < 4; ++i) {
            c += U128::mul64(m.w[i], bm);
            r.w[i] = static_cast<std::uint64_t>(c);
            c >>= 64;
        }
        return na != nb ? -r : r;
    }
    int sign() const {
        if (w[3] >> 63) return -1;
        return (w[0] | w[1] | w[2] | w[3]) ? 1 : 0;
    }
    // Rounded to nearest, ties to even: the top 64 bits, the rest as a sticky bit.
    double to_double() const {
        if (w[3] >> 63) return -(-*this).to_double();
        int top = 3;
        while (top > 0 && w[top] == 0) --top;
        if (top == 0) {
            if (w[0] < (1ull << 53)) return static_cast<double>(static_cast<std::int64_t>(w[0]));
        }
        const int lz = w[top] ? clz64(w[top]) : 64;
        std::uint64_t m = lz ? (w[top] << lz) | (top > 0 ? w[top - 1] >> (64 - lz) : 0) : w[top];
        bool sticky = top > 0 && lz ? (w[top - 1] << lz) != 0 : (top > 0 && w[top - 1] != 0);
        for (int i = top - 2; i >= 0; --i) sticky = sticky || w[i] != 0;
        std::uint64_t keep = m >> 11;
        const std::uint64_t rest = m & 0x7FF;
        if (rest > 0x400 || (rest == 0x400 && (sticky || (keep & 1)))) ++keep;
        return std::ldexp(static_cast<double>(static_cast<std::int64_t>(keep)), 64 * top - lz + 11);
    }
    DD to_dd() const {
        if (w[3] >> 63) return -(-*this).to_dd();
        DD s(0.0);
        for (int i = 3; i >= 0; --i) {
            const DD l = dd_of_u64(w[i]);
            s = s + DD(std::ldexp(l.hi, 64 * i), std::ldexp(l.lo, 64 * i));
        }
        return s;
    }
};

// A small vector of ints held in place up to 14 entries (more on the heap), so that cutting
// the small polygons of a cube allocates nothing.
class Ints {
public:
    static constexpr std::size_t kInline = 14;
    Ints() = default;
    Ints(const Ints &o) : n_(o.n_), heap_(o.heap_) {
        if (n_ <= kInline) std::copy(o.in_, o.in_ + n_, in_);
    }
    Ints(Ints &&o) noexcept : n_(o.n_), heap_(std::move(o.heap_)) {
        if (n_ <= kInline) std::copy(o.in_, o.in_ + n_, in_);
        o.n_ = 0;
        o.heap_.clear();
    }
    Ints &operator=(const Ints &o) {
        if (this != &o) {
            n_ = o.n_;
            heap_ = o.heap_;
            if (n_ <= kInline) std::copy(o.in_, o.in_ + n_, in_);
        }
        return *this;
    }
    Ints &operator=(Ints &&o) noexcept {
        if (this != &o) {
            n_ = o.n_;
            heap_ = std::move(o.heap_);
            if (n_ <= kInline) std::copy(o.in_, o.in_ + n_, in_);
            o.n_ = 0;
            o.heap_.clear();
        }
        return *this;
    }
    void push_back(int x) {
        if (n_ == kInline) heap_.assign(in_, in_ + n_);
        if (n_ >= kInline) heap_.push_back(x);
        else in_[n_] = x;
        ++n_;
    }
    void clear() {
        n_ = 0;
        heap_.clear();
    }
    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    int *begin() { return n_ > kInline ? heap_.data() : in_; }
    int *end() { return begin() + n_; }
    const int *begin() const { return n_ > kInline ? heap_.data() : in_; }
    const int *end() const { return begin() + n_; }
    int &operator[](std::size_t i) { return begin()[i]; }
    int operator[](std::size_t i) const { return begin()[i]; }

private:
    std::size_t n_ = 0;
    int in_[kInline] = {};
    std::vector<int> heap_;
};

}  // namespace cut_detail

class CutCells {
public:
    using I128 = cut_detail::I128;
    using Big = cut_detail::Big;
    static cut_detail::U128 gcd3(I128 a, I128 b, I128 c) { return cut_detail::gcd3(a, b, c); }
    using Ints = cut_detail::Ints;

    // What a cell of the grid is in the fit: solid, air of the source's component, or air in
    // the matched layer.
    static constexpr std::uint8_t kSolid = 0, kAir = 1, kLayer = 2;
    // Why an input was refused (Stats::refused), in the order the tests run (DESIGN 5).
    enum : int {
        kRefusedNot = 0,
        kRefusedFrame = 1,          // R0: a face's vertex 2^24 cells or more from the frame's middle
        kRefusedGridFar = 2,        // the grid's doubles do not separate a centre from its faces
        kRefusedSourceOutside = 3,  // the source in a cell on the grid's edge or beyond
        kRefusedSourceInZone = 4,   // the source inside a zone of (i)
        kRefusedThinLayer = 5,      // (ii)
        kRefusedSlit = 6,           // (i)
        kRefusedPocket = 7,         // the source's component holds no centre
        kRefusedLayer = 8,          // a face in a cube of the matched layer with air in it
        kRefusedLayerReach = 9,     // T-4
        // (10 and 11 are not used.)
        kRefusedNextToLayer = 12,   // a cube next to a matched-layer cell holds faces and air
        kRefusedWidths = 13,        // a cell's lattice width off 1 by more than 2^-24 (set by the solver)
        // The source refusals of DESIGN 5.5 (no cell of its stencil; the stencil past the
        // grid's interior; the source in the matched layer) are made by the solver's fitted
        // stencil (fdtd.hpp, FittedStencil::points) and thrown as std::invalid_argument with
        // their messages; they carry no code here.
    };
    // The width of refusal (i)'s zones, in cells (see 5. of the head comment).
    static constexpr double kGap = 0.01;
    // The directions to a cell's neighbours, in the wave solver's order: the six axial
    // ones, then the twelve edge-diagonal ones.
    static constexpr int kSteps[18][3] = {
        {-1, 0, 0},  {1, 0, 0},  {0, -1, 0},  {0, 1, 0},  {0, 0, -1}, {0, 0, 1},
        {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},  {1, 1, 0},  {-1, 0, -1}, {1, 0, -1},
        {-1, 0, 1},  {1, 0, 1},  {0, -1, -1}, {0, 1, -1}, {0, -1, 1},  {0, 1, 1}};

    // The solver's grid: its origin, cell and shape, and its own doubles of the cells'
    // faces and centres per axis (face_at[a][i] the plane of the - face of cell i along a,
    // n[a] + 1 of them; centre_at[a][i] the centre of cell i, n[a] of them), the depth of
    // the matched layer in cells (0: none).
    struct Grid {
        Vec3 origin;
        double cell = 0.0;
        std::array<std::size_t, 3> n{};
        std::array<std::vector<double>, 3> face_at, centre_at;
        std::size_t layer = 0;
    };

    struct Stats {
        std::size_t faces = 0;            // faces given to the fit (reaching the grid)
        std::size_t degenerate_faces = 0; // of them, those whose three lattice points are collinear
        double degenerate_area = 0.0;     // their area as given (m^2)
        int quantum_bits = 0;             // s: the lattice is 2^-s m
        std::size_t planes = 0;           // the model's planes (groups of faces on one exact plane)
        std::size_t membrane_faces = 0;   // faces lying exactly on a grid plane
        std::size_t cubes = 0;            // cubes holding faces
        std::size_t largest_planes = 0;   // most planes in one cube
        std::size_t leaves = 0;           // pieces of all cubes' trees
        std::size_t regions = 0;          // regions of cubes holding faces
        std::size_t squares = 0;          // squares cut down the trees
        std::size_t ties = 0;             // regions given to one of owners sharing the most area
        std::size_t leaks = 0;            // regions of the air holding a solid cell's centre (0 by construction)
        double dropped_area = 0.0;        // open area to no owner, or between owners not linked
        double diagonal_area = 0.0;       // open area between edge-diagonal owners (counted once)
        double far_volume = 0.0;          // air owned by a cell whose cube does not touch its cube
        std::size_t farthest = 0;         // most cells between an owner and a cube of its air
        std::size_t air_cells = 0;        // cells of the source's component (layer cells included)
        double air_volume = 0.0;          // the air cells' air (cells^3, every cell measured between its planes on Q)
        double balance = 0.0;             // |face_area - covered_area| / max(1, face_area): a measure, 0 up to rounding
        std::size_t layer_cells = 0;      // of them, in the matched layer
        double edge_area = 0.0;           // open area of the air towards the grid's edge (cell faces)
        std::size_t sealed_components = 0;  // components without a centre, closed (not the air)
        double sealed_volume = 0.0;
        std::size_t other_components = 0;   // components with a centre, other than the air
        double other_volume = 0.0;          // their cut volume
        std::size_t edge_pockets = 0;       // components without a centre open only to the grid's edge
        double edge_pocket_volume = 0.0;
        std::size_t unowned_regions = 0;    // regions of the air without an owner (refused when not 0)
        double unowned_volume = 0.0;
        double face_area = 0.0;     // the faces' parts in cubes holding faces and in squares (membranes), cell faces
        double covered_area = 0.0;  // the same, as the facets and squares cover it (each face counted once)
        double overlap_area = 0.0;  // covered more than once (exactly coplanar faces)
        // The partition: the area of every polygon whose pieces the fit forms (each facet
        // between two leaves of a cube, each square), and on each of its two sides the
        // pieces' bare area plus the walls the fit gives that side, cell faces.
        double partition_area = 0.0, partition_plus = 0.0, partition_minus = 0.0;
        // max over the two sides of |partition_area - partition_side| / max(1, partition_area):
        // a measure, 0 up to rounding
        double partition = 0.0;
        // The self-checks (K1 to K8): each one's largest residual relative to its threshold (a
        // comparison fails above 1; an exact check's residual is 0 when it passes) and its count
        // of failed comparisons. A failed check throws std::logic_error, so a fit that is returned
        // has every count 0. K1 walls per region and material of each cube; K2 the volume of each
        // cube; K3 the copy of the cubes' regions; K4 the rows' volumes and walls; K5 the symmetry
        // of the open areas between rows; K6 the open area of the squares' pieces, by what sum_up
        // makes of it; K7 the regions on the two sides of each square's piece; K8 the owners.
        double check_wall = 0.0, check_volume = 0.0, check_copy = 0.0, check_rows = 0.0;
        double check_open_symmetry = 0.0, check_open = 0.0, check_piece_sides = 0.0, check_owners = 0.0;
        std::size_t check_wall_failed = 0, check_volume_failed = 0, check_copy_failed = 0, check_rows_failed = 0;
        std::size_t check_open_symmetry_failed = 0, check_open_failed = 0, check_piece_sides_failed = 0;
        std::size_t check_owners_failed = 0;
        // Decisions that read a measure, counted where the measure lies near the threshold: the
        // owner rule's comparisons within 1e-13 of its band's edge (1e-12), and the open areas of
        // slivers within 1e-12 of the gate (1e-9 cell faces).
        std::size_t owner_near_band = 0, sliver_near_gate = 0;
        // In the checking build: whether a floating-point underflow was raised during the fit.
        int underflow_raised = 0;
        std::size_t exact_calls = 0;     // signs decided by the exact stage, in cubes and squares
        std::size_t exact_t_calls = 0;   // of those, signs of a point of three planes (T)
        std::size_t refined_points = 0;  // points whose approximation was made from exact coordinates
        int refused = 0;                 // why the input was refused (kRefused...), 0 when fitted
        std::size_t boundary_intervals = 0;  // odd intervals of the mesh's mod-2 boundary
        std::size_t free_pieces = 0;         // of them, those with a part on no other face
        double free_length = 0.0;            // the length of those parts (cells)
        std::size_t zone_cubes = 0;          // cubes cut again by the zones of (i)
        std::size_t layer_pairs = 0;         // pairs of faces tested exactly by (ii)
        std::size_t layer_candidates = 0;    // of them, nearly parallel, overlapping and within 7/4 kGap
        double seconds = 0.0;
        std::array<double, 4> stage{};   // faces, cubes, squares, sums
    };

    // The mesh exactly as given: vertices (n x 3 doubles) and faces (m x 3 indices into them,
    // checked by the caller), the same arrays the BVH was built from.
    struct Mesh {
        const double *vertices = nullptr;
        std::size_t vertex_count = 0;
        const std::int64_t *faces = nullptr;
        std::size_t face_count = 0;
        std::array<Vec3, 3> corners(std::size_t f) const {
            std::array<Vec3, 3> p;
            for (int k = 0; k < 3; ++k) {
                const double *v = vertices + 3 * static_cast<std::size_t>(faces[3 * f + static_cast<std::size_t>(k)]);
                p[static_cast<std::size_t>(k)] = Vec3{v[0], v[1], v[2]};
            }
            return p;
        }
    };

    CutCells(const Mesh &mesh, const std::int64_t *face_material, std::size_t materials, const Grid &grid,
             const Vec3 &source, std::size_t threads)
        : origin_(grid.origin), cell_(grid.cell), n_(grid.n), sy_(grid.n[0]), sz_(grid.n[0] * grid.n[1]),
          layer_(grid.layer), materials_(materials) {
        for (int a = 0; a < 3; ++a) {
            const std::size_t m = n_[static_cast<std::size_t>(a)];
            if (grid.face_at[static_cast<std::size_t>(a)].size() != m + 1 ||
                grid.centre_at[static_cast<std::size_t>(a)].size() != m)
                throw std::invalid_argument("the grid's doubles: n + 1 faces and n centres per axis");
        }
        cut_detail::require_arithmetic();
#ifdef ACOUSTIC_SPACES_CUT_CHECK
        std::feclearexcept(FE_UNDERFLOW);
#endif
        const auto start = std::chrono::steady_clock::now();
        auto lap = start;
        auto mark = [&](int k) {
            const auto now = std::chrono::steady_clock::now();
            stats_.stage[static_cast<std::size_t>(k)] = std::chrono::duration<double>(now - lap).count();
            lap = now;
        };
        try {
            frame(mesh, grid);
            lattice_grid(grid);
            faces_global(mesh, face_material);
            boundary();
            layer_candidates(threads);
            std::vector<std::pair<std::size_t, std::size_t>> pairs = faces_into_cubes(threads);
            membranes(threads);
            mark(0);
            cut_cubes(pairs, zone_cubes(), threads);
            mark(1);
            cut_squares(threads);
            mark(2);
            air(source);
            sum_up();
            sides();
        } catch (const Refusal &r) {
            stats_.refused = r.code;
            refusal_ = r.message;
            clear_rows();
        }
#ifdef ACOUSTIC_SPACES_CUT_CHECK
        stats_.underflow_raised = underflow_ || std::fetestexcept(FE_UNDERFLOW) ? 1 : 0;
#endif
        stats_.exact_calls = exact_e_ + exact_t_;
        stats_.exact_t_calls = exact_t_;
        stats_.refined_points = refined_;
        mark(3);
        stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        cubes_.clear();
        cubes_.shrink_to_fit();
        squares_.clear();
        squares_.shrink_to_fit();
        regions_.clear();
        regions_.shrink_to_fit();
    }

    // One row per air cell that owns cut air or lies next to an edge cell (shares a face with
    // one), ascending: the cell's index, its air volume (cells; a row that owns nothing cut
    // holds its own volume between its planes), open area per direction of kSteps (n x 18,
    // cell faces), wall area per material and last to the grid's edge (n x (materials + 1),
    // cell faces; the edge column holds the open pieces in the edge plane between an edge cell
    // and a region the row owns, in its own cube or in a neighbour's, and the whole faces its
    // cell shares with edge cells where no square was cut, each w_b w_c of its lattice widths),
    // a column of zeros kept for the binding (leak_area: no region of the air holds a solid
    // centre), and the cut air's thickness per axial face (n x 6, cells).
    const std::vector<std::int64_t> &cells() const { return row_cell_; }
    const std::vector<double> &air_volume() const { return air_volume_; }
    const std::vector<double> &open_area() const { return open_area_; }
    const std::vector<double> &wall_area() const { return wall_area_; }
    const std::vector<double> &leak_area() const { return leak_area_row_; }
    const std::vector<double> &sliver() const { return sliver_; }
    // Each cell's width along axis a on the lattice (cells): an air cell that owns nothing cut
    // holds width_x width_y width_z cells^3 of air.
    const std::vector<double> &widths(int a) const { return width_[static_cast<std::size_t>(a)]; }
    // Each cell's kind in the fit (kSolid, kAir, kLayer), x fastest.
    const std::vector<std::uint8_t> &air() const { return type_; }
    // Whether each side of the grid (-x, +x, -y, +y, -z, +z) is open, decided from the cells'
    // kinds and exact open flags, never from an area: a side is open when an air cell lies next
    // to an edge cell on that side, when an open piece of a square in that side's edge plane
    // joins a region of the source's air to the edge cell, or when a cell of the matched layer
    // lies within the layer's width of that side. A refused fit has no open side.
    const std::array<bool, 6> &open_sides() const { return open_sides_; }
    // Whether any side of the grid is open.
    bool open() const {
        for (const bool s : open_sides_)
            if (s) return true;
        return false;
    }
    std::size_t materials() const { return materials_; }
    const Stats &stats() const { return stats_; }
    // A refused input: its code (Stats::refused, 0 when fitted) and its message; then the
    // rows are empty and every cell is kSolid.
    bool refused() const { return stats_.refused != kRefusedNot; }
    const std::string &refusal() const { return refusal_; }
    // Throws std::invalid_argument with the message when the input was refused.
    void check() const {
        if (refused()) throw std::invalid_argument(refusal_);
    }
    // The wave solver's condition on the grid (refusal 13): every cell's width on the lattice
    // lies within 2^-24 of a whole cell, decided exactly on the integers of the grid's planes,
    // |G_i+1 - G_i - S| 2^24 <= S with S = cell 2^s the cell in quanta. The solver steps a plain
    // cell with the stencil's own weight and a row with its lattice widths, so a link between
    // the two differs on its sides by at most 2^-24 of its weight, 2^6 below the update's margin
    // of 2^-18. A fit already refused keeps its refusal. CutFit, which only measures, does not
    // apply it.
    void require_whole_widths() {
        if (refused()) return;
        int e = 0;
        const double m = std::frexp(S_, &e);
        const I128 M = static_cast<I128>(static_cast<std::int64_t>(std::ldexp(m, 53)));
        const int k = e - 53;           // S = M 2^k exactly
        const int up = k < 0 ? -k : 0;  // both sides times 2^up, so that both are integers
        const I128 S = k >= 0 ? M << k : M;
        const I128 limit = S >> 24;     // |d - S| 2^24 > S exactly when |d - S| > floor(S / 2^24)
        const I128 huge = static_cast<I128>(1) << 100;
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t i = 0; i < n_[a]; ++i) {
                I128 d = G_[a][i + 1] - G_[a][i];
                bool off = d.negative() || (up > 0 && d >= (huge >> up));
                if (!off) {
                    d = d << up;
                    I128 diff = d - S;
                    if (diff.negative()) diff = -diff;
                    off = diff > limit;
                }
                if (!off) continue;
                static const char *const axis[3] = {"x", "y", "z"};
                const double at = origin_[static_cast<int>(a)] + static_cast<double>(i) * cell_;
                stats_.refused = kRefusedWidths;
                refusal_ = std::string("fitted walls: the grid lies too far from the origin of its coordinates for "
                                       "its cells to keep their width: the cell from ") +
                           axis[a] + " = " + std::to_string(at) + " m is " + exact_text(width_[a][i]) +
                           " cells wide on the fit's lattice, more than 2^-24 off a whole cell, where the solver's "
                           "plain cells take a whole one. Move the model and the grid nearer the origin of the "
                           "coordinates, or use a larger cell.";
                clear_rows();
                return;
            }
    }

    // ---- the points (T-1, DESIGN 4.7.2 and 4.7.3) ------------------------------------------

    // A point on Q: its lattice coordinates, its centre bracket per axis (j: the largest with
    // C_j <= Q, -1 below the first centre), whether it is tied (Q = C_j: the point lies on that
    // plane of centres and, moved by delta, just past it), and its fractions
    // lo = (Q - C_j) / (C_j+1 - C_j) and hi = (C_j+1 - Q) / (C_j+1 - C_j), each an exact ratio
    // rounded (0 and 1 where j has no centre above).
    struct Located {
        I128 q[3] = {0, 0, 0};
        long long j[3] = {0, 0, 0};
        bool tied[3] = {false, false, false};
        double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
    };
    // n / d correctly rounded, for 0 <= n <= d < 2^63.
    static double quotient(I128 n, I128 d) {
        const I128 two53 = static_cast<I128>(1) << 53;
        if (d < two53) return static_cast<double>(static_cast<std::int64_t>(n)) / static_cast<double>(static_cast<std::int64_t>(d));
        if (n == d) return 1.0;
        if (n == 0) return 0.0;
        auto bits = [](I128 x) { int b = 0; while (x != 0) { x >>= 1; ++b; } return b; };
        const int t = 63 + bits(d) - bits(n);  // n 2^t / d lies in [2^62, 2^64)
        const I128 num = n << t;
        const std::uint64_t q = static_cast<std::uint64_t>(num / d);
        const bool sticky = num % d != 0;
        const int lz = cut_detail::clz64(q);
        const std::uint64_t m = q << lz;
        std::uint64_t keep = m >> 11;
        const std::uint64_t rest = m & 0x7FF;
        if (rest > 0x400 || (rest == 0x400 && (sticky || (keep & 1)))) ++keep;
        return std::ldexp(static_cast<double>(static_cast<std::int64_t>(keep)), 11 - lz - t);
    }
    Located locate(const Vec3 &p) const {
        Located L;
        for (int a = 0; a < 3; ++a) {
            const std::size_t A = static_cast<std::size_t>(a);
            L.q[a] = quantize(p[a], a);
            const std::vector<I128> &C = Cq_[A];
            const auto it = std::upper_bound(C.begin(), C.end(), L.q[a]);
            L.j[a] = static_cast<long long>(it - C.begin()) - 1;
            if (L.j[a] >= 0) L.tied[a] = C[static_cast<std::size_t>(L.j[a])] == L.q[a];
            if (L.j[a] >= 0 && L.j[a] + 1 < static_cast<long long>(C.size())) {
                const I128 c0 = C[static_cast<std::size_t>(L.j[a])], c1 = C[static_cast<std::size_t>(L.j[a] + 1)];
                L.lo[a] = quotient(L.q[a] - c0, c1 - c0);
                L.hi[a] = quotient(c1 - L.q[a], c1 - c0);
            }
        }
        return L;
    }
    // The lattice centre of cell i.
    void centre_q(std::size_t i, I128 C[3]) const {
        const auto p = at(i);
        for (int a = 0; a < 3; ++a) C[a] = Cq_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)]];
    }
    // The faces (indices into the fit's faces, not degenerate; membranes included) whose closed
    // box on the lattice meets the closed box of the lattice points pts (exact, integers): a
    // superset of the faces a way between points of that box can cross, since a face crossed
    // by a way moved by (eps, eps^2, eps^3) meets the closed segment. In model order.
    std::vector<std::size_t> faces_near(const I128 (*pts)[3], std::size_t count) const {
        I128 lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = hi[a] = pts[0][a];
            for (std::size_t k = 1; k < count; ++k) {
                lo[a] = std::min(lo[a], pts[k][a]);
                hi[a] = std::max(hi[a], pts[k][a]);
            }
        }
        std::vector<std::size_t> out;
        for (std::size_t t = 0; t < faces_.size(); ++t) {
            const FaceX &F = faces_[t];
            if (F.plane < 0) continue;  // membranes (faces on grid planes) are faces too
            bool meets = true;
            for (int a = 0; a < 3 && meets; ++a) meets = F.qhi[a] >= lo[a] && F.qlo[a] <= hi[a];
            if (meets) out.push_back(t);
        }
        return out;
    }
    // The way from X + delta to Y + delta, delta = (eps, eps^2, eps^3): -1 when it crosses no
    // face of `near`, else the model index of the first face of `near` it crosses (4.7.2).
    std::int64_t way(const I128 X[3], const I128 Y[3], const std::vector<std::size_t> &nearby_faces) const {
        ++ways_;
        std::int64_t x[3], y[3];
        for (int a = 0; a < 3; ++a) {
            x[a] = small(X[a]);
            y[a] = small(Y[a]);
        }
        for (const std::size_t t : nearby_faces) {
            if (crosses(faces_[t], x, y)) return static_cast<std::int64_t>(faces_[t].face);
        }
        return -1;
    }
    // Whether lattice point X lies in the grid's closed box [G_0, G_n] per axis (exact). Every
    // such point is inside the frame (|Q| < 2^61), so way() and on_face() can take it; a point
    // outside it has no cell of any stencil.
    bool in_grid_box(const I128 X[3]) const {
        for (int a = 0; a < 3; ++a) {
            const std::vector<I128> &G = G_[static_cast<std::size_t>(a)];
            if (X[a] < G.front() || X[a] > G.back()) return false;
        }
        return true;
    }
    // Whether lattice point X lies on a face: on its plane and in its closed triangle (exact);
    // false for a point outside the grid's box.
    bool on_face(const I128 X[3]) const {
        if (!in_grid_box(X)) return false;
        const I128 pts[1][3] = {{X[0], X[1], X[2]}};
        std::int64_t x[3];
        for (int a = 0; a < 3; ++a) x[a] = small(X[a]);
        for (const std::size_t t : faces_near(pts, 1)) {
            const FaceX &F = faces_[t];
            I128 n[3];
            normal_of(F, n);
            const W256 v = W256::mul(n[0], x[0] - F.q[0][0]) + W256::mul(n[1], x[1] - F.q[0][1]) +
                           W256::mul(n[2], x[2] - F.q[0][2]);
            if (v.sign() != 0) continue;
            int ax = 0;
            for (int a = 1; a < 3; ++a)
                if (mag(n[a]) > mag(n[ax])) ax = a;
            const int u = (ax + 1) % 3, w = (ax + 2) % 3;
            int pos = 0, neg = 0;
            for (int k = 0; k < 3; ++k) {
                const std::int64_t *P = F.q[k], *R = F.q[(k + 1) % 3];
                const I128 o = static_cast<I128>(R[u] - P[u]) * (x[w] - P[w]) -
                               static_cast<I128>(R[w] - P[w]) * (x[u] - P[u]);
                pos += o > 0;
                neg += o < 0;
            }
            if (pos == 0 || neg == 0) return true;
        }
        return false;
    }
    // Ways decided, and (way, face) pairs past the box filter W0 (all exact: 4.7.2's signs are
    // taken by the 256-bit stage directly, with no float filter).
    std::size_t ways() const { return ways_; }
    const std::array<std::size_t, 3> &shape() const { return n_; }
    double cell() const { return cell_; }
    // The matched layer's depth (cells) and cell i's centre (m), for messages.
    std::size_t layer() const { return layer_; }
    Vec3 cell_centre(std::size_t i) const { return centre_of(i); }
    std::size_t ways_exact() const { return ways_exact_; }

private:
    using W256 = cut_detail::I256;
    mutable std::size_t ways_ = 0, ways_exact_ = 0;
    static cut_detail::U128 mag(I128 v) {
        return v < 0 ? static_cast<cut_detail::U128>(0) - static_cast<cut_detail::U128>(v) : static_cast<cut_detail::U128>(v);
    }
    template <typename Face>
    static void normal_of(const Face &F, I128 n[3]) {
        const std::int64_t *A = F.q[0], *B = F.q[1], *C = F.q[2];
        const I128 u[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]}, v[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
        n[0] = u[1] * v[2] - u[2] * v[1];
        n[1] = u[2] * v[0] - u[0] * v[2];
        n[2] = u[0] * v[1] - u[1] * v[0];
    }
    // The sign of c + d . delta: that of c when it is not zero, else of the first nonzero d.
    static int perturbed(int c, const I128 d[3]) {
        if (c != 0) return c;
        for (int a = 0; a < 3; ++a)
            if (d[a] != 0) return d[a] > 0 ? 1 : -1;
        return 0;
    }
    // W0 to W2 of 4.7.2 for one triangle.
    template <typename Face>
    bool crosses(const Face &F, const std::int64_t x[3], const std::int64_t y[3]) const {
        const std::int64_t *A = F.q[0], *B = F.q[1], *C = F.q[2];
        for (int a = 0; a < 3; ++a) {  // W0
            const std::int64_t tmin = std::min({A[a], B[a], C[a]}), tmax = std::max({A[a], B[a], C[a]});
            if (std::min(x[a], y[a]) >= tmax || std::max(x[a], y[a]) < tmin) return false;
        }
        ++ways_exact_;
        I128 n[3];
        normal_of(F, n);
        auto plane = [&](const std::int64_t z[3]) {  // W1
            const W256 v = W256::mul(n[0], z[0] - A[0]) + W256::mul(n[1], z[1] - A[1]) + W256::mul(n[2], z[2] - A[2]);
            return perturbed(v.sign(), n);
        };
        if (plane(x) == plane(y)) return false;
        const std::int64_t s[3] = {y[0] - x[0], y[1] - x[1], y[2] - x[2]};
        const std::int64_t *T[3] = {A, B, C};
        int first = 0;
        for (int k = 0; k < 3; ++k) {  // W2: tau = det[U - X, V - X, s] - delta . m
            const std::int64_t *U = T[k], *V = T[(k + 1) % 3];
            const std::int64_t a[3] = {U[0] - x[0], U[1] - x[1], U[2] - x[2]};
            const std::int64_t b[3] = {V[0] - x[0], V[1] - x[1], V[2] - x[2]};
            const I128 bs[3] = {static_cast<I128>(b[1]) * s[2] - static_cast<I128>(b[2]) * s[1],
                                static_cast<I128>(b[2]) * s[0] - static_cast<I128>(b[0]) * s[2],
                                static_cast<I128>(b[0]) * s[1] - static_cast<I128>(b[1]) * s[0]};
            const W256 det = W256::mul(bs[0], a[0]) + W256::mul(bs[1], a[1]) + W256::mul(bs[2], a[2]);
            const std::int64_t e[3] = {V[0] - U[0], V[1] - U[1], V[2] - U[2]};
            const I128 m[3] = {-(static_cast<I128>(e[1]) * s[2] - static_cast<I128>(e[2]) * s[1]),
                               -(static_cast<I128>(e[2]) * s[0] - static_cast<I128>(e[0]) * s[2]),
                               -(static_cast<I128>(e[0]) * s[1] - static_cast<I128>(e[1]) * s[0])};
            const int tau = perturbed(det.sign(), m);
            if (tau == 0) throw std::logic_error("fitted walls: a way's edge sign is zero after the perturbation");
            if (k == 0) first = tau;
            else if (tau != first) return false;
        }
        return true;
    }
    static constexpr std::int64_t kNone = -1, kEdge = -2, kZone = -3;
    struct Refusal {
        int code;
        std::string message;
    };
    [[noreturn]] static void refuse(int code, std::string message) { throw Refusal{code, std::move(message)}; }
    std::string refusal_;

    // ---- planes and points ---------------------------------------------------------------

    // A plane n.x = d with integer n (primitive, its first nonzero component positive) and d,
    // on the lattice; un = n / len, len = |n|, both as doubles, for the filter.
    struct Plane {
        I128 n[3] = {0, 0, 0};
        Big d;
        cut_detail::I256 d4;  // d again, in 256 bits
        double un[3] = {0.0, 0.0, 0.0};
        double len = 0.0;
        cut_detail::DD lun[3];
        cut_detail::DD llen;
    };
    // A plane in a frame (see Table): its offset from the frame's corner, in cells along un.
    struct PlaneLoc {
        double un[3];
        double d;
        cut_detail::DD lun[3], ld;  // the same in double-double, for computed points
    };
    // A point: a lattice point (q) or the meet of three planes (p), with its approximation u
    // in cells from its table's corner and the bound r on u's error (each coordinate).
    // A lattice point's planes are those it was made on (for incidence), -1 where none.
    struct Vtx {
        int p[3] = {-1, -1, -1};
        std::int64_t q[3] = {0, 0, 0};
        bool lattice = false;
        int wid = -1;  // a welded vertex of the mesh: its id
        int wa = -1, wb = -1;  // two welded vertices whose line the point lies on (or -1)
        double u[3] = {0.0, 0.0, 0.0};
        double r = 0.0;
        int ex = -1;  // its exact homogeneous coordinates in the table, once made
    };

    // The points of one cube (or square), in the frame of a lattice corner C, scaled by the
    // cell: u = (x - C) / S, S = h 2^s quanta per cell. Every decision about a point is
    // side(), an exact sign; the approximations only filter and measure.
    class Table {
    public:
        Table(const CutCells &cc, const std::int64_t C[3]) : cc_(cc) {
            for (int a = 0; a < 3; ++a) C_[a] = C[a];
            v.reserve(256);
            // The plane-to-slot index: the thread's shared one (all -1 between tables), or an
            // own one for a table made while another on this thread is alive.
            if (!pool_busy()) {
                pool_busy() = true;
                slot_ = &pool();
            } else {
                slot_ = &own_;
            }
            if (slot_->size() < cc_.planes_.size()) slot_->resize(cc_.planes_.size(), -1);
        }
        ~Table() {
            for (const int p : ids_) (*slot_)[static_cast<std::size_t>(p)] = -1;
            if (slot_ == &pool()) pool_busy() = false;
            cc_.exact_e_ += exact_calls - exact_t;
            cc_.exact_t_ += exact_t;
            cc_.refined_ += refined;
        }
        Table(const Table &) = delete;
        Table &operator=(const Table &) = delete;
        std::vector<Vtx> v;
        std::size_t exact_calls = 0, exact_t = 0, refined = 0;

        int lattice(const std::int64_t q[3], int p0, int p1, int p2, int wid = -1) {
            Vtx x;
            x.lattice = true;
            x.wid = wid;
            x.p[0] = p0;
            x.p[1] = p1;
            x.p[2] = p2;
            double m = 0.0;
            for (int a = 0; a < 3; ++a) {
                x.q[a] = q[a];
                x.u[a] = static_cast<double>(q[a] - C_[a]) / cc_.S_;
                m = std::max(m, std::abs(x.u[a]));
            }
            x.r = 3e-16 * m;
            v.push_back(x);
            return static_cast<int>(v.size()) - 1;
        }

        // The sign of n.x - d for point k against plane p: +1, -1, or 0 exactly on it; s and b
        // the filter's value (cells) and the bound on its error.
        // A sign the filter certifies is the exact sign, so incidence is tried only where the
        // filter cannot decide (an incident point's filter value is within its bound).
        int side(int k, int p, double &s, double &b) {
            const Vtx &x = v[static_cast<std::size_t>(k)];
            if (x.p[0] == p || x.p[1] == p || x.p[2] == p) {
                s = 0.0;
                b = 0.0;
                return 0;
            }
            const PlaneLoc &L = loc(p);  // read before any other loc() call, which may move it
            const double a0 = L.un[0] * x.u[0], a1 = L.un[1] * x.u[1], a2 = L.un[2] * x.u[2];
            s = a0 + a1 + a2 - L.d;
            b = 2.0 * x.r + 4e-15 * (std::abs(a0) + std::abs(a1) + std::abs(a2) + std::abs(L.d)) + 1e-300;
#ifndef ACOUSTIC_SPACES_CUT_CHECK
            if (s > b) return 1;
            if (s < -b) return -1;
#endif
            if ((x.wid >= 0 && cc_.through(x.wid, p)) || (x.wa >= 0 && cc_.through(x.wa, p) && cc_.through(x.wb, p))) {
#ifdef ACOUSTIC_SPACES_CUT_CHECK
                if (exact_side(k, p) != 0) {
                    ++cc_.check_contradictions_;
                    std::fprintf(stderr, "contradiction: point %d incident to plane %d by the mesh, not exactly\n", k, p);
                }
#endif
                s = 0.0;
                b = 0.0;
                return 0;
            }
#ifndef ACOUSTIC_SPACES_CUT_CHECK
            return exact_side(k, p);
#else
            const int e = exact_side(k, p);
            if ((s > b && e != 1) || (s < -b && e != -1)) {
                ++cc_.check_contradictions_;
                if (cc_.check_contradictions_ <= 5)
                    std::fprintf(stderr, "contradiction: point %d (lattice %d, planes %d %d %d, r %.3g) plane %d: filter %.17g bound %.3g exact %d\n",
                                 k, int(x.lattice), x.p[0], x.p[1], x.p[2], x.r, p, s, b, e);
            }
            return e;
#endif
        }
        int side(int k, int p) {
            double s, b;
            return side(k, p, s, b);
        }

        // The axis of plane p where p is an axis plane x_a = g of the lattice (normal e_a, an
        // offset g below 2^63 in magnitude), with g; -1 otherwise.
        int axis_plane(int p, std::int64_t &g) const {
            const Plane &P = cc_.planes_[static_cast<std::size_t>(p)];
            int a = -1;
            for (int c = 0; c < 3; ++c) {
                if (P.n[c] == 0) continue;
                if (a >= 0 || P.n[c] != 1) return -1;
                a = c;
            }
            const std::uint64_t fill = (P.d4.w[0] >> 63) ? ~std::uint64_t{0} : 0;
            if (P.d4.w[1] != fill || P.d4.w[2] != fill || P.d4.w[3] != fill) return -1;
            g = static_cast<std::int64_t>(P.d4.w[0]);
            return a;
        }

        // The point of planes p0, p1 and p2 (which meet in one point). Its coordinates u are
        // made by the first of these kinds that applies, in this order (a point of two kinds gets
        // different last bits from the two, so the order is part of the result):
        //   1. an end of a welded edge whose line lies on two of the planes, the end lying on
        //      the third: that end, a lattice point;
        //   2. three axis planes of three different axes: the lattice point of their offsets;
        //   3. a point of the line of a welded edge A B (on two of the planes) on the third
        //      plane P, not at an end: the exact numerators N_c = den (A_c - C_c) + num (B_c - A_c)
        //      and denominator den, with num = d_P - n_P . A and den = n_P . (B - A), u_c =
        //      R(N_c) / (R(den) S); where P is an axis plane, its coordinate is R(g - C_a) / S;
        //   4. two axis planes of different axes a and b, on no welded line, the third plane
        //      reaching the third axis c: u_a, u_b from the offsets, u_c = R(N) / (R(n_c) S)
        //      with N = d - n_a g_a - n_b g_b - n_c C_c;
        //   5. any other: Cramer's rule in double-double, with a bound on its error made from
        //      bounds on every rounding (a unit normal's component within 3 e, an offset within
        //      6 e, each operation within e, e = 2^-100, each double-double operation lying
        //      within a quarter of e); accepted where that bound is at most 2^-60 times the
        //      larger of 1 and the point's largest coordinate, and made from its exact
        //      coordinates otherwise.
        // R is the conversion of an exact integer to the nearest double, and S the lattice's
        // quanta per cell. r bounds each coordinate's error: 3e-16 max|u| for kinds 1 and 2,
        // 8 2^-53 max|u| for kinds 3 and 4, and for kind 5 the accepted bound plus the final
        // rounding (2^-52 max|u|), or 1.2e-15 max|u| where the point is made exact.
        int point(int p0, int p1, int p2) {
            using cut_detail::DD;
            using cut_detail::I256;
            Vtx x;
            x.p[0] = p0;
            x.p[1] = p1;
            x.p[2] = p2;
            int third = -1;
            cc_.on_line(p0, p1, p2, x.wa, x.wb, third);
            if (third >= 0)
                for (const int w : {x.wa, x.wb})
                    if (cc_.weld_on(w, third)) {
                        const auto &q = cc_.weld_q_[static_cast<std::size_t>(w)];
                        const std::int64_t qq[3] = {q[0], q[1], q[2]};
                        return lattice(qq, p0, p1, p2, w);
                    }
            const int ps[3] = {p0, p1, p2};
            int ax[3], axes = 0;
            std::int64_t g[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                ax[i] = axis_plane(ps[i], g[i]);
                axes += ax[i] >= 0;
            }
            const double eps = 0x1p-53;
            if (axes == 3 && ax[0] != ax[1] && ax[1] != ax[2] && ax[0] != ax[2]) {
                std::int64_t q[3];
                for (int i = 0; i < 3; ++i) q[ax[i]] = g[i];
                const int k = lattice(q, p0, p1, p2);
                v[static_cast<std::size_t>(k)].wa = x.wa;
                v[static_cast<std::size_t>(k)].wb = x.wb;
                return k;
            }
            if (x.wa >= 0 && third >= 0) {
                const auto &A = cc_.weld_q_[static_cast<std::size_t>(x.wa)];
                const auto &B = cc_.weld_q_[static_cast<std::size_t>(x.wb)];
                const Plane &P = cc_.planes_[static_cast<std::size_t>(third)];
                const I256 num = P.d4 - (I256::mul(P.n[0], A[0]) + I256::mul(P.n[1], A[1]) + I256::mul(P.n[2], A[2]));
                const I256 den = I256::mul(P.n[0], B[0] - A[0]) + I256::mul(P.n[1], B[1] - A[1]) + I256::mul(P.n[2], B[2] - A[2]);
                if (den.sign() != 0) {
                    const double dS = den.to_double() * cc_.S_;
                    for (int c = 0; c < 3; ++c) {
                        const I256 N = I256::mul64s(den, A[c] - C_[c]) + I256::mul64s(num, B[c] - A[c]);
                        x.u[c] = N.to_double() / dS;
                    }
                    for (int i = 0; i < 3; ++i)
                        if (ps[i] == third && ax[i] >= 0) x.u[ax[i]] = static_cast<double>(g[i] - C_[ax[i]]) / cc_.S_;
                    double m = 0.0;
                    for (int c = 0; c < 3; ++c) m = (std::max)(m, std::abs(x.u[c]));
                    x.r = 8.0 * eps * m + 1e-300;
                    v.push_back(x);
                    return static_cast<int>(v.size()) - 1;
                }
            }
            if (x.wa < 0 && axes == 2) {
                int ia = -1, ib = -1, ip = -1;
                for (int i = 0; i < 3; ++i) {
                    if (ax[i] < 0) ip = i;
                    else if (ia < 0) ia = i;
                    else ib = i;
                }
                const int a = ax[ia], b = ax[ib], c = 3 - a - b;
                const Plane &P = cc_.planes_[static_cast<std::size_t>(ps[ip])];
                if (a != b && P.n[c] != 0) {
                    const I256 num = P.d4 - (I256::mul(P.n[a], g[ia]) + I256::mul(P.n[b], g[ib]) + I256::mul(P.n[c], C_[c]));
                    x.u[a] = static_cast<double>(g[ia] - C_[a]) / cc_.S_;
                    x.u[b] = static_cast<double>(g[ib] - C_[b]) / cc_.S_;
                    x.u[c] = num.to_double() / (static_cast<double>(P.n[c]) * cc_.S_);
                    double m = 0.0;
                    for (int k = 0; k < 3; ++k) m = (std::max)(m, std::abs(x.u[k]));
                    x.r = 8.0 * eps * m + 1e-300;
                    v.push_back(x);
                    return static_cast<int>(v.size()) - 1;
                }
            }
            bool good = false;
            {
                const PlaneLoc A = loc(p0), B = loc(p1), C = loc(p2);  // copies: loc() may insert
                DD bc[3], ca[3], ab[3];
                crs(B.lun, C.lun, bc);
                crs(C.lun, A.lun, ca);
                crs(A.lun, B.lun, ab);
                const DD W = A.lun[0] * bc[0] + A.lun[1] * bc[1] + A.lun[2] * bc[2];
                const DD e(0x1p-100);
                const DD D = cut_detail::dd_abs(A.ld) + cut_detail::dd_abs(B.ld) + cut_detail::dd_abs(C.ld);
                const DD errN = DD(44.0) * e * D, errW = DD(120.0) * e;
                good = DD(2.0) * errW < cut_detail::dd_abs(W);
                if (good) {
                    DD m(0.0);
                    for (int c = 0; c < 3; ++c) {
                        const DD u = (A.ld * bc[c] + B.ld * ca[c] + C.ld * ab[c]) / W;
                        x.u[c] = static_cast<double>(u);
                        if (m < cut_detail::dd_abs(u)) m = cut_detail::dd_abs(u);
                    }
                    // The bound before the final rounding, rounded up by 2^-50 relative for the
                    // roundings of its own computation.
                    const double rb = static_cast<double>((errN + m * errW) / (cut_detail::dd_abs(W) - errW) + DD(4.0) * e * m) * (1.0 + 0x1p-50);
                    x.r = rb + 0x1p-52 * static_cast<double>(m) + 1e-300;
                    good = rb <= 0x1p-60 * (std::max)(1.0, static_cast<double>(m));
                }
            }
            v.push_back(x);
            const int k = static_cast<int>(v.size()) - 1;
#ifdef ACOUSTIC_SPACES_CUT_EXACT_POINTS
            good = false;
#endif
            if (!good) refine(k);
            return k;
        }
        // The point where segment a b crosses plane p2, the segment lying on planes p0 and p1.
        int cross(int, int, int, double, double, double, double, int p0, int p1, int p2) { return point(p0, p1, p2); }

        double area(const Ints &g) const {
            if (g.size() < 3) return 0.0;
            const double *o = v[static_cast<std::size_t>(g[0])].u;
            double s[3] = {0.0, 0.0, 0.0};
            for (std::size_t i = 1; i + 1 < g.size(); ++i) {
                const double *p = v[static_cast<std::size_t>(g[i])].u, *q = v[static_cast<std::size_t>(g[i + 1])].u;
                const double e1[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
                const double e2[3] = {q[0] - o[0], q[1] - o[1], q[2] - o[2]};
                s[0] += e1[1] * e2[2] - e1[2] * e2[1];
                s[1] += e1[2] * e2[0] - e1[0] * e2[2];
                s[2] += e1[0] * e2[1] - e1[1] * e2[0];
            }
            return 0.5 * std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        }
        const double *u(int k) const { return v[static_cast<std::size_t>(k)].u; }

    private:
        static void crs(const cut_detail::DD *a, const cut_detail::DD *b, cut_detail::DD *o) {
            o[0] = a[1] * b[2] - a[2] * b[1];
            o[1] = a[2] * b[0] - a[0] * b[2];
            o[2] = a[0] * b[1] - a[1] * b[0];
        }
        const CutCells &cc_;
        std::int64_t C_[3];
        std::vector<std::array<Big, 4>> ex_;
        // The planes in this frame: val_[slot], ids_[slot] its plane, (*slot_)[plane] its slot.
        std::vector<PlaneLoc> val_;
        std::vector<int> ids_;
        std::vector<int> own_;
        std::vector<int> *slot_ = nullptr;
        static std::vector<int> &pool() {
            static thread_local std::vector<int> p;
            return p;
        }
        static bool &pool_busy() {
            static thread_local bool b = false;
            return b;
        }

        const PlaneLoc &loc(int p) {
            int &slot = (*slot_)[static_cast<std::size_t>(p)];
            if (slot >= 0) return val_[static_cast<std::size_t>(slot)];
            const Plane &P = cc_.planes_[static_cast<std::size_t>(p)];
            using cut_detail::I256;
            const I256 off = P.d4 - (I256::mul(P.n[0], C_[0]) + I256::mul(P.n[1], C_[1]) + I256::mul(P.n[2], C_[2]));
            const cut_detail::DD ld = off.to_dd() / (P.llen * cut_detail::DD(cc_.S_));
            val_.push_back(PlaneLoc{{P.un[0], P.un[1], P.un[2]}, off.to_double() / (P.len * cc_.S_), {P.lun[0], P.lun[1], P.lun[2]}, ld});
            ids_.push_back(p);
            slot = static_cast<int>(val_.size()) - 1;
            return val_.back();
        }

        // Exact homogeneous coordinates (X, W) of a point made of three planes: X = dA (nB x
        // nC) + dB (nC x nA) + dC (nA x nB), W = nA . (nB x nC).
        const std::array<Big, 4> &exact(int k) {
            Vtx &x = v[static_cast<std::size_t>(k)];
            if (x.ex >= 0) return ex_[static_cast<std::size_t>(x.ex)];
            const Plane &A = cc_.planes_[static_cast<std::size_t>(x.p[0])];
            const Plane &B = cc_.planes_[static_cast<std::size_t>(x.p[1])];
            const Plane &C = cc_.planes_[static_cast<std::size_t>(x.p[2])];
            auto crs = [](const Plane &P, const Plane &Q, Big out[3]) {
                out[0] = Big(P.n[1]) * Big(Q.n[2]) - Big(P.n[2]) * Big(Q.n[1]);
                out[1] = Big(P.n[2]) * Big(Q.n[0]) - Big(P.n[0]) * Big(Q.n[2]);
                out[2] = Big(P.n[0]) * Big(Q.n[1]) - Big(P.n[1]) * Big(Q.n[0]);
            };
            Big bc[3], ca[3], ab[3];
            crs(B, C, bc);
            crs(C, A, ca);
            crs(A, B, ab);
            std::array<Big, 4> h;
            for (int c = 0; c < 3; ++c) h[static_cast<std::size_t>(c)] = A.d * bc[c] + B.d * ca[c] + C.d * ab[c];
            h[3] = Big(A.n[0]) * bc[0] + Big(A.n[1]) * bc[1] + Big(A.n[2]) * bc[2];
            if (h[3].sign() == 0) throw std::logic_error("fitted walls: a point of three planes that do not meet");
            ex_.push_back(h);
            x.ex = static_cast<int>(ex_.size()) - 1;
            return ex_.back();
        }

        void refine(int k) {
            const std::array<Big, 4> h = exact(k);
            Vtx &x = v[static_cast<std::size_t>(k)];
            const double w = h[3].to_double();
            double m = 0.0;
            for (int c = 0; c < 3; ++c) {
                x.u[c] = (h[static_cast<std::size_t>(c)] - Big(C_[c]) * h[3]).to_double() / (w * cc_.S_);
                m = std::max(m, std::abs(x.u[c]));
            }
            x.r = 1.2e-15 * m + 1e-300;
            ++refined;
        }

        int exact_side(int k, int p) {
            ++exact_calls;
            const Plane &P = cc_.planes_[static_cast<std::size_t>(p)];
            const Vtx &x = v[static_cast<std::size_t>(k)];
            if (x.lattice) {
                using cut_detail::I256;
                return (I256::mul(P.n[0], x.q[0]) + I256::mul(P.n[1], x.q[1]) + I256::mul(P.n[2], x.q[2]) - P.d4).sign();
            }
            ++exact_t;
            const std::array<Big, 4> &h = exact(k);
            const Big s = Big(P.n[0]) * h[0] + Big(P.n[1]) * h[1] + Big(P.n[2]) * h[2] - P.d * h[3];
            return s.sign() * h[3].sign();
        }
    };

    // A convex polygon on plane `plane`: its points and, for each edge from v[i] to v[i + 1],
    // the other plane its line lies on.
    struct Poly {
        int plane = -1;
        Ints v, e;
    };

    // Splits polygon g by plane p (its points' signs in S): its part on the - side into
    // minus, on the + side into plus (either may be null). A point exactly on p is in both;
    // an edge whose ends lie strictly on either side gets the point where it crosses p
    // (cross(i), the same point for both parts). The new edge along p lies on p.
    template <typename Cross>
    static void split_poly(const Poly &g, const int *s, int p, Cross cross, Poly *minus, Poly *plus) {
        if (minus) {
            minus->plane = g.plane;
            minus->v.clear();
            minus->e.clear();
        }
        if (plus) {
            plus->plane = g.plane;
            plus->v.clear();
            plus->e.clear();
        }
        const std::size_t m = g.v.size();
        for (std::size_t i = 0; i < m; ++i) {
            const std::size_t j = (i + 1) % m;
            const int sa = s[i], sb = s[j];
            const int a = g.v[i], e = g.e[i];
            const bool crosses = (sa > 0 && sb < 0) || (sa < 0 && sb > 0);
            const int x = crosses ? cross(i) : -1;
            for (int side : {1, -1}) {
                Poly *out = side > 0 ? plus : minus;
                if (!out) continue;
                const int ta = sa * side, tb = sb * side;
                if (ta > 0) {
                    out->v.push_back(a);
                    out->e.push_back(e);
                    if (tb < 0) {
                        out->v.push_back(x);
                        out->e.push_back(p);
                    }
                } else if (ta == 0) {
                    out->v.push_back(a);
                    out->e.push_back(tb < 0 ? p : e);
                } else if (tb > 0) {
                    out->v.push_back(x);
                    out->e.push_back(e);
                }
            }
        }
    }

    // ---- the frame and the lattice ----------------------------------------------------------

    // Q(x) on axis a: round((x - X0_a) 2^s), formed exactly (ties to even).
    I128 quantize(double x, int a) const {
        const I128 x0 = cut_detail::to_i128(std::ldexp(X0_[a], s_));
        const double y = std::ldexp(x, s_);
        // A located point further than 2^120 quanta (far outside any grid) is held at that
        // distance: it stays outside the grid's box, and to_i128 stays defined.
        const double cap = std::ldexp(1.0, 120);
        if (!(std::abs(y) < cap)) return y > 0 ? cut_detail::to_i128(cap) : -cut_detail::to_i128(cap);
        if (std::abs(y) >= 4.0e18) return cut_detail::to_i128(y) - x0;
        const double fl = std::floor(y), fr = y - fl;
        I128 t = static_cast<I128>(static_cast<std::int64_t>(fl)) - x0;
        if (fr > 0.5 || (fr == 0.5 && (t & 1) != 0)) t += 1;
        return t;
    }
    static std::int64_t small(I128 x) {
        if (x >= (static_cast<I128>(1) << 61) || x <= -(static_cast<I128>(1) << 61))
            throw std::logic_error("fitted walls: a lattice coordinate outside the frame");
        return static_cast<std::int64_t>(x);
    }

    // The frame over the faces whose bounding box meets the grid's box grown by 2h, and the grid's box.
    void frame(const Mesh &mesh, const Grid &grid) {
        const double h = cell_;
        double glo[3], ghi[3];
        for (int a = 0; a < 3; ++a) {
            glo[a] = grid.face_at[static_cast<std::size_t>(a)].front() - 2.0 * h;
            ghi[a] = grid.face_at[static_cast<std::size_t>(a)].back() + 2.0 * h;
        }
        double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
        given_.assign(mesh.face_count, 0);
        for (std::size_t f = 0; f < mesh.face_count; ++f) {
            const std::array<Vec3, 3> p = mesh.corners(f);
            bool meets = true;
            for (int a = 0; a < 3 && meets; ++a) {
                const double mn = std::min({p[0][a], p[1][a], p[2][a]}), mx = std::max({p[0][a], p[1][a], p[2][a]});
                meets = mx >= glo[a] && mn <= ghi[a];
            }
            if (!meets) continue;
            given_[f] = 1;
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) {
                    lo[a] = std::min(lo[a], p[static_cast<std::size_t>(k)][a]);
                    hi[a] = std::max(hi[a], p[static_cast<std::size_t>(k)][a]);
                }
        }
        // The grid's own box too: every point of a stencil (T-1) lies in it, so
        // on Q under 2^61 however far the grid reaches past the faces.
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], grid.face_at[static_cast<std::size_t>(a)].front());
            hi[a] = std::max(hi[a], grid.face_at[static_cast<std::size_t>(a)].back());
        }
        double E = 0.0;
        for (int a = 0; a < 3; ++a) E = std::max(E, (hi[a] - lo[a]) / 2.0);
        E += 2.0 * h;
        if (E / h >= std::ldexp(1.0, 24)) {
            refuse(kRefusedFrame,
                "fitted walls: a face reaching the grid has a vertex " + std::to_string(E / h) +
                " cells from the middle of the faces reaching it; fitted walls hold coordinates in 64-bit "
                "integers at 2^-35 cells or finer, which needs them within 2^24 cells. Split the face.");
        }
        int e = 0;
        const double m = std::frexp(E, &e);
        const int p = (m > 0.5 ? e : e - 1) + 1;
        s_ = 61 - p;
        for (int a = 0; a < 3; ++a) X0_[a] = std::ldexp(std::nearbyint(std::ldexp((lo[a] + hi[a]) / 2.0, -(p - 2))), p - 2);
        S_ = std::ldexp(h, s_);
        stats_.quantum_bits = s_;
        double reach = 0.0;
        for (int a = 0; a < 3; ++a)
            for (const double x : grid.face_at[static_cast<std::size_t>(a)]) reach = std::max(reach, std::abs(x));
        if (std::nextafter(reach, 1e308) - reach >= h / 4.0) {
            refuse(kRefusedGridFar, "fitted walls: the grid lies too far from the origin of its coordinates for its cell (" +
                                        std::to_string(reach) + " m against a cell of " + std::to_string(h) + " m)");
        }
    }

    // The grid planes and centres on the lattice, and their planes.
    void lattice_grid(const Grid &grid) {
        for (std::size_t a = 0; a < 3; ++a) {
            G_[a].clear();
            Cq_[a].clear();
            for (const double x : grid.face_at[a]) G_[a].push_back(quantize(x, static_cast<int>(a)));
            for (const double x : grid.centre_at[a]) Cq_[a].push_back(quantize(x, static_cast<int>(a)));
            for (std::size_t i = 0; i < n_[a]; ++i) {
                if (!(G_[a][i] < Cq_[a][i] && Cq_[a][i] < G_[a][i + 1]))
                    throw std::logic_error("fitted walls: a centre not inside its cell on the lattice");
            }
            // Each cell's width on the lattice, in cells: every cell is measured
            // between its own planes G_i and G_i+1, the uncut ones too, so that the air is
            // vol(Omega) on Q however far the grid's doubles lie from the origin.
            width_[a].clear();
            for (std::size_t i = 0; i < n_[a]; ++i) width_[a].push_back(static_cast<double>(G_[a][i + 1] - G_[a][i]) / S_);
        }
    }
    // An uncut cell's volume (cells^3, a measure on Q), and the area of its face along axis a.
    double cell_volume(std::size_t i) const {
        const auto p = at(i);
        return width_[0][p[0]] * width_[1][p[1]] * width_[2][p[2]];
    }
    double face_area_of(std::size_t i, int a) const {
        const auto p = at(i);
        const std::size_t b = static_cast<std::size_t>((a + 1) % 3), c = static_cast<std::size_t>((a + 2) % 3);
        return width_[b][p[b]] * width_[c][p[c]];
    }

    // The id of plane (n, d), made primitive with its first nonzero component positive; one
    // id per plane.
    int plane_id(I128 n0, I128 n1, I128 n2, const std::int64_t at[3]) {
        cut_detail::U128 g = gcd3(n0, n1, n2);
        if (g == 0) throw std::logic_error("fitted walls: a plane without a normal");
        I128 n[3] = {cut_detail::exact_div(n0, g), cut_detail::exact_div(n1, g), cut_detail::exact_div(n2, g)};
        const I128 lead = n[0] != 0 ? n[0] : (n[1] != 0 ? n[1] : n[2]);
        if (lead < 0)
            for (I128 &c : n) c = -c;
        using cut_detail::I256;
        return plane_of(n, I256::mul(n[0], at[0]) + I256::mul(n[1], at[1]) + I256::mul(n[2], at[2]));
    }
    // |n| < 2^127 and |d| < 2^254 hold for every plane made here: a face's normal is a cross
    // product of two lattice differences (each component below 2^125), a coverage plane's a
    // lattice difference, and d is n at a lattice point (|q| < 2^63).
    int plane_of(const I128 n[3], const cut_detail::I256 &d) {
        PlaneKey key{{n[0], n[1], n[2]}, d};
        const auto it = plane_map_.find(key);
        if (it != plane_map_.end()) return it->second;
        Plane P;
        double len2 = 0.0;
        for (int c = 0; c < 3; ++c) {
            P.n[c] = n[c];
            const double x = static_cast<double>(n[c]);
            len2 += x * x;
        }
        P.d = Big::of_limbs(d.w);
        P.d4 = d;
        P.len = std::sqrt(len2);
        for (int c = 0; c < 3; ++c) P.un[c] = static_cast<double>(n[c]) / P.len;
        cut_detail::DD l2(0.0), nd[3];
        for (int c = 0; c < 3; ++c) {
            nd[c] = cut_detail::I256::of(n[c]).to_dd();
            l2 = l2 + nd[c] * nd[c];
        }
        P.llen = cut_detail::dd_sqrt(l2);
        for (int c = 0; c < 3; ++c) P.lun[c] = nd[c] / P.llen;
        planes_.push_back(P);
        const int id = static_cast<int>(planes_.size()) - 1;
        plane_map_.emplace(key, id);
        return id;
    }
    struct PlaneKey {
        std::array<I128, 3> n;
        cut_detail::I256 d;
        bool operator<(const PlaneKey &o) const {
            if (n != o.n) return n < o.n;
            return (d - o.d).sign() < 0;
        }
        bool operator==(const PlaneKey &o) const {
            return n == o.n && d.w[0] == o.d.w[0] && d.w[1] == o.d.w[1] && d.w[2] == o.d.w[2] && d.w[3] == o.d.w[3];
        }
    };
    struct PlaneHash {
        std::size_t operator()(const PlaneKey &k) const {
            std::uint64_t h = 1469598103934665603ull;
            auto mix = [&](std::uint64_t x) { h = (h ^ x) * 1099511628211ull; };
            for (const I128 &c : k.n) {
                mix(static_cast<std::uint64_t>(c));
                mix(static_cast<std::uint64_t>(static_cast<cut_detail::U128>(c) >> 64));
            }
            for (const std::uint64_t x : k.d.w) mix(x);
            return static_cast<std::size_t>(h);
        }
    };
    // The plane of grid plane i along axis a.
    int grid_plane(int a, std::size_t i) {
        I128 n[3] = {0, 0, 0};
        n[a] = 1;
        return plane_of(n, cut_detail::I256::of(G_[static_cast<std::size_t>(a)][i]));
    }

    // ---- the faces, once for the whole model ---------------------------------------------------

    struct FaceX {
        std::size_t face = 0;  // its index in the model
        std::int64_t q[3][3] = {};
        int plane = -1;        // its group's plane; -1: degenerate
        int orient = 0;        // +1 when its normal (right hand on its corners) is the plane's n
        int axis = 0;          // the plane's dominant axis
        int e[3] = {-1, -1, -1};  // the coverage plane of edge k (q[k] to q[k + 1])
        int in[3] = {0, 0, 0};    // the side of e[k] the face lies on
        std::int32_t material = 0;
        int w[3] = {-1, -1, -1};  // its welded vertices' ids
        std::size_t rank = 0;  // its place in key order
        int mem_axis = -1;     // a membrane: the grid plane's axis and index
        std::size_t mem_index = 0;
        double lo[3], hi[3];   // its box (m), from the given doubles
        std::int64_t qlo[3], qhi[3];  // its box on the lattice (every candidate test reads this)
        bool layer = false;    // in a candidate pair of refusal (ii)
    };

    void faces_global(const Mesh &mesh, const std::int64_t *face_material) {
        // The lattice points and the welded identity (equal Q), the degenerate faces.
        for (std::size_t f = 0; f < mesh.face_count; ++f) {
            if (!given_[f]) continue;
            FaceX F;
            F.face = f;
            F.material = static_cast<std::int32_t>(face_material[f]);
            const std::array<Vec3, 3> p = mesh.corners(f);
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) F.q[k][a] = small(quantize(p[static_cast<std::size_t>(k)][a], a));
            for (int a = 0; a < 3; ++a) {
                F.lo[a] = std::min({p[0][a], p[1][a], p[2][a]});
                F.hi[a] = std::max({p[0][a], p[1][a], p[2][a]});
                F.qlo[a] = std::min({F.q[0][a], F.q[1][a], F.q[2][a]});
                F.qhi[a] = std::max({F.q[0][a], F.q[1][a], F.q[2][a]});
            }
            faces_.push_back(F);
        }
        stats_.faces = faces_.size();
        planes_.reserve(4 * faces_.size() + n_[0] + n_[1] + n_[2] + 3);
        plane_map_.reserve(planes_.capacity());
        // Planes of the faces, in the order of their tuples (so group ids follow it).
        std::vector<std::pair<PlaneKey, std::size_t>> keyed;
        for (std::size_t t = 0; t < faces_.size(); ++t) {
            FaceX &F = faces_[t];
            I128 u[3], w[3];
            for (int a = 0; a < 3; ++a) {
                u[a] = static_cast<I128>(F.q[1][a]) - F.q[0][a];
                w[a] = static_cast<I128>(F.q[2][a]) - F.q[0][a];
            }
            I128 n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]};
            if (n[0] == 0 && n[1] == 0 && n[2] == 0) {
                stats_.degenerate_faces += 1;
                const std::array<Vec3, 3> p = mesh.corners(F.face);
                stats_.degenerate_area += 0.5 * norm(cross(p[1] - p[0], p[2] - p[0]));
                continue;
            }
            const cut_detail::U128 g = gcd3(n[0], n[1], n[2]);
            for (I128 &c : n) c = cut_detail::exact_div(c, g);
            const I128 lead = n[0] != 0 ? n[0] : (n[1] != 0 ? n[1] : n[2]);
            F.orient = lead > 0 ? 1 : -1;
            if (lead < 0)
                for (I128 &c : n) c = -c;
            using cut_detail::I256;
            const I256 d = I256::mul(n[0], F.q[0][0]) + I256::mul(n[1], F.q[0][1]) + I256::mul(n[2], F.q[0][2]);
            keyed.push_back({PlaneKey{{n[0], n[1], n[2]}, d}, t});
            int ax = 0;
            for (int a = 1; a < 3; ++a)
                if (cut_detail::mag(n[a]) > cut_detail::mag(n[ax])) ax = a;
            F.axis = ax;
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
        for (const auto &[key, t] : keyed) {
            const I128 n[3] = {key.n[0], key.n[1], key.n[2]};
            faces_[t].plane = plane_of(n, key.d);
        }
        groups_ = planes_.size();
        stats_.planes = groups_;
        // The grid's planes.
        for (int a = 0; a < 3; ++a) {
            grid_plane_[static_cast<std::size_t>(a)].clear();
            for (std::size_t i = 0; i <= n_[static_cast<std::size_t>(a)]; ++i)
                grid_plane_[static_cast<std::size_t>(a)].push_back(grid_plane(a, i));
        }
        // Membranes, coverage planes, keys.
        for (FaceX &F : faces_) {
            if (F.plane < 0) continue;
            const Plane &P = planes_[static_cast<std::size_t>(F.plane)];
            for (int a = 0; a < 3 && F.mem_axis < 0; ++a) {
                if (P.n[a] != 1 || P.n[(a + 1) % 3] != 0 || P.n[(a + 2) % 3] != 0) continue;
                const std::vector<I128> &G = G_[static_cast<std::size_t>(a)];
                const I128 d = static_cast<I128>(F.q[0][a]);
                const auto it = std::lower_bound(G.begin(), G.end(), d);
                if (it != G.end() && *it == d) {
                    F.mem_axis = a;
                    F.mem_index = static_cast<std::size_t>(it - G.begin());
                    stats_.membrane_faces += 1;
                }
            }
            for (int k = 0; k < 3; ++k) {
                const std::int64_t *A = F.q[k], *B = F.q[(k + 1) % 3], *Cc = F.q[(k + 2) % 3];
                const I128 dd[3] = {static_cast<I128>(B[0]) - A[0], static_cast<I128>(B[1]) - A[1],
                                    static_cast<I128>(B[2]) - A[2]};
                I128 m[3] = {0, 0, 0};  // (B - A) x e_axis
                if (F.axis == 0) {
                    m[1] = dd[2];
                    m[2] = -dd[1];
                } else if (F.axis == 1) {
                    m[0] = -dd[2];
                    m[2] = dd[0];
                } else {
                    m[0] = dd[1];
                    m[1] = -dd[0];
                }
                F.e[k] = plane_id(m[0], m[1], m[2], A);
                const Plane &E = planes_[static_cast<std::size_t>(F.e[k])];
                using cut_detail::I256;
                F.in[k] = (I256::mul(E.n[0], Cc[0]) + I256::mul(E.n[1], Cc[1]) + I256::mul(E.n[2], Cc[2]) - E.d4).sign();
                if (F.in[k] == 0) throw std::logic_error("fitted walls: a face's coverage plane holds its corner");
            }
        }
        // Key order: the welded corners sorted, the orientation, the material.
        std::vector<std::size_t> order(faces_.size());
        for (std::size_t t = 0; t < order.size(); ++t) order[t] = t;
        using Key = std::tuple<std::array<std::array<std::int64_t, 3>, 3>, int, std::int32_t>;
        std::vector<Key> keys(faces_.size());
        for (std::size_t t = 0; t < faces_.size(); ++t) {
            const FaceX &F = faces_[t];
            std::array<std::array<std::int64_t, 3>, 3> c;
            for (int k = 0; k < 3; ++k)
                for (int a = 0; a < 3; ++a) c[static_cast<std::size_t>(k)][static_cast<std::size_t>(a)] = F.q[k][a];
            std::sort(c.begin(), c.end());
            keys[t] = Key(c, F.orient, F.material);
        }
        std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return keys[x] < keys[y]; });
        for (std::size_t r = 0; r < order.size(); ++r) faces_[order[r]].rank = r;
        // Welded vertices (equal lattice points), and the planes of the faces through each:
        // a welded vertex lies on each of them (incidence, without arithmetic).
        std::vector<std::array<std::int64_t, 3>> pts;
        for (const FaceX &F : faces_)
            for (int k = 0; k < 3; ++k) pts.push_back({F.q[k][0], F.q[k][1], F.q[k][2]});
        std::sort(pts.begin(), pts.end());
        pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
        std::vector<std::pair<int, int>> on;  // (weld id, plane)
        for (FaceX &F : faces_)
            for (int k = 0; k < 3; ++k) {
                const std::array<std::int64_t, 3> q{F.q[k][0], F.q[k][1], F.q[k][2]};
                F.w[k] = static_cast<int>(std::lower_bound(pts.begin(), pts.end(), q) - pts.begin());
            }
        // Lines: the pairs of planes whose meet is the line of a welded edge of the mesh
        // (a face's plane and its edge's coverage plane; the planes of two faces sharing the
        // edge), with the edge's two welded vertices.
        line_.clear();
        std::vector<std::array<int, 4>> edges;  // (w low, w high, plane, -)
        for (const FaceX &F : faces_) {
            if (F.plane < 0) continue;
            for (int k = 0; k < 3; ++k) {
                const int a = F.w[k], b = F.w[(k + 1) % 3];
                on.push_back({a, F.plane});
                on.push_back({a, F.e[k]});
                on.push_back({b, F.e[k]});
                line_.push_back({pair_key(F.plane, F.e[k]), std::min(a, b), std::max(a, b)});
                edges.push_back({std::min(a, b), std::max(a, b), F.plane, 0});
            }
        }
        std::sort(edges.begin(), edges.end());
        for (std::size_t i = 0; i < edges.size();) {
            std::size_t j = i;
            while (j < edges.size() && edges[j][0] == edges[i][0] && edges[j][1] == edges[i][1]) ++j;
            for (std::size_t x = i; x < j; ++x)
                for (std::size_t y = x + 1; y < j; ++y)
                    if (edges[x][2] != edges[y][2]) line_.push_back({pair_key(edges[x][2], edges[y][2]), edges[i][0], edges[i][1]});
            i = j;
        }
        std::sort(line_.begin(), line_.end());
        line_hash_.assign(std::max<std::size_t>(16, std::size_t{1} << (2 + cut_detail::bit_width(line_.size()))), -1);
        for (std::size_t k = 0; k < line_.size(); ++k) {
            if (k > 0 && line_[k][0] == line_[k - 1][0]) continue;  // the first of a key, as a search would find
            std::size_t i = line_slot(line_[k][0]);
            while (line_hash_[i] != -1) i = (i + 1) & (line_hash_.size() - 1);
            line_hash_[i] = static_cast<std::int64_t>(k);
        }
        std::sort(on.begin(), on.end());
        on.erase(std::unique(on.begin(), on.end()), on.end());
        through_start_.assign(pts.size() + 1, 0);
        for (const auto &e : on) through_start_[static_cast<std::size_t>(e.first) + 1] += 1;
        for (std::size_t k = 0; k < pts.size(); ++k) through_start_[k + 1] += through_start_[k];
        through_.clear();
        for (const auto &e : on) through_.push_back(e.second);
        weld_q_ = std::move(pts);
    }
    static std::int64_t pair_key(int a, int b) {
        return (static_cast<std::int64_t>(std::min(a, b)) << 32) | static_cast<std::uint32_t>(std::max(a, b));
    }
    // A welded edge whose line a point of planes p0, p1, p2 lies on, if two of its planes
    // meet in such a line (see line_): wa, wb its vertices, else -1; third the remaining plane.
    void on_line(int p0, int p1, int p2, int &wa, int &wb, int &third) const {
        wa = wb = third = -1;
        const std::int64_t keys[3] = {pair_key(p0, p1), pair_key(p0, p2), pair_key(p1, p2)};
        const int rest[3] = {p2, p1, p0};
        for (int j = 0; j < 3; ++j) {
            for (std::size_t i = line_slot(keys[j]); line_hash_[i] != -1; i = (i + 1) & (line_hash_.size() - 1)) {
                const auto &e = line_[static_cast<std::size_t>(line_hash_[i])];
                if (e[0] != keys[j]) continue;
                wa = static_cast<int>(e[1]);
                wb = static_cast<int>(e[2]);
                third = rest[j];
                return;
            }
        }
    }
    // Whether welded vertex w lies on plane p, exactly (its lattice point in p's equation).
    bool weld_on(int w, int p) const {
        const Plane &P = planes_[static_cast<std::size_t>(p)];
        const auto &q = weld_q_[static_cast<std::size_t>(w)];
        using cut_detail::I256;
        return (I256::mul(P.n[0], q[0]) + I256::mul(P.n[1], q[1]) + I256::mul(P.n[2], q[2]) - P.d4).sign() == 0;
    }
    std::size_t line_slot(std::int64_t key) const {
        return static_cast<std::size_t>((static_cast<std::uint64_t>(key) * 0x9E3779B97F4A7C15ull) >> 20) & (line_hash_.size() - 1);
    }
    // Whether welded vertex w lies on plane p by incidence (a face through it lies on p, or
    // the coverage plane of an edge of such a face through it).
    bool through(int w, int p) const {
        const auto b = through_.begin() + static_cast<std::ptrdiff_t>(through_start_[static_cast<std::size_t>(w)]);
        const auto e = through_.begin() + static_cast<std::ptrdiff_t>(through_start_[static_cast<std::size_t>(w) + 1]);
        return std::binary_search(b, e, p);
    }

    // ---- the boundary of the mesh, its zones (refusal (i)) -------------------------------------

    // An odd interval of the mod-2 boundary: its ends (lattice points), the faces with an
    // edge on its line that overlaps it, and whether part of it lies on no other face.
    struct Interval {
        std::array<std::int64_t, 3> a{}, b{};
        std::vector<std::size_t> faces;  // indices into faces_
        std::size_t line = 0;            // its line (index into line_faces_)
        int kind = -1;                   // 1: some part lies on no other face (free); 0: contact only
        double free_length = 0.0;        // cells (a measure)
    };
    // The zone of an interval, sigma + [-R, R]^3: its facet planes, each with the sign of
    // n.x - d inside it (-1: n.x <= d; +1: n.x >= d).
    struct Zone {
        std::vector<std::pair<int, int>> facets;
    };

    // The mod-2 boundary on lines: every edge of every face given (degenerate ones too) is a
    // segment of its line, keyed exactly by its primitive direction d (first nonzero component
    // positive) and its moment q x d; the segments of a line are summed mod 2 by a sweep over
    // t = q . d. The odd intervals, their kinds, and their zones (R = ceil(kGap h 2^s) quanta,
    // formed exactly).
    void boundary() {
        struct Seg {
            std::array<I128, 3> d, m;
            I128 ta, tb;
            std::size_t face;
            std::array<std::int64_t, 3> A, B;
        };
        std::vector<Seg> segs;
        for (std::size_t t = 0; t < faces_.size(); ++t) {
            const FaceX &F = faces_[t];
            for (int k = 0; k < 3; ++k) {
                std::array<std::int64_t, 3> A{F.q[k][0], F.q[k][1], F.q[k][2]};
                std::array<std::int64_t, 3> B{F.q[(k + 1) % 3][0], F.q[(k + 1) % 3][1], F.q[(k + 1) % 3][2]};
                if (A == B) continue;
                I128 d[3] = {static_cast<I128>(B[0]) - A[0], static_cast<I128>(B[1]) - A[1], static_cast<I128>(B[2]) - A[2]};
                const cut_detail::U128 g = gcd3(d[0], d[1], d[2]);
                for (I128 &c : d) c = cut_detail::exact_div(c, g);
                const I128 lead = d[0] != 0 ? d[0] : (d[1] != 0 ? d[1] : d[2]);
                if (lead < 0) {
                    for (I128 &c : d) c = -c;
                    std::swap(A, B);
                }
                Seg s;
                s.d = {d[0], d[1], d[2]};
                s.m = {A[1] * d[2] - A[2] * d[1], A[2] * d[0] - A[0] * d[2], A[0] * d[1] - A[1] * d[0]};
                s.ta = A[0] * d[0] + A[1] * d[1] + A[2] * d[2];
                s.tb = B[0] * d[0] + B[1] * d[1] + B[2] * d[2];
                s.face = t;
                s.A = A;
                s.B = B;
                segs.push_back(s);
            }
        }
        std::sort(segs.begin(), segs.end(), [](const Seg &x, const Seg &y) {
            if (x.d != y.d) return x.d < y.d;
            if (x.m != y.m) return x.m < y.m;
            if (x.ta != y.ta) return x.ta < y.ta;
            if (x.tb != y.tb) return x.tb < y.tb;
            return x.face < y.face;
        });
        intervals_.clear();
        line_faces_.clear();
        std::vector<std::pair<I128, std::array<std::int64_t, 3>>> ev;
        for (std::size_t i = 0; i < segs.size();) {
            std::size_t j = i;
            while (j < segs.size() && segs[j].d == segs[i].d && segs[j].m == segs[i].m) ++j;
            ev.clear();
            for (std::size_t k = i; k < j; ++k) {
                ev.push_back({segs[k].ta, segs[k].A});
                ev.push_back({segs[k].tb, segs[k].B});
            }
            std::sort(ev.begin(), ev.end(), [](const auto &x, const auto &y) { return x.first < y.first; });
            // parity after each distinct t; an odd run from t_s to t_e is one interval
            bool odd = false, open = false;
            I128 ts = 0;
            std::array<std::int64_t, 3> ps{};
            const std::size_t first_interval = intervals_.size();
            for (std::size_t e = 0; e < ev.size();) {
                std::size_t f = e;
                int count = 0;
                while (f < ev.size() && ev[f].first == ev[e].first) {
                    ++count;
                    ++f;
                }
                const bool now = odd != ((count & 1) != 0);
                if (now && !open) {
                    open = true;
                    ts = ev[e].first;
                    ps = ev[e].second;
                } else if (!now && open) {
                    open = false;
                    Interval I;
                    I.a = ps;
                    I.b = ev[e].second;
                    I.line = line_faces_.size();
                    for (std::size_t k = i; k < j; ++k)
                        if (segs[k].ta < ev[e].first && segs[k].tb > ts) I.faces.push_back(segs[k].face);
                    std::sort(I.faces.begin(), I.faces.end());
                    I.faces.erase(std::unique(I.faces.begin(), I.faces.end()), I.faces.end());
                    intervals_.push_back(std::move(I));
                }
                odd = now;
                e = f;
            }
            if (intervals_.size() > first_interval) {
                std::vector<std::size_t> on;
                for (std::size_t k = i; k < j; ++k) on.push_back(segs[k].face);
                std::sort(on.begin(), on.end());
                on.erase(std::unique(on.begin(), on.end()), on.end());
                line_faces_.push_back(std::move(on));
            }
            i = j;
        }
        stats_.boundary_intervals = intervals_.size();
        // R = ceil(kGap h 2^s) quanta: the exact product kGap h as p + e (Dekker's product), scaled.
        {
            const cut_detail::DD pe = cut_detail::DD::two_prod(kGap, cell_);
            const double p = pe.hi, e = pe.lo;
            const double y = std::ldexp(p, s_);
            double c = std::ceil(y);
            if (c == y && e > 0.0) c += 1.0;
            R_ = static_cast<std::int64_t>(c);
        }
        if (intervals_.empty()) return;
        for (std::size_t k = 0; k < intervals_.size(); ++k) interval_kind(intervals_[k]);
        for (const Interval &I : intervals_) {
            stats_.free_pieces += I.kind == 1 ? 1 : 0;
            stats_.free_length += I.free_length;
        }
        // The zones' planes (serially: plane ids are made here).
        using cut_detail::I256;
        zones_.assign(intervals_.size(), Zone{});
        for (std::size_t k = 0; k < intervals_.size(); ++k) {
            const Interval &I = intervals_[k];
            Zone &Z = zones_[k];
            auto add_plane = [&](I128 n0, I128 n1, I128 n2, const I256 &d, int inner) {
                I128 n[3] = {n0, n1, n2};
                I256 dd = d;
                const I128 lead = n[0] != 0 ? n[0] : (n[1] != 0 ? n[1] : n[2]);
                if (lead < 0) {
                    for (I128 &c : n) c = -c;
                    dd = -dd;
                    inner = -inner;
                }
                Z.facets.push_back({plane_of(n, dd), inner});
            };
            for (int a = 0; a < 3; ++a) {
                I128 n[3] = {0, 0, 0};
                n[a] = 1;
                const I128 hi = static_cast<I128>(std::max(I.a[static_cast<std::size_t>(a)], I.b[static_cast<std::size_t>(a)])) + R_;
                const I128 lo = static_cast<I128>(std::min(I.a[static_cast<std::size_t>(a)], I.b[static_cast<std::size_t>(a)])) - R_;
                add_plane(n[0], n[1], n[2], I256::of(hi), -1);
                add_plane(n[0], n[1], n[2], I256::of(lo), 1);
            }
            I128 d[3] = {static_cast<I128>(I.b[0]) - I.a[0], static_cast<I128>(I.b[1]) - I.a[1], static_cast<I128>(I.b[2]) - I.a[2]};
            {
                const cut_detail::U128 g = gcd3(d[0], d[1], d[2]);
                for (I128 &c : d) c = cut_detail::exact_div(c, g);
            }
            for (int a = 0; a < 3; ++a) {
                I128 m[3];  // d x e_a
                if (a == 0) {
                    m[0] = 0;
                    m[1] = d[2];
                    m[2] = -d[1];
                } else if (a == 1) {
                    m[0] = -d[2];
                    m[1] = 0;
                    m[2] = d[0];
                } else {
                    m[0] = d[1];
                    m[1] = -d[0];
                    m[2] = 0;
                }
                if (m[0] == 0 && m[1] == 0 && m[2] == 0) continue;
                const cut_detail::U128 g = gcd3(m[0], m[1], m[2]);
                for (I128 &c : m) c = cut_detail::exact_div(c, g);
                const I256 off = I256::mul(m[0], I.a[0]) + I256::mul(m[1], I.a[1]) + I256::mul(m[2], I.a[2]);
                const I128 l1 = (m[0] < 0 ? -m[0] : m[0]) + (m[1] < 0 ? -m[1] : m[1]) + (m[2] < 0 ? -m[2] : m[2]);
                const I256 reach = I256::of(l1 * static_cast<I128>(R_));
                add_plane(m[0], m[1], m[2], off + reach, -1);
                add_plane(m[0], m[1], m[2], off - reach, 1);
            }
        }
    }

    // Whether some part of interval I lies on no face other than those with an edge on its
    // line (exact): each face whose plane holds the line covers the parameters lambda in
    // [0, 1] of a + lambda (b - a) inside its closed triangle (a closed range of exact
    // fractions); the interval is free where their union leaves a gap of positive length.
    void interval_kind(Interval &I) const {
        struct Frac {
            Big n, d;  // d > 0
        };
        auto less = [](const Frac &x, const Frac &y) { return Big::compare(x.n * y.d, y.n * x.d) < 0; };
        const std::vector<std::size_t> &own = line_faces_[I.line];
        // the interval's closed box on the lattice (not from the doubles X0 + 2^-s q, which
        // round by up to 32 quanta, past the 8-quantum margin, and would drop faces that hold
        // the interval, which would then read as free in the message and the counts)
        std::int64_t lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(I.a[static_cast<std::size_t>(a)], I.b[static_cast<std::size_t>(a)]);
            hi[a] = std::max(I.a[static_cast<std::size_t>(a)], I.b[static_cast<std::size_t>(a)]);
        }
        const I128 ab[3] = {static_cast<I128>(I.b[0]) - I.a[0], static_cast<I128>(I.b[1]) - I.a[1], static_cast<I128>(I.b[2]) - I.a[2]};
        std::vector<std::pair<Frac, Frac>> cover;
        for (std::size_t t = 0; t < faces_.size(); ++t) {
            const FaceX &F = faces_[t];
            if (F.plane < 0 || std::binary_search(own.begin(), own.end(), t)) continue;
            bool box = true;
            for (int a = 0; a < 3 && box; ++a) box = F.qhi[a] >= lo[a] && F.qlo[a] <= hi[a];
            if (!box) continue;
            const Plane &P = planes_[static_cast<std::size_t>(F.plane)];
            using cut_detail::I256;
            if (P.n[0] * ab[0] + P.n[1] * ab[1] + P.n[2] * ab[2] != 0) continue;
            if ((I256::mul(P.n[0], I.a[0]) + I256::mul(P.n[1], I.a[1]) + I256::mul(P.n[2], I.a[2]) - P.d4).sign() != 0) continue;
            // the face's own normal (right hand on its corners)
            I128 u[3], w[3];
            for (int a = 0; a < 3; ++a) {
                u[a] = static_cast<I128>(F.q[1][a]) - F.q[0][a];
                w[a] = static_cast<I128>(F.q[2][a]) - F.q[0][a];
            }
            const Big nf[3] = {Big(u[1]) * Big(w[2]) - Big(u[2]) * Big(w[1]), Big(u[2]) * Big(w[0]) - Big(u[0]) * Big(w[2]),
                               Big(u[0]) * Big(w[1]) - Big(u[1]) * Big(w[0])};
            Frac L{Big(0), Big(1)}, H{Big(1), Big(1)};
            bool empty = false;
            for (int k = 0; k < 3 && !empty; ++k) {
                const std::int64_t *U = F.q[k], *V = F.q[(k + 1) % 3];
                const I128 e[3] = {static_cast<I128>(V[0]) - U[0], static_cast<I128>(V[1]) - U[1], static_cast<I128>(V[2]) - U[2]};
                const I128 au[3] = {static_cast<I128>(I.a[0]) - U[0], static_cast<I128>(I.a[1]) - U[1], static_cast<I128>(I.a[2]) - U[2]};
                auto cr = [](const I128 *x, const I128 *y, Big out[3]) {
                    out[0] = Big(x[1]) * Big(y[2]) - Big(x[2]) * Big(y[1]);
                    out[1] = Big(x[2]) * Big(y[0]) - Big(x[0]) * Big(y[2]);
                    out[2] = Big(x[0]) * Big(y[1]) - Big(x[1]) * Big(y[0]);
                };
                Big c0[3], c1[3];
                cr(e, au, c0);
                cr(e, ab, c1);
                const Big f0 = c0[0] * nf[0] + c0[1] * nf[1] + c0[2] * nf[2];
                const Big f1 = c1[0] * nf[0] + c1[1] * nf[1] + c1[2] * nf[2];
                // f0 + lambda f1 >= 0
                if (f1.sign() == 0) {
                    if (f0.sign() < 0) empty = true;
                } else if (f1.sign() > 0) {
                    const Frac b{-f0, f1};  // lambda >= -f0 / f1
                    if (less(L, b)) L = b;
                } else {
                    const Frac b{f0, -f1};  // lambda <= f0 / -f1
                    if (less(b, H)) H = b;
                }
            }
            if (empty || !less(L, H)) continue;
            cover.push_back({L, H});
        }
        std::sort(cover.begin(), cover.end(), [&](const auto &x, const auto &y) { return less(x.first, y.first); });
        Frac at{Big(0), Big(1)};
        double gap = 0.0;
        bool free = false;
        auto value = [](const Frac &x) { return x.n.to_double() / x.d.to_double(); };
        for (const auto &[L, H] : cover) {
            if (less(at, L)) {
                free = true;
                gap += value(L) - value(at);
            }
            if (less(at, H)) at = H;
        }
        const Frac one{Big(1), Big(1)};
        if (less(at, one)) {
            free = true;
            gap += 1.0 - value(at);
        }
        I.kind = free ? 1 : 0;
        double len2 = 0.0;
        for (int a = 0; a < 3; ++a) len2 += static_cast<double>(ab[a]) * static_cast<double>(ab[a]);
        I.free_length = gap * std::sqrt(len2) / S_;
    }

    // ---- refusal (ii): nearly parallel faces close together ---------------------------------

    // A candidate pair: faces f1 < f2 (indices into faces_) on different planes, at under
    // 7.18 degrees, whose projections along f1's dominant axis overlap in positive area O;
    // up: somewhere inside O, 0 < z2 - z1 < 7/4 R along that axis (f2 above f1); down:
    // there -7/4 R < z2 - z1 < 0. area: O's area (m^2, a measure for the message).
    struct LayerPair {
        std::size_t f1 = 0, f2 = 0;
        int axis = 0;
        bool up = false, down = false;
        double area = 0.0;
    };

    // The candidates: faces meeting a common cell's cube grown by gq = ceil(7/4 sqrt 3 R) + 2
    // quanta on the lattice (the slab test exact in integers, the triangle test
    // in each cube's own frame with 1e-6 cells to spare; a superset of every pair within 7/4 R
    // of each other along an axis over an area), on different planes, nearly parallel by the
    // doubles (a superset), then decided exactly (layer_pair).
    void layer_candidates(std::size_t threads) {
        const I128 gq = static_cast<I128>(std::ceil(1.75 * std::sqrt(3.0) * static_cast<double>(R_) * (1.0 + 1e-12))) + 2;
        const std::size_t workers = std::max<std::size_t>(1, worker_threads(threads));
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> found(workers);
        fit_for(faces_.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t w) {
            for (std::size_t t = begin; t < end; ++t) {
                const FaceX &F = faces_[t];
                if (F.plane < 0) continue;
                std::size_t r0[3], r1[3];
                bool any = true;
                for (int a = 0; a < 3 && any; ++a)
                    any = cube_range(a, F.qlo[a], F.qhi[a], gq, 0, n_[static_cast<std::size_t>(a)] - 1, r0[a], r1[a]);
                if (!any) continue;
                std::vector<double> loc[3], half[3];
                for (int a = 0; a < 3; ++a)
                    for (std::size_t i = r0[a]; i <= r1[a]; ++i) {
                        for (int k = 0; k < 3; ++k) loc[a].push_back(local(a, i, F.q[k][a]));
                        const I128 W = G_[static_cast<std::size_t>(a)][i + 1] - G_[static_cast<std::size_t>(a)][i];
                        half[a].push_back(0.5 + static_cast<double>(gq) / static_cast<double>(W) + 1e-6);
                    }
                for (std::size_t z = r0[2]; z <= r1[2]; ++z)
                    for (std::size_t y = r0[1]; y <= r1[1]; ++y)
                        for (std::size_t x = r0[0]; x <= r1[0]; ++x) {
                            const std::size_t ix = 3 * (x - r0[0]), iy = 3 * (y - r0[1]), iz = 3 * (z - r0[2]);
                            const Vec3 t3[3] = {{loc[0][ix], loc[1][iy], loc[2][iz]},
                                                {loc[0][ix + 1], loc[1][iy + 1], loc[2][iz + 1]},
                                                {loc[0][ix + 2], loc[1][iy + 2], loc[2][iz + 2]}};
                            const Vec3 hw{half[0][x - r0[0]], half[1][y - r0[1]], half[2][z - r0[2]]};
                            if (meets(t3, hw)) found[w].push_back({index(x, y, z), t});
                        }
            }
        });
        std::vector<std::pair<std::size_t, std::size_t>> all;
        for (auto &f : found) all.insert(all.end(), f.begin(), f.end());
        std::sort(all.begin(), all.end());
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        const double close = 63.0 / 64.0 - 1e-9;
        for (std::size_t i = 0; i < all.size();) {
            std::size_t j = i;
            while (j < all.size() && all[j].first == all[i].first) ++j;
            for (std::size_t x = i; x < j; ++x)
                for (std::size_t y = x + 1; y < j; ++y) {
                    const FaceX &A = faces_[all[x].second], &B = faces_[all[y].second];
                    if (A.plane == B.plane) continue;
                    const Plane &P = planes_[static_cast<std::size_t>(A.plane)], &Q = planes_[static_cast<std::size_t>(B.plane)];
                    const double c = P.un[0] * Q.un[0] + P.un[1] * Q.un[1] + P.un[2] * Q.un[2];
                    if (c * c <= close) continue;
                    pairs.push_back({std::min(all[x].second, all[y].second), std::max(all[x].second, all[y].second)});
                }
            i = j;
        }
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
        stats_.layer_pairs = pairs.size();
        layer_pairs_.clear();
        std::vector<LayerPair> tested(pairs.size());
        std::vector<char> kept(pairs.size(), 0);
        fit_for(pairs.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t k = begin; k < end; ++k) kept[k] = layer_pair(pairs[k].first, pairs[k].second, tested[k]) ? 1 : 0;
        });
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            if (!kept[k]) continue;
            layer_pairs_.push_back(tested[k]);
            faces_[pairs[k].first].layer = faces_[pairs[k].second].layer = true;
        }
        stats_.layer_candidates = layer_pairs_.size();
    }

    // The exact tests of a pair (DESIGN 5.3): the angle (n1.n2)^2 > 63/64 |n1|^2 |n2|^2; the
    // projections along f1's dominant axis a (lattice points, 2D orientations in 128 bits)
    // overlap in positive area (no edge line separates them); the overlap O is f1's triangle
    // clipped by f2's edge lines, its vertices held as meets of two edge lines; and the sign
    // of z2 - z1 and of 4 (z2 - z1) -+ 7 R at O's vertices (exact, in Big).
    bool layer_pair(std::size_t f1, std::size_t f2, LayerPair &out) const {
        const FaceX &A = faces_[f1], &B = faces_[f2];
        const Plane &P1 = planes_[static_cast<std::size_t>(A.plane)], &P2 = planes_[static_cast<std::size_t>(B.plane)];
        const int a = A.axis, b = (a + 1) % 3, c = (a + 2) % 3;
        if (P2.n[a] == 0) return false;  // cannot happen under 7.18 degrees of f1's dominant axis
        // 2D points, counter-clockwise
        I128 p[2][3][2];
        for (int k = 0; k < 3; ++k) {
            p[0][k][0] = A.q[k][b];
            p[0][k][1] = A.q[k][c];
            p[1][k][0] = B.q[k][b];
            p[1][k][1] = B.q[k][c];
        }
        auto orient = [](const I128 *u, const I128 *v, const I128 *w) {
            const I128 x = (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]);
            return x > 0 ? 1 : (x < 0 ? -1 : 0);
        };
        for (int t = 0; t < 2; ++t) {
            const int o = orient(p[t][0], p[t][1], p[t][2]);
            if (o == 0) return false;
            if (o < 0)
                for (int i = 0; i < 2; ++i) std::swap(p[t][1][i], p[t][2][i]);
        }
        for (int t = 0; t < 2; ++t)
            for (int k = 0; k < 3; ++k) {
                bool separates = true;
                for (int j = 0; j < 3 && separates; ++j) separates = orient(p[t][k], p[t][(k + 1) % 3], p[1 - t][j]) <= 0;
                if (separates) return false;
            }
        // The angle, exactly where the doubles (within 1e-15 of it) cannot tell.
        {
            const double cs = P1.un[0] * P2.un[0] + P1.un[1] * P2.un[1] + P1.un[2] * P2.un[2];
            if (!(cs * cs > 63.0 / 64.0 + 1e-9)) {
                const Big dot = Big(P1.n[0]) * Big(P2.n[0]) + Big(P1.n[1]) * Big(P2.n[1]) + Big(P1.n[2]) * Big(P2.n[2]);
                const Big l1 = Big(P1.n[0]) * Big(P1.n[0]) + Big(P1.n[1]) * Big(P1.n[1]) + Big(P1.n[2]) * Big(P1.n[2]);
                const Big l2 = Big(P2.n[0]) * Big(P2.n[0]) + Big(P2.n[1]) * Big(P2.n[1]) + Big(P2.n[2]) * Big(P2.n[2]);
                if (Big::compare(Big(64) * dot * dot, Big(63) * l1 * l2) <= 0) return false;
            }
        }
        // Lines a x + b y = c through two lattice points, their left side (inside) positive.
        struct Line {
            I128 a, b;
            Big c;
        };
        auto line = [](const I128 *u, const I128 *v) {
            Line L;
            L.a = -(v[1] - u[1]);
            L.b = v[0] - u[0];
            L.c = Big(L.a) * Big(u[0]) + Big(L.b) * Big(u[1]);
            return L;
        };
        struct V2 {
            Big X, Y, W;  // (X / W, Y / W), W > 0
        };
        auto meet = [](const Line &L, const Line &M) {
            V2 r;
            r.W = Big(L.a) * Big(M.b) - Big(M.a) * Big(L.b);
            r.X = L.c * Big(M.b) - M.c * Big(L.b);
            r.Y = Big(L.a) * M.c - Big(M.a) * L.c;
            if (r.W.sign() < 0) {
                r.W = -r.W;
                r.X = -r.X;
                r.Y = -r.Y;
            }
            return r;
        };
        auto value = [](const Line &L, const V2 &v) { return (Big(L.a) * v.X + Big(L.b) * v.Y - L.c * v.W).sign(); };
        std::vector<Line> lines;
        std::vector<V2> pv;
        std::vector<int> pe;  // the line of the edge from vertex i to i + 1
        for (int k = 0; k < 3; ++k) {
            lines.push_back(line(p[0][k], p[0][(k + 1) % 3]));
            pv.push_back({Big(p[0][k][0]), Big(p[0][k][1]), Big(1)});
            pe.push_back(k);
        }
        for (int k = 0; k < 3; ++k) {
            lines.push_back(line(p[1][k], p[1][(k + 1) % 3]));
            const int L = 3 + k;
            std::vector<int> sg(pv.size());
            for (std::size_t i = 0; i < pv.size(); ++i) sg[i] = value(lines[static_cast<std::size_t>(L)], pv[i]);
            std::vector<V2> nv;
            std::vector<int> ne;
            const std::size_t m = pv.size();
            for (std::size_t i = 0; i < m; ++i) {
                const std::size_t j = (i + 1) % m;
                const int sa = sg[i], sb = sg[j];
                if (sa > 0) {
                    nv.push_back(pv[i]);
                    ne.push_back(pe[i]);
                    if (sb < 0) {
                        nv.push_back(meet(lines[static_cast<std::size_t>(pe[i])], lines[static_cast<std::size_t>(L)]));
                        ne.push_back(L);
                    }
                } else if (sa == 0) {
                    nv.push_back(pv[i]);
                    ne.push_back(sb < 0 ? L : pe[i]);
                } else if (sb > 0) {
                    nv.push_back(meet(lines[static_cast<std::size_t>(pe[i])], lines[static_cast<std::size_t>(L)]));
                    ne.push_back(pe[i]);
                }
            }
            pv.swap(nv);
            pe.swap(ne);
            if (pv.size() < 3) return false;  // cannot happen: no line separates them
        }
        // z_i = (d_i - n_ib x - n_ic y) / n_ia; D = n1a n2a W (z2 - z1), compared with 7R/4.
        const Big n1a(P1.n[a]), n1b(P1.n[b]), n1c(P1.n[c]), n2a(P2.n[a]), n2b(P2.n[b]), n2c(P2.n[c]);
        const int s12 = ((P1.n[a] > 0) == (P2.n[a] > 0)) ? 1 : -1;
        const Big K = n1a * n2a * Big(7 * static_cast<I128>(R_));
        bool pos = false, neg = false, below = false, above = false;  // z2 - z1 > 0, < 0, < 7R/4, > -7R/4 at a vertex
        for (const V2 &v : pv) {
            const Big D = n1a * (P2.d * v.W - n2b * v.X - n2c * v.Y) - n2a * (P1.d * v.W - n1b * v.X - n1c * v.Y);
            const int sd = D.sign() * s12;
            const int lo = (Big(4) * D - K * v.W).sign() * s12;  // the sign of 4 (z2 - z1) - 7 R
            const int hi = (Big(4) * D + K * v.W).sign() * s12;  // the sign of 4 (z2 - z1) + 7 R
            pos = pos || sd > 0;
            neg = neg || sd < 0;
            below = below || lo < 0;
            above = above || hi > 0;
        }
        // z2 - z1 is linear over O. If the planes are parallel it is constant there and every
        // vertex says the same; otherwise it is not constant on O, and 0 < z2 - z1 < 7R/4
        // somewhere inside O iff it is > 0 at some vertex and < 7R/4 at some vertex (and
        // likewise below 0).
        out.f1 = f1;
        out.f2 = f2;
        out.axis = a;
        out.up = pos && below;
        out.down = neg && above;
        if (!out.up && !out.down) return false;
        // O's area by the shoelace formula on the vertices taken from the first one: each difference is
        // formed exactly, (X_i W_0 - X_0 W_i) / (W_i W_0), and rounded once, so that a sliver of O far from
        // the lattice's origin keeps its size (from the coordinates themselves, each product would round
        // by about 2^-52 m^2).
        std::vector<double> dx(pv.size()), dy(pv.size());
        for (std::size_t i = 0; i < pv.size(); ++i) {
            const double w = (pv[i].W * pv[0].W).to_double();
            dx[i] = (pv[i].X * pv[0].W - pv[0].X * pv[i].W).to_double() / w;
            dy[i] = (pv[i].Y * pv[0].W - pv[0].Y * pv[i].W).to_double() / w;
        }
        double area = 0.0;
        for (std::size_t i = 1; i + 1 < pv.size(); ++i) area += dx[i] * dy[i + 1] - dx[i + 1] * dy[i];
        out.area = 0.5 * std::abs(area) * std::ldexp(1.0, -2 * s_);
        return true;
    }

    // The odd intervals named, each as what it is (DESIGN 5.2, T-5): free edges (a part on
    // no other face) and contact edges (lying wholly on other faces, or shared by an odd
    // number of faces), with the faces (their indices in the model) and the ends (m).
    std::string describe_intervals(const std::vector<int> &which) const {
        auto point = [&](const std::array<std::int64_t, 3> &q) {
            std::string s = "(";
            for (int a = 0; a < 3; ++a) {
                s += std::to_string(X0_[a] + std::ldexp(static_cast<double>(q[static_cast<std::size_t>(a)]), -s_));
                s += a < 2 ? ", " : ")";
            }
            return s;
        };
        std::string out[2];
        std::size_t count[2] = {0, 0};
        for (const int k : which) {
            const Interval &I = intervals_[static_cast<std::size_t>(k)];
            const int kind = I.kind == 1 ? 0 : 1;
            if (++count[kind] > 6) continue;
            std::string &s = out[kind];
            if (!s.empty()) s += "; ";
            s += "faces";
            for (std::size_t j = 0; j < I.faces.size() && j < 4; ++j) s += " " + std::to_string(faces_[I.faces[j]].face);
            if (I.faces.size() > 4) s += " ...";
            s += " from " + point(I.a) + " to " + point(I.b);
        }
        std::string r;
        if (count[0]) {
            r += "these free edges (an edge of the model that ends in the air): " + out[0];
            if (count[0] > 6) r += "; and " + std::to_string(count[0] - 6) + " more";
        }
        if (count[1]) {
            if (!r.empty()) r += "; and ";
            r += "these contact edges (where a face's edge lies on another face or is shared by an odd number of faces): " + out[1];
            if (count[1] > 6) r += "; and " + std::to_string(count[1] - 6) + " more";
        }
        return r;
    }

    // Candidate (cube, zone) pairs: cubes not on the grid's edge that the zone may meet with
    // positive volume (a superset, exact on the lattice: the interval in
    // steps of about half a cell, step i from a + floor((b - a) i / steps) to the next, each
    // step's box grown by R + 1 quanta, the cubes by cube_range).
    std::vector<std::pair<std::size_t, int>> zone_cubes() const {
        std::vector<std::pair<std::size_t, int>> out;
        for (std::size_t k = 0; k < intervals_.size(); ++k) {
            const Interval &I = intervals_[k];
            double len = 0.0;
            for (int a = 0; a < 3; ++a)
                len = std::max(len, std::abs(static_cast<double>(I.b[static_cast<std::size_t>(a)]) -
                                             static_cast<double>(I.a[static_cast<std::size_t>(a)])) / S_);
            const std::size_t steps = static_cast<std::size_t>(std::ceil(2.0 * len)) + 1;
            auto at = [&](int a, std::size_t i) {  // floor(a + (b - a) i / steps), exact
                const I128 A = I.a[static_cast<std::size_t>(a)], D = static_cast<I128>(I.b[static_cast<std::size_t>(a)]) - A;
                const I128 num = D * static_cast<I128>(i), den = static_cast<I128>(steps);
                I128 q = num / den;
                if (num % den != 0 && num < 0) q -= 1;
                return A + q;
            };
            for (std::size_t i = 0; i < steps; ++i) {
                std::size_t r0[3], r1[3];
                bool any = true;
                for (int a = 0; a < 3 && any; ++a) {
                    const I128 p0 = at(a, i), p1 = at(a, i + 1);
                    any = cube_range(a, std::min(p0, p1) - R_, std::max(p0, p1) + 1 + R_, 0, 1,
                                     n_[static_cast<std::size_t>(a)] - 2, r0[a], r1[a]);
                }
                if (!any) continue;
                for (std::size_t z = r0[2]; z <= r1[2]; ++z)
                    for (std::size_t y = r0[1]; y <= r1[1]; ++y)
                        for (std::size_t x = r0[0]; x <= r1[0]; ++x) out.push_back({index(x, y, z), static_cast<int>(k)});
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // ---- 1. faces into cubes -------------------------------------------------------------------

    std::size_t index(std::size_t x, std::size_t y, std::size_t z) const { return x + sy_ * y + sz_ * z; }
    std::array<std::size_t, 3> at(std::size_t i) const { return {i % n_[0], (i / sy_) % n_[1], i / sz_}; }
    bool edge(std::size_t i) const {
        const auto p = at(i);
        for (int a = 0; a < 3; ++a)
            if (p[static_cast<std::size_t>(a)] == 0 || p[static_cast<std::size_t>(a)] + 1 == n_[static_cast<std::size_t>(a)]) return true;
        return false;
    }
    std::size_t depth(std::size_t i) const {
        const auto p = at(i);
        std::size_t d = ~std::size_t{0};
        for (int a = 0; a < 3; ++a)
            d = std::min({d, p[static_cast<std::size_t>(a)], n_[static_cast<std::size_t>(a)] - 1 - p[static_cast<std::size_t>(a)]});
        return d;
    }
    std::size_t neighbour(std::size_t i, int k) const {
        const long long step = kSteps[k][0] + kSteps[k][1] * static_cast<long long>(sy_) +
                               kSteps[k][2] * static_cast<long long>(sz_);
        return static_cast<std::size_t>(static_cast<long long>(i) + step);
    }
    int direction(std::size_t a, std::size_t b) const {
        const auto p = at(a), q = at(b);
        const long long d[3] = {static_cast<long long>(q[0]) - static_cast<long long>(p[0]),
                                static_cast<long long>(q[1]) - static_cast<long long>(p[1]),
                                static_cast<long long>(q[2]) - static_cast<long long>(p[2])};
        for (int k = 0; k < 18; ++k)
            if (kSteps[k][0] == d[0] && kSteps[k][1] == d[1] && kSteps[k][2] == d[2]) return k;
        return -1;
    }
    std::size_t distance(std::size_t a, std::size_t b) const {
        const auto p = at(a), q = at(b);
        long long d = 0;
        for (int k = 0; k < 3; ++k)
            d = std::max(d, std::abs(static_cast<long long>(q[static_cast<std::size_t>(k)]) - static_cast<long long>(p[static_cast<std::size_t>(k)])));
        return static_cast<std::size_t>(d);
    }
    static int opposite(int k) { return k < 6 ? k ^ 1 : 6 + 4 * ((k - 6) / 4) + 3 - (k - 6) % 4; }

    // parallel_for for the fit: every worker first checks the rounding mode; in the checking
    // build it also records whether its part of the fit raised an underflow.
    template <typename Body>
    void fit_for(std::size_t n, std::size_t threads, Body body) const {
        parallel_for(n, threads, [&](std::size_t begin, std::size_t end, std::size_t w) {
            cut_detail::require_rounding("a worker thread of the fit");
#ifdef ACOUSTIC_SPACES_CUT_CHECK
            std::feclearexcept(FE_UNDERFLOW);
#endif
            body(begin, end, w);
#ifdef ACOUSTIC_SPACES_CUT_CHECK
            if (std::fetestexcept(FE_UNDERFLOW)) underflow_ = 1;
#endif
        });
    }

    template <typename Item>
    void in_turn(std::size_t n, std::size_t threads, Item item) {
        const std::size_t T = std::max<std::size_t>(1, std::min(worker_threads(threads), n));
        fit_for(T, T, [&](std::size_t begin, std::size_t end, std::size_t) {
            for (std::size_t w = begin; w < end; ++w)
                for (std::size_t k = w; k < n; k += T) item(k);
        });
    }

    // Every candidate test reads the lattice, never X0 + 2^-s q - origin in doubles. That
    // double rounds by ulp(|x|) / h cells (6e-6 cells at 2e9 m with 4 cm cells), which passes
    // the 1e-6-cell margins, and the
    // cube of index i is [G_i, G_i+1] on Q, not [origin + i h, origin + (i + 1) h].
    //
    // The cubes i in [first, last] along axis a whose closed slab [G_i - grow, G_i+1 + grow]
    // meets [lo, hi] on the lattice (exact, integers); false when none.
    bool cube_range(int a, I128 lo, I128 hi, I128 grow, std::size_t first, std::size_t last, std::size_t &i0,
                    std::size_t &i1) const {
        const std::vector<I128> &G = G_[static_cast<std::size_t>(a)];
        // the largest i with G_i - grow <= hi, and the smallest i with G_i+1 + grow >= lo
        const long long top = static_cast<long long>(std::upper_bound(G.begin(), G.end(), hi + grow) - G.begin()) - 1;
        const long long bot = static_cast<long long>(std::lower_bound(G.begin(), G.end(), lo - grow) - G.begin()) - 1;
        const long long b = std::max(bot, static_cast<long long>(first)), t = std::min(top, static_cast<long long>(last));
        if (b > t) return false;
        i0 = static_cast<std::size_t>(b);
        i1 = static_cast<std::size_t>(t);
        return true;
    }
    // Lattice coordinate q along axis a in cube i's own frame: (q - G_i) / (G_i+1 - G_i) - 0.5,
    // so that the cube is exactly [-0.5, 0.5] there. The two differences are exact integers;
    // the conversions and the division round by under 3 2^-53 of the value, and the value is
    // under 2^27 (|q - G_i| < 2^62, G_i+1 - G_i > 2^35 under R0), so under 2^-24 cells.
    double local(int a, std::size_t i, std::int64_t q) const {
        const std::vector<I128> &G = G_[static_cast<std::size_t>(a)];
        const I128 g0 = G[i], w = G[i + 1] - g0;
        return static_cast<double>(static_cast<I128>(q) - g0) / static_cast<double>(w) - 0.5;
    }

    // Whether a triangle (in a cube's own frame, local(), where the cube is [-0.5, 0.5]^3)
    // meets the box of half-widths hw: the separating axes of a triangle and a box (the box's
    // three, the triangle's normal, the nine cross products of their edges). With hw = 0.5 +
    // 1e-6 (+ a growth) a superset of the exact test: the coordinates are within 2^-24 cells of
    // the exact ones (local()), and each projection rounds by under 3 2^-53 |ax|_1 max|t|, under
    // 5e-8 |ax|_1; any axis that separates proves the two sets apart.
    static bool meets(const Vec3 t[3], const Vec3 &hw) {
        const Vec3 e[3] = {t[1] - t[0], t[2] - t[1], t[0] - t[2]};
        auto separated = [&](const Vec3 &ax) {
            const double p0 = dot(ax, t[0]), p1 = dot(ax, t[1]), p2 = dot(ax, t[2]);
            const double r = hw.x * std::abs(ax.x) + hw.y * std::abs(ax.y) + hw.z * std::abs(ax.z);
            return std::min({p0, p1, p2}) > r || std::max({p0, p1, p2}) < -r;
        };
        for (int a = 0; a < 3; ++a) {
            const double lo = std::min({t[0][a], t[1][a], t[2][a]}), hi = std::max({t[0][a], t[1][a], t[2][a]});
            if (lo > hw[a] || hi < -hw[a]) return false;
        }
        if (separated(cross(e[0], e[1]))) return false;
        const Vec3 u[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (const Vec3 &a : u)
            for (const Vec3 &b : e)
                if (separated(cross(a, b))) return false;
        return true;
    }

    // Candidate (cube, face) pairs: cubes of cells not on the grid's edge that the face's
    // triangle meets by meets() (a superset; each pair is kept or dropped by the exact clip
    // in cut_cube).
    std::vector<std::pair<std::size_t, std::size_t>> faces_into_cubes(std::size_t threads) const {
        const std::size_t workers = std::max<std::size_t>(1, worker_threads(threads));
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> found(workers);
        fit_for(faces_.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t w) {
            for (std::size_t t = begin; t < end; ++t) {
                const FaceX &F = faces_[t];
                if (F.plane < 0 || F.mem_axis >= 0) continue;
                std::size_t r0[3], r1[3];
                bool any = true;
                for (int a = 0; a < 3 && any; ++a)
                    any = cube_range(a, F.qlo[a], F.qhi[a], 0, 1, n_[static_cast<std::size_t>(a)] - 2, r0[a], r1[a]);
                if (!any) continue;
                // each corner's coordinate in each cube's own frame, per axis
                std::vector<double> loc[3];
                for (int a = 0; a < 3; ++a)
                    for (std::size_t i = r0[a]; i <= r1[a]; ++i)
                        for (int k = 0; k < 3; ++k) loc[a].push_back(local(a, i, F.q[k][a]));
                const Vec3 hw{0.5 + 1e-6, 0.5 + 1e-6, 0.5 + 1e-6};
                for (std::size_t z = r0[2]; z <= r1[2]; ++z)
                    for (std::size_t y = r0[1]; y <= r1[1]; ++y)
                        for (std::size_t x = r0[0]; x <= r1[0]; ++x) {
                            const std::size_t ix = 3 * (x - r0[0]), iy = 3 * (y - r0[1]), iz = 3 * (z - r0[2]);
                            const Vec3 t3[3] = {{loc[0][ix], loc[1][iy], loc[2][iz]},
                                                {loc[0][ix + 1], loc[1][iy + 1], loc[2][iz + 1]},
                                                {loc[0][ix + 2], loc[1][iy + 2], loc[2][iz + 2]}};
                            if (meets(t3, hw)) found[w].push_back({index(x, y, z), t});
                        }
            }
        });
        std::vector<std::pair<std::size_t, std::size_t>> all;
        for (auto &f : found) all.insert(all.end(), f.begin(), f.end());
        std::sort(all.begin(), all.end());
        return all;
    }

    // The lattice corner of cell i's cube.
    void corner(std::size_t i, std::int64_t C[3]) const {
        const auto p = at(i);
        for (int a = 0; a < 3; ++a) C[a] = small(G_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)]]);
    }

    // A face as a polygon in table T: its lattice points, each on its plane and its two
    // edges' coverage planes.
    Poly face_poly(Table &T, const FaceX &F) const {
        Poly g;
        g.plane = F.plane;
        for (int k = 0; k < 3; ++k) {
            g.v.push_back(T.lattice(F.q[k], F.plane, F.e[(k + 2) % 3], F.e[k], F.w[k]));
            g.e.push_back(F.e[k]);
        }
        return g;
    }

    // Cuts g by plane p in table T and keeps the side `keep` (+1: n.x >= d, -1: <=): false
    // when no part of positive area is left. sig: scratch.
    template <typename Make>
    static bool clip(Table &T, Poly &g, int p, int keep, Make &&make) {
        thread_local std::vector<int> s;
        thread_local std::vector<double> f, b;
        const std::size_t m = g.v.size();
        s.resize(m);
        f.resize(m);
        b.resize(m);
        bool in = false, out = false;
        for (std::size_t i = 0; i < m; ++i) {
            s[i] = T.side(g.v[i], p, f[i], b[i]);
            in = in || s[i] * keep > 0;
            out = out || s[i] * keep < 0;
        }
        if (!in) return false;
        if (!out) return true;
        Poly r;
        auto cross = [&](std::size_t i) {
            const std::size_t j = (i + 1) % m;
            return make(T, g, i, j, s[i], f[i], b[i], f[j], b[j], p);
        };
        if (keep > 0) split_poly(g, s.data(), p, cross, nullptr, &r);
        else split_poly(g, s.data(), p, cross, &r, nullptr);
        g = std::move(r);
        return true;
    }
    // The point where edge i (to j) of polygon g crosses plane p: on g's plane, the edge's
    // plane and p.
    static int poly_cross(Table &T, const Poly &g, std::size_t i, std::size_t j, int si, double fi, double bi, double fj,
                          double bj, int p) {
        return T.cross(g.v[i], g.v[j], si, fi, bi, fj, bj, g.plane, g.e[i], p);
    }
    // ---- 2-3. the cubes: trees, caps, covering ---------------------------------------------------

    // A face of a convex polyhedron: its polygon (on a grid plane of the cube, or on the plane
    // of tree node `node`, the polyhedron on its + side when plus), its points in order
    // counter-clockwise seen from outside.
    struct Facet {
        Poly g;
        int node = -1;
        bool plus = false;
    };
    using Polyhedron = std::vector<Facet>;
    // A node splits by plane `plane`; child[0] its - side, child[1] its + side: a node's index,
    // or ~leaf for a leaf.
    struct Node {
        int plane;
        std::array<int, 2> child;
    };
    using Walls = std::vector<std::pair<std::int32_t, double>>;

    // One self-check's result: the largest residual relative to its threshold (a comparison
    // fails above 1), the count of failed comparisons, and the first failure's description.
    struct SelfCheck {
        double worst = 0.0;
        std::size_t failed = 0;
        std::string first;
        // A stored value against the same value formed apart. The threshold is 1e-9 max(1, scale),
        // scale being the sum of the magnitudes of the terms of the local sum; with scale < 0 the
        // comparison is exact (the two are added in the same order, or are not sums).
        template <typename Place>
        void compare(double stored, double formed, double scale, Place place) {
            const double tau = scale < 0.0 ? 0.0 : 1e-9 * (std::max)(1.0, scale);
            const double d = std::abs(stored - formed);
            const double r = stored == formed ? 0.0 : (tau > 0.0 ? d / tau : std::numeric_limits<double>::infinity());
            if (r > worst || r != r) worst = r;
            if (!(r <= 1.0)) {
                ++failed;
                if (first.empty()) first = place() + ": " + exact_text(stored) + " against " + exact_text(formed) + " (threshold " + exact_text(tau) + ")";
            }
        }
        void merge(const SelfCheck &o) {
            if (o.worst > worst || o.worst != o.worst) worst = o.worst;
            if (first.empty()) first = o.first;
            failed += o.failed;
        }
    };
    static std::string exact_text(double x) {
        char t[40];
        std::snprintf(t, sizeof t, "%.17g", x);
        return t;
    }
    // Writes a check's result into Stats and throws std::logic_error if it failed.
    static void finish(const SelfCheck &k, const char *name, double &worst, std::size_t &failed) {
        worst = k.worst;
        failed = k.failed;
        if (k.failed) throw std::logic_error(std::string("fitted walls: self-check ") + name + " failed at " + k.first);
    }
    std::string cell_text(std::size_t i) const {
        const auto p = at(i);
        return "cell " + std::to_string(i) + " (" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ", " + std::to_string(p[2]) + ")";
    }

    // One cube's result.
    struct Cube {
        std::size_t cell = 0;
        bool tree = false;              // holds faces (a cube with membranes only has none)
        std::vector<Node> nodes;        // its tree (root: node 0, or leaf 0 without nodes)
        std::vector<int> leaf_region;   // each leaf's region
        std::vector<double> volume;     // each region's volume (cells)
        std::vector<Walls> wall;        // each region's walls inside the cube
        int centre = 0;                 // the region holding the centre
        // Refusal (i), where zones meet the cube (else empty): each leaf's zone (-1: none),
        // each leaf's region without the zones (-1: in a zone), their count, and the pairs
        // (region without the zones, zone) joined by a bare facet in the cube.
        std::vector<int> leaf_zone, leaf_region_z;
        int regions_z = 0;
        std::vector<std::pair<int, int>> zone_adj;
        // Refusal (ii): (face, region on its plane's + side, region on its - side) for each
        // part of positive area of a candidate face in the cube.
        std::vector<std::array<std::int64_t, 3>> layer_sides;
        std::size_t planes = 0, leaves = 0, exact_calls = 0, refined = 0;
        double face_area = 0.0, covered = 0.0, overlap = 0.0;
        // The partition of the facets between leaves (Stats::partition_area and its sides).
        double parted = 0.0, parted_plus = 0.0, parted_minus = 0.0;
        SelfCheck k_wall, k_volume;  // K1 and K2
    };

    static double total(const Walls &w) {
        double t = 0.0;
        for (const auto &e : w) t += e.second;
        return t;
    }

    static void add(Walls &w, std::int32_t m, double a) {
        if (!(a > 0.0)) return;
        for (auto &e : w) {
            if (e.first == m) {
                e.second += a;
                return;
            }
        }
        w.push_back({m, a});
    }

    static int root(const std::vector<Node> &nodes) { return nodes.empty() ? ~0 : 0; }

    // The side of point k against plane p in T, with the signs of every point kept for one
    // plane at a time (see Signs).
    struct SignCache {
        std::vector<std::uint32_t> stamp;
        std::vector<std::int8_t> s;
        std::vector<double> f, b;
        std::uint32_t now = 0;
        void ensure(std::size_t n) {
            if (stamp.size() >= n) return;
            stamp.resize(n + n / 2 + 64, 0);
            s.resize(stamp.size());
            f.resize(stamp.size());
            b.resize(stamp.size());
        }
        void begin(std::size_t n) {
            ensure(n);
            if (++now == 0) {
                std::fill(stamp.begin(), stamp.end(), 0);
                now = 1;
            }
        }
        int get(Table &T, int k, int p) {
            const auto i = static_cast<std::size_t>(k);
            ensure(i + 1);
            if (stamp[i] == now) return s[i];
            double fv, bv;
            const int r = T.side(k, p, fv, bv);
            stamp[i] = now;
            s[i] = static_cast<std::int8_t>(r);
            f[i] = fv;
            b[i] = bv;
            return r;
        }
        void set_zero(int k) {
            const auto i = static_cast<std::size_t>(k);
            ensure(i + 1);
            stamp[i] = now;
            s[i] = 0;
            f[i] = 0.0;
            b[i] = 0.0;
        }
    };

    // Cuts polygon g down the tree from `from` (a node, or ~leaf): each part of positive
    // area with its leaf. A part with no point strictly on a side of a node's plane goes to
    // the other side whole.
    static void filter(Table &T, const std::vector<Node> &nodes, int from, const Poly &g,
                       std::vector<std::pair<int, Poly>> &out) {
        if (from < 0) {
            out.emplace_back(~from, g);
            return;
        }
        thread_local std::vector<std::pair<int, Poly>> stack;
        stack.clear();
        stack.emplace_back(from, g);
        thread_local std::vector<int> s;
        thread_local std::vector<double> f, b;
        while (!stack.empty()) {
            std::pair<int, Poly> top = std::move(stack.back());
            stack.pop_back();
            if (top.first < 0) {
                out.emplace_back(~top.first, std::move(top.second));
                continue;
            }
            const Node &N = nodes[static_cast<std::size_t>(top.first)];
            const Poly &v = top.second;
            const std::size_t m = v.v.size();
            s.resize(m);
            f.resize(m);
            b.resize(m);
            bool pos = false, neg = false;
            for (std::size_t i = 0; i < m; ++i) {
                s[i] = T.side(v.v[i], N.plane, f[i], b[i]);
                pos = pos || s[i] > 0;
                neg = neg || s[i] < 0;
            }
            if (!neg) {
                stack.emplace_back(N.child[1], std::move(top.second));
            } else if (!pos) {
                stack.emplace_back(N.child[0], std::move(top.second));
            } else {
                Poly minus, plus;
                auto cross = [&](std::size_t i) {
                    const std::size_t j = (i + 1) % m;
                    return T.cross(v.v[i], v.v[j], s[i], f[i], b[i], f[j], b[j], v.plane, v.e[i], N.plane);
                };
                split_poly(v, s.data(), N.plane, cross, &minus, &plus);
                stack.emplace_back(N.child[0], std::move(minus));
                stack.emplace_back(N.child[1], std::move(plus));
            }
        }
    }

    // Splits polyhedron P by plane p into minus and plus, both closed by a cap on p cut at
    // node `node`. Every point's side is the exact sign; a new point on an edge is made once
    // per edge (made, by the edge's two points) and shared by the facets on it. The cap's
    // points come from the edges of the + side's facets that lie on p, chained round it.
    // False when P lies on one side of p (no point strictly on one of them).
    static bool split_leaf(Table &T, SignCache &C, const Polyhedron &P, int p, int node, Polyhedron &minus,
                           Polyhedron &plus, std::vector<std::pair<std::uint64_t, int>> &made) {
        C.begin(T.v.size());
        bool pos = false, neg = false;
        for (const Facet &f : P)
            for (const int k : f.g.v) {
                const int s = C.get(T, k, p);
                pos = pos || s > 0;
                neg = neg || s < 0;
            }
        if (!pos || !neg) return false;
        minus.clear();
        plus.clear();
        std::vector<int> s;
        for (const Facet &f : P) {
            const std::size_t m = f.g.v.size();
            s.resize(m);
            bool fp = false, fn = false;
            for (std::size_t i = 0; i < m; ++i) {
                s[i] = C.get(T, f.g.v[i], p);
                fp = fp || s[i] > 0;
                fn = fn || s[i] < 0;
            }
            if (!fn) {
                plus.push_back(f);
                continue;
            }
            if (!fp) {
                minus.push_back(f);
                continue;
            }
            auto cross = [&](std::size_t i) {
                const int a = f.g.v[i], b = f.g.v[(i + 1) % m];
                const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(std::min(a, b))) << 32) |
                                          static_cast<std::uint32_t>(std::max(a, b));
                for (const auto &e : made)
                    if (e.first == key) return e.second;
                const auto ia = static_cast<std::size_t>(a), ib = static_cast<std::size_t>(b);
                const int x = T.cross(a, b, C.s[ia], C.f[ia], C.b[ia], C.f[ib], C.b[ib], f.g.plane, f.g.e[i], p);
                C.set_zero(x);
                made.push_back({key, x});
                return x;
            };
            Facet fm, fq;
            split_poly(f.g, s.data(), p, cross, &fm.g, &fq.g);
            fm.node = fq.node = f.node;
            fm.plus = fq.plus = f.plus;
            minus.push_back(std::move(fm));
            plus.push_back(std::move(fq));
        }
        // The cap: each + facet's edge on p, reversed, chained.
        std::vector<std::array<int, 3>> edges;  // from, to, the facet's plane
        for (const Facet &f : plus) {
            const std::size_t m = f.g.v.size();
            for (std::size_t i = 0; i < m; ++i) {
                const int a = f.g.v[i], b = f.g.v[(i + 1) % m];
                if (C.get(T, a, p) == 0 && C.get(T, b, p) == 0) edges.push_back({b, a, f.g.plane});
            }
        }
        Facet cap;
        cap.g.plane = p;
        cap.node = node;
        cap.plus = true;
        if (edges.size() < 3) throw std::logic_error("fitted walls: a cap of fewer than three edges");
        int at = edges[0][0];
        for (std::size_t n = 0; n < edges.size(); ++n) {
            std::size_t k = 0;
            while (k < edges.size() && edges[k][0] != at) ++k;
            if (k == edges.size()) throw std::logic_error("fitted walls: a cap that does not close");
            cap.g.v.push_back(at);
            cap.g.e.push_back(edges[k][2]);
            at = edges[k][1];
        }
        if (at != edges[0][0]) throw std::logic_error("fitted walls: a cap that does not close");
        Facet rev;
        rev.g.plane = p;
        rev.node = node;
        rev.plus = false;
        const std::size_t m = cap.g.v.size();
        for (std::size_t j = 0; j < m; ++j) {
            rev.g.v.push_back(cap.g.v[m - 1 - j]);
            rev.g.e.push_back(j + 1 < m ? cap.g.e[m - 2 - j] : cap.g.e[m - 1]);
        }
        plus.push_back(std::move(cap));
        minus.push_back(std::move(rev));
        return true;
    }

    // Volume of a convex polyhedron (cells): pyramids from the mean of its points.
    static double volume(const Table &T, const Polyhedron &P) {
        double m[3] = {0.0, 0.0, 0.0};
        std::size_t count = 0;
        for (const Facet &f : P)
            for (const int k : f.g.v) {
                const double *u = T.u(k);
                for (int c = 0; c < 3; ++c) m[c] += u[c];
                ++count;
            }
        for (double &c : m) c /= static_cast<double>(std::max<std::size_t>(count, 1));
        double v = 0.0;
        for (const Facet &f : P) {
            const Ints &g = f.g.v;
            if (g.size() < 3) continue;
            const double *o = T.u(g[0]);
            double s[3] = {0.0, 0.0, 0.0};
            for (std::size_t i = 1; i + 1 < g.size(); ++i) {
                const double *p = T.u(g[i]), *q = T.u(g[i + 1]);
                const double e1[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
                const double e2[3] = {q[0] - o[0], q[1] - o[1], q[2] - o[2]};
                s[0] += e1[1] * e2[2] - e1[2] * e2[1];
                s[1] += e1[2] * e2[0] - e1[0] * e2[2];
                s[2] += e1[0] * e2[1] - e1[1] * e2[0];
            }
            v += std::abs(s[0] * (o[0] - m[0]) + s[1] * (o[1] - m[1]) + s[2] * (o[2] - m[2])) / 6.0;
        }
        return v;
    }

    // What a polygon on a plane is, covered by the faces of that plane (in key order) as a
    // union: whether some part of it of positive area is bare (exact), the bare part's area,
    // the covered area per material on the plane's + and - sides (PLAN 2.3: the first face
    // covering a part whose normal points to that side, else the first covering it), the
    // covered area, and the sum over the faces of the area of each face's part in it.
    struct Covering {
        bool bare = false;
        double open = 0.0, covered = 0.0, faces = 0.0;
        Walls plus, minus;
    };
    // covering, when given, gets the faces of (ii)'s candidate pairs that cover a part of
    // positive area of g (a part with a point strictly inside each coverage plane).
    Covering cover(Table &T, const Poly &g, const std::vector<const FaceX *> &faces,
                   std::vector<const FaceX *> *covering = nullptr) const {
        Covering out;
        struct Part {
            Poly g;
            int first = -1, plus = -1, minus = -1;
        };
        std::vector<Part> parts{{g, -1, -1, -1}}, next;
        std::vector<int> s;
        std::vector<double> f, b;
        for (std::size_t t = 0; t < faces.size(); ++t) {
            const FaceX &F = *faces[t];
            next.clear();
            for (Part &q : parts) {
                Poly inside = std::move(q.g);
                bool gone = false;
                for (int k = 0; k < 3 && !gone; ++k) {
                    const int p = F.e[k], keep = F.in[k];
                    const std::size_t m = inside.v.size();
                    s.resize(m);
                    f.resize(m);
                    b.resize(m);
                    bool in = false, outp = false;
                    for (std::size_t i = 0; i < m; ++i) {
                        s[i] = T.side(inside.v[i], p, f[i], b[i]);
                        in = in || s[i] * keep > 0;
                        outp = outp || s[i] * keep < 0;
                    }
                    if (!outp) continue;
                    if (!in) {
                        next.push_back({std::move(inside), q.first, q.plus, q.minus});
                        gone = true;
                        break;
                    }
                    Poly lo, hi;
                    auto cross = [&](std::size_t i) {
                        const std::size_t j = (i + 1) % m;
                        return T.cross(inside.v[i], inside.v[j], s[i], f[i], b[i], f[j], b[j], inside.plane, inside.e[i], p);
                    };
                    split_poly(inside, s.data(), p, cross, &lo, &hi);
                    if (keep > 0) {
                        next.push_back({std::move(lo), q.first, q.plus, q.minus});
                        inside = std::move(hi);
                    } else {
                        next.push_back({std::move(hi), q.first, q.plus, q.minus});
                        inside = std::move(lo);
                    }
                }
                if (gone) continue;
                if (covering && F.layer && (covering->empty() || covering->back() != &F)) covering->push_back(&F);
                out.faces += T.area(inside.v);
                Part in{std::move(inside), q.first, q.plus, q.minus};
                const int ti = static_cast<int>(t);
                if (in.first < 0) in.first = ti;
                if (in.plus < 0 && F.orient > 0) in.plus = ti;
                if (in.minus < 0 && F.orient < 0) in.minus = ti;
                next.push_back(std::move(in));
            }
            parts.swap(next);
        }
        for (const Part &q : parts) {
            const double a = T.area(q.g.v);
            if (q.first < 0) {
                out.bare = true;
                out.open += a;
                continue;
            }
            out.covered += a;
            add(out.plus, faces[static_cast<std::size_t>(q.plus >= 0 ? q.plus : q.first)]->material, a);
            add(out.minus, faces[static_cast<std::size_t>(q.minus >= 0 ? q.minus : q.first)]->material, a);
        }
        return out;
    }

    // The cube's eight corners and six facets in T (points counter-clockwise from outside).
    Polyhedron cube_poly(Table &T, std::size_t i) const {
        const auto p = at(i);
        int gp[3][2];
        std::int64_t gv[3][2];
        for (int a = 0; a < 3; ++a)
            for (int s = 0; s < 2; ++s) {
                gp[a][s] = grid_plane_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)] + static_cast<std::size_t>(s)];
                gv[a][s] = small(G_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)] + static_cast<std::size_t>(s)]);
            }
        int corner[2][2][2];
        for (int x = 0; x < 2; ++x)
            for (int y = 0; y < 2; ++y)
                for (int z = 0; z < 2; ++z) {
                    const std::int64_t q[3] = {gv[0][x], gv[1][y], gv[2][z]};
                    corner[x][y][z] = T.lattice(q, gp[0][x], gp[1][y], gp[2][z]);
                }
        Polyhedron P;
        for (int a = 0; a < 3; ++a) {
            const int b = (a + 1) % 3, c = (a + 2) % 3;
            for (int s = 0; s < 2; ++s) {
                // (b, c) corners: counter-clockwise about +e_a on the + side, the reverse on the -
                static constexpr int up[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
                static constexpr int down[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
                const auto &order = s ? up : down;
                Facet f;
                f.g.plane = gp[a][s];
                for (int k = 0; k < 4; ++k) {
                    int idx[3];
                    idx[a] = s;
                    idx[b] = order[k][0];
                    idx[c] = order[k][1];
                    f.g.v.push_back(corner[idx[0]][idx[1]][idx[2]]);
                    // the edge to the next corner: the coordinate that stays
                    const int nb = order[(k + 1) % 4][0], nc = order[(k + 1) % 4][1];
                    f.g.e.push_back(nb == order[k][0] ? gp[b][nb] : gp[c][nc]);
                }
                P.push_back(std::move(f));
            }
        }
        return P;
    }

    Cube cut_cube(std::size_t cell, const std::vector<std::pair<std::size_t, std::size_t>> &pairs, std::size_t from,
                  std::size_t to, const std::vector<int> &zones) const {
        Cube C;
        C.cell = cell;
        const std::size_t i = C.cell;
        std::int64_t corner0[3];
        corner(i, corner0);
        Table T(*this, corner0);
        Polyhedron cube = cube_poly(T, i);
        const auto p = at(i);
        // The faces' parts in the closed cube, of positive area (exact).
        struct Part {
            const FaceX *F;
            Poly g;
        };
        std::vector<Part> parts;
        auto make = [](Table &Tt, const Poly &g, std::size_t a, std::size_t b, int sa, double fa, double ba, double fb,
                       double bb, int pl) { return poly_cross(Tt, g, a, b, sa, fa, ba, fb, bb, pl); };
        for (std::size_t k = from; k < to; ++k) {
            const FaceX &F = faces_[pairs[k].second];
            Poly g = face_poly(T, F);
            bool keep = true;
            for (int a = 0; a < 3 && keep; ++a) {
                keep = clip(T, g, grid_plane_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)]], 1, make) &&
                       clip(T, g, grid_plane_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)] + 1], -1, make);
            }
            if (!keep) continue;
            C.face_area += T.area(g.v);
            parts.push_back({&F, std::move(g)});
        }
        if (parts.empty() && zones.empty()) return C;
        C.tree = true;
        std::stable_sort(parts.begin(), parts.end(), [](const Part &x, const Part &y) {
            return x.F->plane != y.F->plane ? x.F->plane < y.F->plane : x.F->rank < y.F->rank;
        });
        // The tree: each plane in turn splits the leaves its faces' parts have area in.
        std::vector<Polyhedron> leaves{std::move(cube)};
        std::vector<std::pair<int, int>> parent{{-1, 0}};
        SignCache SC;
        std::vector<std::pair<int, Poly>> found;
        std::vector<std::pair<std::uint64_t, int>> made;
        Polyhedron minus, plus;
        std::vector<std::pair<int, std::vector<const FaceX *>>> groups;  // plane, its faces
        for (std::size_t a = 0; a < parts.size();) {
            const int pl = parts[a].F->plane;
            std::vector<const FaceX *> fs;
            std::vector<char> hit(leaves.size(), 0);
            for (; a < parts.size() && parts[a].F->plane == pl; ++a) {
                fs.push_back(parts[a].F);
                found.clear();
                filter(T, C.nodes, root(C.nodes), parts[a].g, found);
                for (const auto &q : found) hit[static_cast<std::size_t>(q.first)] = 1;
            }
            groups.push_back({pl, std::move(fs)});
            made.clear();
            const std::size_t now = leaves.size();
            for (std::size_t l = 0; l < now; ++l) {
                if (!hit[l]) continue;
                const int node = static_cast<int>(C.nodes.size());
                if (!split_leaf(T, SC, leaves[l], pl, node, minus, plus, made))
                    throw std::logic_error("fitted walls: a face's part in a leaf its plane does not split");
                const int fresh = static_cast<int>(leaves.size());
                C.nodes.push_back({pl, {~static_cast<int>(l), ~fresh}});
                const auto [up, side] = parent[l];
                if (up >= 0) C.nodes[static_cast<std::size_t>(up)].child[static_cast<std::size_t>(side)] = node;
                parent[l] = {node, 0};
                parent.push_back({node, 1});
                leaves[l] = std::move(minus);
                leaves.push_back(std::move(plus));
                minus = Polyhedron();
                plus = Polyhedron();
            }
        }
        C.planes = groups.size();
        // The zones of refusal (i): each zone in turn clips the leaves not yet in a zone by
        // its facet planes (a leaf crossed by one is split; its part outside is left), and
        // the leaves left after the last facet lie inside the zone.
        std::vector<int> leaf_zone;
        if (!zones.empty()) {
            leaf_zone.assign(leaves.size(), -1);
            std::vector<std::size_t> cand, keep;
            for (const int z : zones) {
                cand.clear();
                for (std::size_t l = 0; l < leaves.size(); ++l)
                    if (leaf_zone[l] < 0) cand.push_back(l);
                for (const auto &[pl, inner] : zones_[static_cast<std::size_t>(z)].facets) {
                    keep.clear();
                    made.clear();
                    for (const std::size_t l : cand) {
                        const int node = static_cast<int>(C.nodes.size());
                        if (split_leaf(T, SC, leaves[l], pl, node, minus, plus, made)) {
                            const int fresh = static_cast<int>(leaves.size());
                            C.nodes.push_back({pl, {~static_cast<int>(l), ~fresh}});
                            const auto [up, side] = parent[l];
                            if (up >= 0) C.nodes[static_cast<std::size_t>(up)].child[static_cast<std::size_t>(side)] = node;
                            parent[l] = {node, 0};
                            parent.push_back({node, 1});
                            leaves[l] = std::move(minus);
                            leaves.push_back(std::move(plus));
                            leaf_zone.push_back(-1);
                            minus = Polyhedron();
                            plus = Polyhedron();
                            keep.push_back(inner < 0 ? l : static_cast<std::size_t>(fresh));
                        } else {
                            // on one side: the sign of any point not on the plane
                            int s = 0;
                            for (const Facet &f : leaves[l]) {
                                for (const int k : f.g.v) {
                                    s = SC.get(T, k, pl);
                                    if (s != 0) break;
                                }
                                if (s != 0) break;
                            }
                            if (s == 0) throw std::logic_error("fitted walls: a leaf on a zone's plane");
                            if (s == inner) keep.push_back(l);
                        }
                    }
                    cand.swap(keep);
                    if (cand.empty()) break;
                }
                for (const std::size_t l : cand) leaf_zone[l] = z;
            }
        }
        const std::size_t m = leaves.size();
        C.leaves = m;
        // Facets: each cap from its + side, cut into the leaves across it, and covered.
        std::vector<std::size_t> link(m);
        for (std::size_t l = 0; l < m; ++l) link[l] = l;
        auto find = [&](std::size_t q) {
            while (link[q] != q) q = link[q] = link[link[q]];
            return q;
        };
        std::vector<Walls> leaf_wall(m);
        // Every wall cover() gives, as (leaf, material, area), for K1.
        std::vector<std::tuple<std::size_t, std::int32_t, double>> given;
        static const std::vector<const FaceX *> no_faces;
        auto faces_of = [&](int pl) -> const std::vector<const FaceX *> & {
            for (const auto &g : groups)
                if (g.first == pl) return g.second;
            if (zones.empty()) throw std::logic_error("fitted walls: a node's plane without faces");
            return no_faces;  // a zone's plane that holds no face of this cube: bare
        };
        // Without the zones: links only between leaves outside every zone.
        std::vector<std::size_t> link_z;
        if (!zones.empty()) {
            link_z.resize(m);
            for (std::size_t l = 0; l < m; ++l) link_z[l] = l;
        }
        auto find_z = [&](std::size_t q) {
            while (link_z[q] != q) q = link_z[q] = link_z[link_z[q]];
            return q;
        };
        std::vector<std::pair<std::size_t, int>> leaf_adj;  // (leaf outside the zones, zone across a bare facet)
        bool any_layer = false;
        for (const Part &q : parts) any_layer = any_layer || q.F->layer;
        std::vector<const FaceX *> cov;
        std::vector<std::array<std::size_t, 3>> layer_leaves;  // (face, + leaf, - leaf)
        for (std::size_t a = 0; a < m; ++a) {
            for (const Facet &f : leaves[a]) {
                if (f.node < 0 || !f.plus) continue;
                found.clear();
                filter(T, C.nodes, C.nodes[static_cast<std::size_t>(f.node)].child[0], f.g, found);
                C.parted += T.area(f.g.v);
                for (const auto &[b, g] : found) {
                    cov.clear();
                    const Covering w = cover(T, g, faces_of(f.g.plane), any_layer ? &cov : nullptr);
                    C.parted_plus += w.open + total(w.plus);
                    C.parted_minus += w.open + total(w.minus);
                    for (const FaceX *F : cov)
                        layer_leaves.push_back({static_cast<std::size_t>(F - faces_.data()), a, static_cast<std::size_t>(b)});
                    C.covered += w.faces;
                    C.overlap += w.faces - w.covered;
                    if (w.bare && !zones.empty()) {
                        const int za = leaf_zone[a], zb = leaf_zone[static_cast<std::size_t>(b)];
                        if (za < 0 && zb < 0) link_z[find_z(a)] = find_z(static_cast<std::size_t>(b));
                        else if (za < 0) leaf_adj.push_back({a, zb});
                        else if (zb < 0) leaf_adj.push_back({static_cast<std::size_t>(b), za});
                    }
                    if (w.bare) link[find(a)] = find(static_cast<std::size_t>(b));
                    for (const auto &e : w.plus) add(leaf_wall[a], e.first, e.second);
                    for (const auto &e : w.minus) add(leaf_wall[static_cast<std::size_t>(b)], e.first, e.second);
                    for (const auto &e : w.plus) given.push_back({a, e.first, e.second});
                    for (const auto &e : w.minus) given.push_back({static_cast<std::size_t>(b), e.first, e.second});
                }
            }
        }
        // Regions: the joined leaves, numbered in the order of their first leaf.
        C.leaf_region.assign(m, -1);
        std::vector<int> region_of_root(m, -1);
        for (std::size_t l = 0; l < m; ++l) {
            const std::size_t r = find(l);
            if (region_of_root[r] < 0) {
                region_of_root[r] = static_cast<int>(C.volume.size());
                C.volume.push_back(0.0);
                C.wall.emplace_back();
            }
            const int g = region_of_root[r];
            C.leaf_region[l] = g;
            C.volume[static_cast<std::size_t>(g)] += volume(T, leaves[l]);
            for (const auto &e : leaf_wall[l]) add(C.wall[static_cast<std::size_t>(g)], e.first, e.second);
        }
        // K1: each region's walls per material, as stored, against cover()'s walls gathered by
        // the regions of their leaves. K2: the regions' volumes against the cube's.
        {
            std::vector<Walls> formed(C.volume.size());
            for (const auto &[l, mat, area] : given) add(formed[static_cast<std::size_t>(C.leaf_region[l])], mat, area);
            auto of = [](const Walls &w, std::int32_t mat) {
                for (const auto &e : w)
                    if (e.first == mat) return e.second;
                return 0.0;
            };
            for (std::size_t r = 0; r < formed.size(); ++r) {
                auto place = [&]() { return cell_text(cell) + ", region " + std::to_string(r); };
                for (const auto &e : C.wall[r]) {
                    const double f = of(formed[r], e.first);
                    C.k_wall.compare(e.second, f, (std::max)(e.second, f), [&]() { return place() + ", material " + std::to_string(e.first); });
                }
                for (const auto &e : formed[r])
                    if (of(C.wall[r], e.first) == 0.0)
                        C.k_wall.compare(0.0, e.second, e.second, [&]() { return place() + ", material " + std::to_string(e.first); });
            }
            double total = 0.0, scale = 0.0;
            for (const double v : C.volume) {
                total += v;
                scale += std::abs(v);
            }
            C.k_volume.compare(total, cell_volume(cell), scale, [&]() { return cell_text(cell); });
        }
        // The region holding the centre (perturbed: a zero is the + side).
        std::int64_t c[3];
        for (int a = 0; a < 3; ++a) c[a] = small(Cq_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)]]);
        C.centre = C.leaf_region[static_cast<std::size_t>(locate(T, C.nodes, c))];
        for (const auto &[t, a, b] : layer_leaves)
            C.layer_sides.push_back({static_cast<std::int64_t>(t), C.leaf_region[a], C.leaf_region[b]});
        std::sort(C.layer_sides.begin(), C.layer_sides.end());
        C.layer_sides.erase(std::unique(C.layer_sides.begin(), C.layer_sides.end()), C.layer_sides.end());
        if (!zones.empty()) {
            C.leaf_zone = std::move(leaf_zone);
            C.leaf_region_z.assign(m, -1);
            std::vector<int> of_root(m, -1);
            for (std::size_t l = 0; l < m; ++l) {
                if (C.leaf_zone[l] >= 0) continue;
                const std::size_t r = find_z(l);
                if (of_root[r] < 0) of_root[r] = C.regions_z++;
                C.leaf_region_z[l] = of_root[r];
            }
            for (const auto &[l, z] : leaf_adj) C.zone_adj.push_back({C.leaf_region_z[l], z});
            std::sort(C.zone_adj.begin(), C.zone_adj.end());
            C.zone_adj.erase(std::unique(C.zone_adj.begin(), C.zone_adj.end()), C.zone_adj.end());
        }
        C.exact_calls = T.exact_calls;
        C.refined = T.refined;
        return C;
    }

    // The leaf of a tree holding lattice point c moved by (eps, eps^2, eps^3): at an exact
    // zero against a node's plane (whose normal's first nonzero component is positive), the
    // + side.
    static int locate(Table &T, const std::vector<Node> &nodes, const std::int64_t c[3]) {
        const int k = T.lattice(c, -1, -1, -1);
        int at = root(nodes);
        while (at >= 0) {
            const Node &N = nodes[static_cast<std::size_t>(at)];
            at = N.child[T.side(k, N.plane) >= 0 ? 1 : 0];
        }
        return ~at;
    }

    // ---- membranes ---------------------------------------------------------------------------

    // Faces lying exactly on a grid plane between two cells: each square of that plane its
    // part has positive area in (exact), as (square's - cell, axis, face). The cells on
    // either side that are not on the grid's edge become nodes.
    void membranes(std::size_t threads) {
        const std::size_t workers = std::max<std::size_t>(1, worker_threads(threads));
        std::vector<std::vector<std::array<std::size_t, 3>>> found(workers);
        std::vector<std::vector<double>> area(workers);
        fit_for(faces_.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t w) {
            for (std::size_t t = begin; t < end; ++t) {
                const FaceX &F = faces_[t];
                if (F.mem_axis < 0) continue;
                const int a = F.mem_axis, b = (a + 1) % 3, c = (a + 2) % 3;
                const std::size_t i = F.mem_index;
                if (i == 0 || i >= n_[static_cast<std::size_t>(a)]) continue;
                std::size_t r0[3], r1[3];
                bool any = true;
                for (const int q : {b, c}) {
                    const double g0 = (F.lo[q] - origin_[q]) / cell_, g1 = (F.hi[q] - origin_[q]) / cell_;
                    const double lo = std::max(0.0, std::floor(g0) - 1.0);
                    const double hi = std::min(static_cast<double>(n_[static_cast<std::size_t>(q)]) - 1.0, std::floor(g1) + 1.0);
                    any = any && lo <= hi;
                    r0[q] = static_cast<std::size_t>(std::max(0.0, lo));
                    r1[q] = static_cast<std::size_t>(std::max(0.0, hi));
                }
                if (!any) continue;
                auto make = [](Table &Tt, const Poly &g, std::size_t x, std::size_t y, int sa, double fa, double ba, double fb,
                               double bb, int pl) { return poly_cross(Tt, g, x, y, sa, fa, ba, fb, bb, pl); };
                for (std::size_t j = r0[b]; j <= r1[b]; ++j)
                    for (std::size_t k = r0[c]; k <= r1[c]; ++k) {
                        std::size_t idx[3];
                        idx[a] = i - 1;
                        idx[b] = j;
                        idx[c] = k;
                        const std::size_t low = index(idx[0], idx[1], idx[2]);
                        idx[a] = i;
                        const std::size_t high = index(idx[0], idx[1], idx[2]);
                        if (edge(low) && edge(high)) continue;
                        std::int64_t C0[3];
                        corner(low, C0);
                        Table T(*this, C0);
                        Poly g = face_poly(T, F);
                        const bool keep =
                            clip(T, g, grid_plane_[static_cast<std::size_t>(b)][j], 1, make) &&
                            clip(T, g, grid_plane_[static_cast<std::size_t>(b)][j + 1], -1, make) &&
                            clip(T, g, grid_plane_[static_cast<std::size_t>(c)][k], 1, make) &&
                            clip(T, g, grid_plane_[static_cast<std::size_t>(c)][k + 1], -1, make);
                        if (!keep) continue;
                        found[w].push_back({low, static_cast<std::size_t>(a), t});
                        area[w].push_back(T.area(g.v));
                    }
            }
        });
        std::vector<std::pair<std::array<std::size_t, 3>, double>> all;
        for (std::size_t w = 0; w < workers; ++w)
            for (std::size_t k = 0; k < found[w].size(); ++k) all.push_back({found[w][k], area[w][k]});
        std::sort(all.begin(), all.end(), [&](const auto &x, const auto &y) {
            if (x.first[0] != y.first[0] || x.first[1] != y.first[1]) return x.first < y.first;
            return faces_[x.first[2]].rank < faces_[y.first[2]].rank;
        });
        for (const auto &[key, a] : all) {
            membrane_.push_back(key);
            membrane_face_area_ += a;
        }
    }
    void cut_cubes(const std::vector<std::pair<std::size_t, std::size_t>> &pairs,
                   const std::vector<std::pair<std::size_t, int>> &zone_pairs, std::size_t threads) {
        // The cubes: those with candidate faces, and those a zone may meet.
        struct Item {
            std::size_t cell, from, to;
            std::vector<int> zones;
        };
        std::vector<Item> items;
        for (std::size_t k = 0, z = 0; k < pairs.size() || z < zone_pairs.size();) {
            const std::size_t cell = std::min(k < pairs.size() ? pairs[k].first : ~std::size_t{0},
                                              z < zone_pairs.size() ? zone_pairs[z].first : ~std::size_t{0});
            Item it{cell, k, k, {}};
            while (k < pairs.size() && pairs[k].first == cell) ++k;
            it.to = k;
            while (z < zone_pairs.size() && zone_pairs[z].first == cell) it.zones.push_back(zone_pairs[z++].second);
            items.push_back(std::move(it));
        }
        const std::size_t candidates = items.size();
        std::vector<Cube> cubes(candidates);
        in_turn(candidates, threads, [&](std::size_t c) {
            cubes[c] = cut_cube(items[c].cell, pairs, items[c].from, items[c].to, items[c].zones);
        });
        stats_.zone_cubes = 0;
        for (const Item &it : items) stats_.zone_cubes += it.zones.empty() ? 0 : 1;
        // Cells next to a membrane are nodes too (one region, unless they hold faces).
        std::vector<std::size_t> mem_cells;
        for (const auto &m : membrane_) {
            const std::size_t low = m[0], high = neighbour(low, 2 * static_cast<int>(m[1]) + 1);
            for (const std::size_t c : {low, high})
                if (!edge(c)) mem_cells.push_back(c);
        }
        std::sort(mem_cells.begin(), mem_cells.end());
        mem_cells.erase(std::unique(mem_cells.begin(), mem_cells.end()), mem_cells.end());
        std::size_t k = 0;
        SelfCheck k_wall, k_volume;
        for (const Cube &C : cubes) {
            k_wall.merge(C.k_wall);
            k_volume.merge(C.k_volume);
        }
        finish(k_wall, "check_wall", stats_.check_wall, stats_.check_wall_failed);
        finish(k_volume, "check_volume", stats_.check_volume, stats_.check_volume_failed);
        for (Cube &C : cubes) {
            while (k < mem_cells.size() && mem_cells[k] < C.cell) push_bare(mem_cells[k++]);
            if (k < mem_cells.size() && mem_cells[k] == C.cell) ++k;
            else if (!C.tree) continue;
            if (!C.tree) {
                push_bare(C.cell);
                continue;
            }
            stats_.cubes += 1;
            stats_.largest_planes = std::max(stats_.largest_planes, C.planes);
            stats_.leaves += C.leaves;
            stats_.regions += C.volume.size();
            stats_.face_area += C.face_area;
            stats_.covered_area += C.covered;
            stats_.overlap_area += C.overlap;
            stats_.partition_area += C.parted;
            stats_.partition_plus += C.parted_plus;
            stats_.partition_minus += C.parted_minus;
            cubes_.push_back(std::move(C));
        }
        while (k < mem_cells.size()) push_bare(mem_cells[k++]);
    }
    void push_bare(std::size_t cell) {
        Cube C;
        C.cell = cell;
        C.leaf_region = {0};
        C.volume = {cell_volume(cell)};
        C.wall.emplace_back();
        cubes_.push_back(std::move(C));
    }

    // ---- squares between cubes ---------------------------------------------------------------

    struct Piece {
        std::int64_t low, high;  // regions on the square's - and + sides (kEdge: the grid's edge)
        std::int64_t low_z = kNone, high_z = kNone;  // the same without the zones (kZone: in one)
        bool is_open = false;    // some part of it of positive area is bare (exact)
        double open = 0.0;       // the bare part's area (a measure)
        Walls wall_low, wall_high;
    };
    struct Square {
        std::size_t low;
        int axis;
        std::vector<Piece> pieces;
        double faces = 0.0, covered = 0.0;
        // The square's area, and its pieces' bare area plus the walls given to its - and + sides.
        double whole = 0.0, parted_low = 0.0, parted_high = 0.0;
        std::vector<std::pair<std::int64_t, int>> zone_adj;  // (region without the zones, zone) across an open piece
        std::vector<std::array<std::int64_t, 3>> layer_sides;  // (ii): (face, region on + side, region on - side)
    };
    struct Region {
        std::size_t cell = 0;
        double volume = 0.0;
        std::int64_t owner = kNone;
        bool in_air = false;  // in the source's component
        Walls wall;
    };

    std::int64_t region_of(std::size_t cell, const Cube *C, int leaf) const {
        if (C) {
            return cube_first_[static_cast<std::size_t>(C - cubes_.data())] + C->leaf_region[static_cast<std::size_t>(leaf)];
        }
        if (edge(cell)) return kEdge;
        const auto it = std::lower_bound(whole_.begin(), whole_.end(), cell);
        return whole_first_ + static_cast<std::int64_t>(it - whole_.begin());
    }
    // The same without the zones (refusal (i)): kZone for a leaf inside a zone, and
    // zone the zone's index there.
    std::int64_t region_z(std::size_t cell, const Cube *C, int leaf, int &zone) const {
        zone = -1;
        if (C) {
            const std::size_t k = static_cast<std::size_t>(C - cubes_.data());
            if (C->leaf_region_z.empty()) return cube_first_z_[k] + C->leaf_region[static_cast<std::size_t>(leaf)];
            const int r = C->leaf_region_z[static_cast<std::size_t>(leaf)];
            if (r < 0) {
                zone = C->leaf_zone[static_cast<std::size_t>(leaf)];
                return kZone;
            }
            return cube_first_z_[k] + r;
        }
        if (edge(cell)) return kEdge;
        const auto it = std::lower_bound(whole_.begin(), whole_.end(), cell);
        return whole_first_z_ + static_cast<std::int64_t>(it - whole_.begin());
    }
    static int regions_z_of(const Cube &C) {
        return C.leaf_region_z.empty() ? static_cast<int>(C.volume.size()) : C.regions_z;
    }
    const Cube *cube_at(std::size_t cell) const {
        const auto it = std::lower_bound(cube_cells_.begin(), cube_cells_.end(), cell);
        if (it == cube_cells_.end() || *it != cell) return nullptr;
        return &cubes_[static_cast<std::size_t>(it - cube_cells_.begin())];
    }

    void cut_squares(std::size_t threads) {
        std::sort(cubes_.begin(), cubes_.end(), [](const Cube &a, const Cube &b) { return a.cell < b.cell; });
        for (const Cube &C : cubes_) cube_cells_.push_back(C.cell);
        std::int64_t next = 0;
        for (const Cube &C : cubes_) {
            cube_first_.push_back(next);
            next += static_cast<std::int64_t>(C.volume.size());
        }
        whole_first_ = next;
        next = 0;
        for (const Cube &C : cubes_) {
            cube_first_z_.push_back(next);
            next += regions_z_of(C);
        }
        whole_first_z_ = next;
        std::vector<std::pair<std::size_t, int>> keys;
        for (const Cube &C : cubes_) {
            for (int k = 0; k < 6; ++k) {
                const std::size_t j = neighbour(C.cell, k);
                keys.push_back(k % 2 ? std::make_pair(C.cell, k / 2) : std::make_pair(j, k / 2));
                if (!edge(j) && !cube_at(j)) whole_.push_back(j);
            }
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        std::sort(whole_.begin(), whole_.end());
        whole_.erase(std::unique(whole_.begin(), whole_.end()), whole_.end());
        for (const Cube &C : cubes_) {
            for (std::size_t r = 0; r < C.volume.size(); ++r) {
                Region R;
                R.cell = C.cell;
                R.volume = C.volume[r];
                R.wall = C.wall[r];
                regions_.push_back(std::move(R));
            }
        }
        for (const std::size_t j : whole_) {
            Region R;
            R.cell = j;
            R.volume = cell_volume(j);
            regions_.push_back(std::move(R));
        }
        // K3: the regions as copied, against the cubes' regions and the whole cells.
        {
            SelfCheck k3;
            std::size_t g = 0;
            for (const Cube &C : cubes_) {
                for (std::size_t r = 0; r < C.volume.size(); ++r, ++g) {
                    const Region &R = regions_[g];
                    auto place = [&]() { return "region " + std::to_string(g) + " (" + cell_text(C.cell) + ", region " + std::to_string(r) + ")"; };
                    k3.compare(static_cast<double>(R.cell), static_cast<double>(C.cell), -1.0, place);
                    k3.compare(R.volume, C.volume[r], -1.0, place);
                    k3.compare(static_cast<double>(R.wall.size()), static_cast<double>(C.wall[r].size()), -1.0, [&]() { return place() + ", its count of walls"; });
                    for (std::size_t i = 0; i < R.wall.size() && i < C.wall[r].size(); ++i) {
                        k3.compare(static_cast<double>(R.wall[i].first), static_cast<double>(C.wall[r][i].first), -1.0, [&]() { return place() + ", a wall's material"; });
                        k3.compare(R.wall[i].second, C.wall[r][i].second, -1.0, [&]() { return place() + ", a wall's area"; });
                    }
                }
            }
            for (const std::size_t j : whole_) {
                const Region &R = regions_[g];
                auto place = [&]() { return "region " + std::to_string(g) + " (the whole " + cell_text(j) + ")"; };
                k3.compare(static_cast<double>(R.cell), static_cast<double>(j), -1.0, place);
                k3.compare(R.volume, cell_volume(j), -1.0, place);
                k3.compare(static_cast<double>(R.wall.size()), 0.0, -1.0, place);
                ++g;
            }
            k3.compare(static_cast<double>(g), static_cast<double>(regions_.size()), -1.0, []() { return std::string("the count of regions"); });
            finish(k3, "check_copy", stats_.check_copy, stats_.check_copy_failed);
        }
        squares_.resize(keys.size());
        in_turn(keys.size(), threads, [&](std::size_t s) { squares_[s] = cut_square(keys[s].first, keys[s].second); });
        stats_.squares = squares_.size();
        // K7: each piece's regions are regions of the square's two cells, or the grid's edge
        // exactly where that cell is on it.
        {
            SelfCheck k7;
            for (const Square &S : squares_) {
                const std::size_t high = neighbour(S.low, 2 * S.axis + 1);
                for (std::size_t q = 0; q < S.pieces.size(); ++q) {
                    const Piece &P = S.pieces[q];
                    for (int side = 0; side < 2; ++side) {
                        const std::size_t c = side ? high : S.low;
                        const std::int64_t g = side ? P.high : P.low;
                        const bool ok = edge(c) ? g == kEdge
                                                : g >= 0 && static_cast<std::size_t>(g) < regions_.size() && regions_[static_cast<std::size_t>(g)].cell == c;
                        k7.compare(ok ? 0.0 : 1.0, 0.0, -1.0, [&]() {
                            return "the square between " + cell_text(S.low) + " and " + cell_text(high) + ", piece " + std::to_string(q) +
                                   (side ? ", high side" : ", low side") + " (region " + std::to_string(g) + ")";
                        });
                    }
                }
            }
            finish(k7, "check_piece_sides", stats_.check_piece_sides, stats_.check_piece_sides_failed);
        }
        // covered_area counts each face's covered part, as the cubes count it (w.faces), so
        // that faces lying on one another are counted once each on both sides of the balance;
        // their overlap is reported apart, as overlap_area.
        for (const Square &S : squares_) {
            stats_.covered_area += S.faces;
            stats_.overlap_area += S.faces - S.covered;
            stats_.partition_area += S.whole;
            stats_.partition_plus += S.parted_high;
            stats_.partition_minus += S.parted_low;
        }
        stats_.face_area += membrane_face_area_;
        // Two measures, reported, and a fit whose measure passes 1e-9 throws std::logic_error. The
        // wall balance: each face's parts in the cubes and squares are tiled by the pieces that face covers. The partition: each facet between
        // leaves and each square is tiled by its pieces, each piece's area given to each side
        // once, as bare area or as wall. A piece, a covered part or a bare part the fit loses
        // breaks the partition; the balance alone does not see a covered part lost after the
        // per-face sums are formed, nor a bare part.
        stats_.balance = std::abs(stats_.face_area - stats_.covered_area) / std::max(1.0, stats_.face_area);
        stats_.partition = std::max(std::abs(stats_.partition_area - stats_.partition_plus),
                                    std::abs(stats_.partition_area - stats_.partition_minus)) /
                           std::max(1.0, stats_.partition_area);
        if (stats_.balance > 1e-9)
            throw std::logic_error("fitted walls: self-check balance failed at the whole model: " + exact_text(stats_.face_area) +
                                   " cell faces of faces against " + exact_text(stats_.covered_area) + " covered (relative " +
                                   exact_text(stats_.balance) + ", threshold 1e-9)");
        if (stats_.partition > 1e-9)
            throw std::logic_error("fitted walls: self-check partition failed at the whole model: " +
                                   exact_text(stats_.partition_area) + " cell faces of facets and squares against " +
                                   exact_text(stats_.partition_plus) + " and " + exact_text(stats_.partition_minus) +
                                   " given to their + and - sides (relative " + exact_text(stats_.partition) +
                                   ", threshold 1e-9)");
    }

    Square cut_square(std::size_t low, int axis) const {
        Square S;
        S.low = low;
        S.axis = axis;
        const int a = axis, b = (a + 1) % 3, c = (a + 2) % 3;
        const std::size_t high = neighbour(low, 2 * a + 1);
        const Cube *L = cube_at(low), *H = cube_at(high);
        std::int64_t C0[3];
        corner(low, C0);
        Table T(*this, C0);
        const auto p = at(low);
        // The square: on the grid plane between the two cells, counter-clockwise about +e_a.
        Poly sq;
        sq.plane = grid_plane_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)] + 1];
        static constexpr int up[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int k = 0; k < 4; ++k) {
            std::int64_t q[3];
            q[a] = small(G_[static_cast<std::size_t>(a)][p[static_cast<std::size_t>(a)] + 1]);
            const std::size_t jb = p[static_cast<std::size_t>(b)] + static_cast<std::size_t>(up[k][0]);
            const std::size_t jc = p[static_cast<std::size_t>(c)] + static_cast<std::size_t>(up[k][1]);
            q[b] = small(G_[static_cast<std::size_t>(b)][jb]);
            q[c] = small(G_[static_cast<std::size_t>(c)][jc]);
            sq.v.push_back(T.lattice(q, sq.plane, grid_plane_[static_cast<std::size_t>(b)][jb], grid_plane_[static_cast<std::size_t>(c)][jc]));
            const int nb = up[(k + 1) % 4][0], nc = up[(k + 1) % 4][1];
            sq.e.push_back(nb == up[k][0] ? grid_plane_[static_cast<std::size_t>(b)][p[static_cast<std::size_t>(b)] + static_cast<std::size_t>(nb)]
                                          : grid_plane_[static_cast<std::size_t>(c)][p[static_cast<std::size_t>(c)] + static_cast<std::size_t>(nc)]);
        }
        // Its membranes, in key order.
        std::vector<const FaceX *> mem;
        const std::array<std::size_t, 3> lo_key{low, static_cast<std::size_t>(a), 0};
        for (auto it = std::lower_bound(membrane_.begin(), membrane_.end(), lo_key,
                                        [](const std::array<std::size_t, 3> &x, const std::array<std::size_t, 3> &y) {
                                            return x[0] != y[0] ? x[0] < y[0] : x[1] < y[1];
                                        });
             it != membrane_.end() && (*it)[0] == low && (*it)[1] == static_cast<std::size_t>(a); ++it)
            mem.push_back(&faces_[(*it)[2]]);
        bool any_layer = false;
        for (const FaceX *F : mem) any_layer = any_layer || F->layer;
        std::vector<const FaceX *> cov;
        std::vector<std::pair<int, Poly>> lows, highs;
        S.whole = T.area(sq.v);
        if (L && L->tree) filter(T, L->nodes, root(L->nodes), sq, lows);
        else lows.emplace_back(0, sq);
        for (auto &[ll, g] : lows) {
            highs.clear();
            if (H && H->tree) filter(T, H->nodes, root(H->nodes), g, highs);
            else highs.emplace_back(0, g);
            for (auto &[hl, h] : highs) {
                cov.clear();
                const Covering w = cover(T, h, mem, any_layer ? &cov : nullptr);
                S.faces += w.faces;
                S.covered += w.covered;  // the union: S.faces - S.covered is the coplanar overlap
                Piece P;
                P.low = region_of(low, L, ll);
                P.high = region_of(high, H, hl);
                if (!zones_.empty()) {
                    int zl = -1, zh = -1;
                    P.low_z = region_z(low, L, ll, zl);
                    P.high_z = region_z(high, H, hl, zh);
                    if (w.bare && zl >= 0 && P.high_z >= 0) S.zone_adj.push_back({P.high_z, zl});
                    if (w.bare && zh >= 0 && P.low_z >= 0) S.zone_adj.push_back({P.low_z, zh});
                }
                for (const FaceX *F : cov)
                    S.layer_sides.push_back({static_cast<std::int64_t>(F - faces_.data()), P.high, P.low});
                P.is_open = w.bare;
                P.open = w.open;
                P.wall_low = w.minus;  // the + side of the square's normal e_a is high's
                P.wall_high = w.plus;
                S.parted_low += P.open + total(P.wall_low);
                S.parted_high += P.open + total(P.wall_high);
                S.pieces.push_back(std::move(P));
            }
        }
        return S;
    }

    // ---- 4. the air: components, the source, owners ---------------------------------------------

    struct UF {
        std::vector<std::size_t> up;
        explicit UF(std::size_t n) : up(n) {
            for (std::size_t i = 0; i < n; ++i) up[i] = i;
        }
        std::size_t find(std::size_t q) {
            while (up[q] != q) q = up[q] = up[up[q]];
            return q;
        }
        void join(std::size_t a, std::size_t b) {
            a = find(a);
            b = find(b);
            if (a != b) up[std::max(a, b)] = std::min(a, b);
        }
    };

    // The cell along each axis holding lattice coordinate q moved by +eps: the largest i
    // with G_i <= q; -1 when outside the grid.
    long long cell_of(int a, I128 q) const {
        const std::vector<I128> &G = G_[static_cast<std::size_t>(a)];
        const auto it = std::upper_bound(G.begin(), G.end(), q);
        const long long i = static_cast<long long>(it - G.begin()) - 1;
        if (i < 0 || i >= static_cast<long long>(n_[static_cast<std::size_t>(a)])) return -1;
        return i;
    }

    void air(const Vec3 &source) {
        const std::size_t n = n_[0] * n_[1] * n_[2];
        // Runs of cells that are neither nodes nor on the grid's edge, along x.
        std::vector<char> node(n, 0);
        for (const std::size_t c : cube_cells_) node[c] = 1;
        std::vector<std::size_t> row_start(n_[1] * n_[2] + 1, 0);
        std::vector<std::pair<std::size_t, std::size_t>> runs;  // x0, x1 (inclusive)
        for (std::size_t z = 0; z < n_[2]; ++z)
            for (std::size_t y = 0; y < n_[1]; ++y) {
                const std::size_t row = y + n_[1] * z;
                row_start[row] = runs.size();
                if (y == 0 || z == 0 || y + 1 == n_[1] || z + 1 == n_[2]) continue;
                std::size_t x = 1;
                while (x + 1 < n_[0]) {
                    if (node[index(x, y, z)]) {
                        ++x;
                        continue;
                    }
                    const std::size_t x0 = x;
                    while (x + 1 < n_[0] && !node[index(x, y, z)]) ++x;
                    runs.push_back({x0, x - 1});
                }
            }
        row_start[n_[1] * n_[2]] = runs.size();
        const std::size_t R = regions_.size(), base = R;
        UF uf(R + runs.size());
        auto run_of = [&](std::size_t cell) -> std::size_t {
            const auto p = at(cell);
            const std::size_t row = p[1] + n_[1] * p[2];
            const auto b = runs.begin() + static_cast<std::ptrdiff_t>(row_start[row]);
            const auto e = runs.begin() + static_cast<std::ptrdiff_t>(row_start[row + 1]);
            const auto it = std::upper_bound(b, e, p[0], [](std::size_t x, const auto &r) { return x < r.first; }) - 1;
            return base + static_cast<std::size_t>(it - runs.begin());
        };
        auto join_runs = [&](UF &u, std::size_t at) {
            for (std::size_t z = 1; z + 1 < n_[2]; ++z)
                for (std::size_t y = 1; y + 1 < n_[1]; ++y) {
                    const std::size_t row = y + n_[1] * z;
                    for (const std::size_t other : {row - 1, row - n_[1]}) {
                        std::size_t i = row_start[row], j = row_start[other];
                        const std::size_t ie = row_start[row + 1], je = row_start[other + 1];
                        while (i < ie && j < je) {
                            if (runs[i].second < runs[j].first) ++i;
                            else if (runs[j].second < runs[i].first) ++j;
                            else {
                                u.join(at + i, at + j);
                                if (runs[i].second < runs[j].second) ++i;
                                else ++j;
                            }
                        }
                    }
                }
        };
        join_runs(uf, base);
        for (std::size_t w = 0; w < whole_.size(); ++w) uf.join(static_cast<std::size_t>(whole_first_) + w, run_of(whole_[w]));
        for (const Square &S : squares_)
            for (const Piece &P : S.pieces)
                if (P.is_open && P.low >= 0 && P.high >= 0)
                    uf.join(static_cast<std::size_t>(P.low), static_cast<std::size_t>(P.high));
        // The source's component.
        I128 q[3];
        long long source_cell[3];
        for (int a = 0; a < 3; ++a) {
            q[a] = quantize(source[a], a);
            source_cell[a] = cell_of(a, q[a]);
        }
        if (source_cell[0] < 0 || source_cell[1] < 0 || source_cell[2] < 0 || edge(index(static_cast<std::size_t>(source_cell[0]), static_cast<std::size_t>(source_cell[1]), static_cast<std::size_t>(source_cell[2]))))
            refuse(kRefusedSourceOutside, "fitted walls: the source lies outside the grid's interior (in a cell on its edge or "
                                        "beyond): move it inward or enlarge the grid.");
        const std::size_t sc = index(static_cast<std::size_t>(source_cell[0]), static_cast<std::size_t>(source_cell[1]), static_cast<std::size_t>(source_cell[2]));
        std::size_t omega;
        int source_leaf = 0;
        if (const Cube *C = cube_at(sc)) {
            if (C->tree) {
                std::int64_t C0[3], qs[3];
                corner(sc, C0);
                for (int a = 0; a < 3; ++a) qs[a] = small(q[a]);
                Table T(*this, C0);
                source_leaf = locate(T, C->nodes, qs);
            }
            omega = uf.find(static_cast<std::size_t>(region_of(sc, C, source_leaf)));
        } else {
            omega = uf.find(run_of(sc));
        }
        if (zones_.empty() && !layer_pairs_.empty()) thin_layers(uf, omega);
        // Refusal (i): the components again without the zones (leaves inside a zone are no
        // nodes, links only between nodes); refused when a part of the source's air outside
        // the zones is not in the component of the source without them.
        if (!zones_.empty()) {
            const std::size_t base_z = static_cast<std::size_t>(whole_first_z_) + whole_.size();
            UF ufz(base_z + runs.size());
            join_runs(ufz, base_z);
            auto run_z = [&](std::size_t cell) { return run_of(cell) - base + base_z; };
            for (std::size_t w = 0; w < whole_.size(); ++w) ufz.join(static_cast<std::size_t>(whole_first_z_) + w, run_z(whole_[w]));
            for (const Square &S : squares_)
                for (const Piece &P : S.pieces)
                    if (P.is_open && P.low_z >= 0 && P.high_z >= 0)
                        ufz.join(static_cast<std::size_t>(P.low_z), static_cast<std::size_t>(P.high_z));
            std::size_t omega_z = 0;
            if (const Cube *C = cube_at(sc)) {
                int zone = -1;
                const std::int64_t r = region_z(sc, C, source_leaf, zone);
                if (r == kZone) {
                    refuse(kRefusedSourceInZone, "fitted walls: the source lies within kGap = " + std::to_string(kGap * cell_) +
                                                     " m (in each coordinate) of " + describe_intervals({zone}) + ": move it.");
                }
                omega_z = ufz.find(static_cast<std::size_t>(r));
            } else {
                omega_z = ufz.find(run_z(sc));
            }
            if (!layer_pairs_.empty()) thin_layers(uf, omega);
            // The first cell (in index order) holding such a part.
            std::size_t first = ~std::size_t{0}, comp = 0;
            auto consider = [&](std::size_t cell, std::size_t a, std::size_t z) {
                if (cell >= first || uf.find(a) != omega) return;
                const std::size_t cz = ufz.find(z);
                if (cz == omega_z) return;
                first = cell;
                comp = cz;
            };
            for (std::size_t k = 0; k < cubes_.size(); ++k) {
                const Cube &C = cubes_[k];
                if (C.leaf_region_z.empty()) {
                    for (std::size_t r = 0; r < C.volume.size(); ++r)
                        consider(C.cell, static_cast<std::size_t>(cube_first_[k]) + r, static_cast<std::size_t>(cube_first_z_[k]) + r);
                } else {
                    for (std::size_t l = 0; l < C.leaf_region_z.size(); ++l)
                        if (C.leaf_region_z[l] >= 0)
                            consider(C.cell, static_cast<std::size_t>(cube_first_[k] + C.leaf_region[l]),
                                     static_cast<std::size_t>(cube_first_z_[k] + C.leaf_region_z[l]));
                }
            }
            for (std::size_t w = 0; w < whole_.size(); ++w)
                consider(whole_[w], static_cast<std::size_t>(whole_first_) + w, static_cast<std::size_t>(whole_first_z_) + w);
            for (std::size_t z = 1; z + 1 < n_[2]; ++z)
                for (std::size_t y = 1; y + 1 < n_[1]; ++y) {
                    const std::size_t row = y + n_[1] * z;
                    for (std::size_t r = row_start[row]; r < row_start[row + 1]; ++r)
                        consider(index(runs[r].first, y, z), base + r, base_z + r);
                }
            if (first != ~std::size_t{0}) {
                std::vector<int> named;
                for (std::size_t k = 0; k < cubes_.size(); ++k)
                    for (const auto &[r, zn] : cubes_[k].zone_adj)
                        if (ufz.find(static_cast<std::size_t>(cube_first_z_[k] + r)) == comp) named.push_back(zn);
                for (const Square &S : squares_)
                    for (const auto &[r, zn] : S.zone_adj)
                        if (ufz.find(static_cast<std::size_t>(r)) == comp) named.push_back(zn);
                std::sort(named.begin(), named.end());
                named.erase(std::unique(named.begin(), named.end()), named.end());
                const Vec3 c = centre_of(first);
                refuse(kRefusedSlit,
                       "fitted walls: the air in the cell centred at (" + std::to_string(c.x) + ", " + std::to_string(c.y) +
                           ", " + std::to_string(c.z) + ") m reaches the source only through places within kGap = " +
                           std::to_string(kGap * cell_) + " m (in each coordinate) of " + describe_intervals(named) +
                           ". The opening there is narrower than fitted walls take as given (a crack or slit left open: a "
                           "T-junction, a vertex not welded, faces meeting short of each other; or edges ending close "
                           "together on a surface). Weld the faces along it, close it, or widen it past 2 sqrt 3 kGap = " +
                           std::to_string(2.0 * std::sqrt(3.0) * kGap * cell_) + " m.");
            }
        }
        // Which cells are air: the region holding the centre is in the source's component.
        type_.assign(n, kSolid);
        for (std::size_t z = 1; z + 1 < n_[2]; ++z)
            for (std::size_t y = 1; y + 1 < n_[1]; ++y) {
                const std::size_t row = y + n_[1] * z;
                for (std::size_t r = row_start[row]; r < row_start[row + 1]; ++r) {
                    if (uf.find(base + r) != omega) continue;
                    for (std::size_t x = runs[r].first; x <= runs[r].second; ++x) type_[index(x, y, z)] = kAir;
                }
            }
        for (std::size_t k = 0; k < cubes_.size(); ++k) {
            const Cube &C = cubes_[k];
            if (uf.find(static_cast<std::size_t>(cube_first_[k] + C.centre)) == omega) type_[C.cell] = kAir;
        }
        for (std::size_t r = 0; r < R; ++r) regions_[r].in_air = uf.find(r) == omega;
        std::size_t air = 0;
        double omega_volume = 0.0;
        for (std::size_t i = 0; i < n; ++i) air += type_[i] != kSolid;
        for (std::size_t r = 0; r < static_cast<std::size_t>(whole_first_); ++r)
            if (regions_[r].in_air) omega_volume += regions_[r].volume;
        if (air == 0) {
            refuse(kRefusedPocket, "fitted walls: the source lies in a closed space of the model that holds no cell centre (" +
                                        std::to_string(omega_volume) + " cells of air): move it into the room, or use a finer grid.");
        }
        // The matched layer: its cells in the air; a cube there holding faces with air in it is refused.
        for (std::size_t i = 0; i < n && layer_ > 0; ++i) {
            if (type_[i] == kAir && depth(i) <= layer_) {
                type_[i] = kLayer;
                stats_.layer_cells += 1;
                const auto p = at(i);
                for (std::size_t a = 0; a < 3; ++a) {
                    if (p[a] <= layer_) open_sides_[2 * a] = true;
                    if (p[a] + 1 + layer_ >= n_[a]) open_sides_[2 * a + 1] = true;
                }
            }
        }
        for (std::size_t k = 0; k < cubes_.size() && layer_ > 0; ++k) {
            if (depth(cubes_[k].cell) > layer_) continue;
            for (std::size_t r = 0; r < cubes_[k].volume.size(); ++r)
                if (regions_[static_cast<std::size_t>(cube_first_[k]) + r].in_air) {
                    const Vec3 c = centre_of(cubes_[k].cell);
                    refuse(kRefusedLayer,
                           "fitted walls: the model reaches into the matched layer: faces of it cut the air the source "
                           "reaches in the cell centred at (" + std::to_string(c.x) + ", " + std::to_string(c.y) + ", " +
                               std::to_string(c.z) + ") m, which lies within the layer's " + std::to_string(layer_) +
                               " cells inside the grid's edge, where the layer takes the air as open space. Move the "
                               "model away from the grid's edge, or enlarge the margin round it.");
                }
        }
        // A cube next to a cell of the matched layer that holds faces (a face crossing it, or lying in
        // one of its own faces, which makes it a cube of the fit) and air is refused: the layer's update
        // reads its six neighbours' pressures whole, so each cell next to it must own its whole cube and
        // the whole face towards it.
        for (std::size_t k = 0; k < cubes_.size() && layer_ > 0; ++k) {
            const std::size_t cell = cubes_[k].cell;
            if (depth(cell) != layer_ + 1) continue;
            bool next_to_layer = false;
            for (int d = 0; d < 6; ++d) next_to_layer = next_to_layer || type_[neighbour(cell, d)] == kLayer;
            if (!next_to_layer) continue;
            for (std::size_t r = 0; r < cubes_[k].volume.size(); ++r)
                if (regions_[static_cast<std::size_t>(cube_first_[k]) + r].in_air) {
                    const Vec3 c = centre_of(cell);
                    refuse(kRefusedNextToLayer,
                           "fitted walls: faces of the model lie in the cell next to the matched layer centred at (" +
                               std::to_string(c.x) + ", " + std::to_string(c.y) + ", " + std::to_string(c.z) +
                               ") m, which holds air the source reaches: the layer needs whole faces towards the air. "
                               "Move the model away from the grid's edge, or enlarge the margin round it.");
                }
        }
        stats_.air_cells = air;
        // Owners: a region holding an air cell's centre is that cell's; a cell without faces
        // owns its cube. Then, layer by layer until none is added, each region of the air
        // without an owner goes to the air cell whose regions it shares the most open area
        // with, through pieces open by the exact flag (each piece's measure taken as at least
        // 0); within 1e-12 the lowest cell index (counted in ties).
        for (std::size_t k = 0; k < cubes_.size(); ++k) {
            const Cube &C = cubes_[k];
            Region &G = regions_[static_cast<std::size_t>(cube_first_[k] + C.centre)];
            if (type_[C.cell] == kAir) G.owner = static_cast<std::int64_t>(C.cell);
        }
        for (std::size_t w = 0; w < whole_.size(); ++w)
            if (type_[whole_[w]] != kSolid) regions_[static_cast<std::size_t>(whole_first_) + w].owner = static_cast<std::int64_t>(whole_[w]);
        struct Offer {
            std::int64_t region, owner;
            double area;
        };
        std::vector<Offer> offers;
        for (;;) {
            offers.clear();
            for (const Square &S : squares_) {
                for (const Piece &P : S.pieces) {
                    if (!P.is_open || P.low < 0 || P.high < 0) continue;
                    const Region &A = regions_[static_cast<std::size_t>(P.low)];
                    const Region &B = regions_[static_cast<std::size_t>(P.high)];
                    auto offer = [&](const Region &to, std::int64_t r, const Region &by) {
                        if (to.owner >= 0 || !to.in_air || by.owner < 0) return;
                        if (type_[static_cast<std::size_t>(by.owner)] != kAir) return;
                        offers.push_back({r, by.owner, std::max(P.open, 0.0)});
                    };
                    offer(A, P.low, B);
                    offer(B, P.high, A);
                }
            }
            if (offers.empty()) break;
            std::stable_sort(offers.begin(), offers.end(), [](const Offer &x, const Offer &y) {
                return x.region != y.region ? x.region < y.region : x.owner < y.owner;
            });
            std::vector<std::pair<std::int64_t, std::int64_t>> given;
            for (std::size_t k = 0; k < offers.size();) {
                const std::int64_t r = offers[k].region;
                std::int64_t best = kNone;
                double most = -1.0;
                bool tie = false;
                while (k < offers.size() && offers[k].region == r) {
                    const std::int64_t o = offers[k].owner;
                    double area = 0.0;
                    while (k < offers.size() && offers[k].region == r && offers[k].owner == o) area += offers[k++].area;
                    if (std::abs(std::abs(area - most) - 1e-12) <= 1e-13) stats_.owner_near_band += 1;
                    if (area > most + 1e-12) {
                        tie = false;
                        most = area;
                        best = o;
                    } else if (std::abs(area - most) <= 1e-12) {
                        tie = true;
                    }
                }
                if (tie) stats_.ties += 1;
                given.push_back({r, best});
            }
            // Every round gives an owner to a region that had none (an offer is made only to such a
            // region); a round that gives none would repeat for ever.
            bool gave = false;
            for (const auto &[r, o] : given) {
                Region &G = regions_[static_cast<std::size_t>(r)];
                gave = gave || G.owner < 0;
                G.owner = o;
            }
            if (!gave)
                throw std::logic_error("fitted walls: self-check check_owners failed at the owner rule: a round gave no region "
                                       "an owner that had none (" + std::to_string(given.size()) + " offers taken)");
        }
        check_owners();
        // T-4: a region of the air left without an owner (open only to the matched layer).
        const Region *unowned = nullptr;
        for (const Region &G : regions_) {
            if (!G.in_air || G.owner >= 0) continue;
            stats_.unowned_regions += 1;
            stats_.unowned_volume += G.volume;
            if (!unowned) unowned = &G;
        }
        if (unowned) {
            const Vec3 c = centre_of(unowned->cell);
            refuse(kRefusedLayerReach,
                "fitted walls: the model reaches within a cell of the matched layer near (" + std::to_string(c.x) + ", " +
                std::to_string(c.y) + ", " + std::to_string(c.z) + ") m: a part of the air the source reaches there (" +
                std::to_string(stats_.unowned_volume * cell_ * cell_ * cell_) +
                " m^3) is open only to the layer, and no cell of the grid can hold it. Move the model away from the "
                "grid's edge, or enlarge the margin round it.");
        }
        // The other components: with a centre, without one (sealed, or open only to the edge).
        std::vector<char> centre(uf.up.size(), 0), to_edge(uf.up.size(), 0);
        std::vector<double> vol(uf.up.size(), 0.0);
        for (std::size_t r = static_cast<std::size_t>(base); r < uf.up.size(); ++r) centre[uf.find(r)] = 1;
        for (std::size_t k = 0; k < cubes_.size(); ++k) centre[uf.find(static_cast<std::size_t>(cube_first_[k] + cubes_[k].centre))] = 1;
        for (std::size_t w = 0; w < whole_.size(); ++w) centre[uf.find(static_cast<std::size_t>(whole_first_) + w)] = 1;
        for (std::size_t r = 0; r < static_cast<std::size_t>(whole_first_); ++r) vol[uf.find(r)] += regions_[r].volume;
        for (const Square &S : squares_)
            for (const Piece &P : S.pieces) {
                if (!P.is_open) continue;
                if (P.low == kEdge && P.high >= 0) to_edge[uf.find(static_cast<std::size_t>(P.high))] = 1;
                if (P.high == kEdge && P.low >= 0) to_edge[uf.find(static_cast<std::size_t>(P.low))] = 1;
            }
        for (std::size_t r = 0; r < uf.up.size(); ++r) {
            if (uf.find(r) != r || r == omega) continue;
            if (centre[r]) {
                stats_.other_components += 1;
                stats_.other_volume += vol[r];
            } else if (to_edge[r]) {
                stats_.edge_pockets += 1;
                stats_.edge_pocket_volume += vol[r];
            } else {
                stats_.sealed_components += 1;
                stats_.sealed_volume += vol[r];
            }
        }
        // The invariant: no region of the air holds a solid cell's centre.
        for (std::size_t k = 0; k < cubes_.size(); ++k) {
            const Cube &C = cubes_[k];
            if (regions_[static_cast<std::size_t>(cube_first_[k] + C.centre)].in_air && type_[C.cell] == kSolid) stats_.leaks += 1;
        }
    }

    // Refusal (ii): a candidate pair whose faces each have a part of positive area facing
    // the source's air on its side towards the other (DESIGN 5.3; read on the whole face, a
    // superset of the part over O': it refuses more, never less).
    void thin_layers(UF &uf, std::size_t omega) {
        std::vector<char> plus(faces_.size(), 0), minus(faces_.size(), 0);
        auto mark = [&](const std::array<std::int64_t, 3> &e, std::int64_t first) {
            const auto t = static_cast<std::size_t>(e[0]);
            if (e[1] >= 0 && uf.find(static_cast<std::size_t>(first + e[1])) == omega) plus[t] = 1;
            if (e[2] >= 0 && uf.find(static_cast<std::size_t>(first + e[2])) == omega) minus[t] = 1;
        };
        for (std::size_t k = 0; k < cubes_.size(); ++k)
            for (const auto &e : cubes_[k].layer_sides) mark(e, cube_first_[k]);
        for (const Square &S : squares_)
            for (const auto &e : S.layer_sides) mark(e, 0);
        for (const LayerPair &L : layer_pairs_) {
            const Plane &P1 = planes_[static_cast<std::size_t>(faces_[L.f1].plane)];
            const Plane &P2 = planes_[static_cast<std::size_t>(faces_[L.f2].plane)];
            const int s1 = P1.n[L.axis] > 0 ? 1 : -1, s2 = P2.n[L.axis] > 0 ? 1 : -1;
            auto faces_air = [&](std::size_t t, int side) { return side > 0 ? plus[t] != 0 : minus[t] != 0; };
            const bool up = L.up && faces_air(L.f1, s1) && faces_air(L.f2, -s2);
            const bool down = L.down && faces_air(L.f1, -s1) && faces_air(L.f2, s2);
            if (!up && !down) continue;
            refuse(kRefusedThinLayer,
                   "fitted walls: faces " + std::to_string(faces_[L.f1].face) + " and " + std::to_string(faces_[L.f2].face) +
                       " are nearly parallel (under 7.2 degrees), their projections along the " + "xyz"[L.axis] +
                       " axis overlap over " + significant(L.area) + " m^2 (" + significant(L.area / (cell_ * cell_)) +
                       " cell faces), they lie within 7/4 kGap = " +
                       std::to_string(1.75 * kGap * cell_) + " m of each other along it there, and the room's air reaches "
                       "the side of each that faces the other (a face laid on another, a box standing on another, two "
                       "triangulations of one wall, or a corner sharper than 7.2 degrees); fitted walls would count both as "
                       "wall. Make them one face, weld them, or keep them at least kGap apart.");
        }
    }

    // A refused fit: no open side, every cell kSolid, no row.
    void clear_rows() {
        open_sides_ = {};
        type_.assign(n_[0] * n_[1] * n_[2], kSolid);
        row_cell_.clear();
        air_volume_.clear();
        open_area_.clear();
        wall_area_.clear();
        leak_area_row_.clear();
        sliver_.clear();
    }

    // x with three significant digits, for a message (std::to_string prints a small area as 0.000000).
    static std::string significant(double x) {
        char text[32];
        std::snprintf(text, sizeof text, "%.3g", x);
        return text;
    }

    Vec3 centre_of(std::size_t i) const {
        const auto p = at(i);
        return {origin_.x + (static_cast<double>(p[0]) + 0.5) * cell_, origin_.y + (static_cast<double>(p[1]) + 0.5) * cell_,
                origin_.z + (static_cast<double>(p[2]) + 0.5) * cell_};
    }

    // ---- 5. the rows ---------------------------------------------------------------------------

    std::int64_t owner_of(std::int64_t r) const { return r == kEdge ? kEdge : regions_[static_cast<std::size_t>(r)].owner; }
    // The row of an owner cell (a cell of row_cell_), through row_index_.
    std::size_t row(std::int64_t cell) const { return static_cast<std::size_t>(row_index_[static_cast<std::size_t>(cell)]); }

    // Calls f(i) for every cell i at depth 1 (a cell that shares a face with an edge cell), in
    // ascending order of i.
    template <typename F>
    void for_depth_one(F f) const {
        if (n_[0] < 3 || n_[1] < 3 || n_[2] < 3) return;
        const std::size_t lx = n_[0] - 2, ly = n_[1] - 2, lz = n_[2] - 2;
        for (std::size_t z = 1; z <= lz; ++z)
            for (std::size_t y = 1; y <= ly; ++y) {
                if (z == 1 || z == lz || y == 1 || y == ly) {
                    for (std::size_t x = 1; x <= lx; ++x) f(index(x, y, z));
                } else {
                    f(index(1, y, z));
                    if (lx > 1) f(index(lx, y, z));
                }
            }
    }

    // The rows: every air cell that owns a region, and every air cell next to an edge cell
    // (which may own none: it then holds its own volume between its planes). Each row's air
    // volume, walls, open areas and slivers are the sums of what it owns; every row's faces
    // to edge cells and to cells without faces, where neither side is a cube, are added whole.
    void sum_up() {
        for (const Region &R : regions_)
            if (R.owner >= 0 && type_[static_cast<std::size_t>(R.owner)] == kAir) row_cell_.push_back(R.owner);
        for_depth_one([&](std::size_t i) {
            if (type_[i] == kAir) row_cell_.push_back(static_cast<std::int64_t>(i));
        });
        std::sort(row_cell_.begin(), row_cell_.end());
        row_cell_.erase(std::unique(row_cell_.begin(), row_cell_.end()), row_cell_.end());
        row_index_.assign(type_.size(), -1);
        for (std::size_t r = 0; r < row_cell_.size(); ++r) row_index_[static_cast<std::size_t>(row_cell_[r])] = static_cast<std::int32_t>(r);
        const std::size_t n = row_cell_.size(), w = materials_ + 1;
        air_volume_.assign(n, 0.0);
        open_area_.assign(n * 18, 0.0);
        wall_area_.assign(n * w, 0.0);
        leak_area_row_.assign(n, 0.0);
        sliver_.assign(n * 6, 0.0);
        std::vector<double> sliver_area(n * 6, 0.0);
        auto is_row = [&](std::int64_t o) { return o >= 0 && type_[static_cast<std::size_t>(o)] == kAir; };
        auto add_wall = [&](std::int64_t o, const Walls &walls) {
            const std::size_t r = row(o);
            for (const auto &[m, a] : walls) {
                const std::size_t col = m < 0 ? materials_ : static_cast<std::size_t>(m);
                wall_area_[r * w + col] += a;
            }
        };
        std::vector<char> owns(n, 0);
        for (const Region &R : regions_) {
            if (!is_row(R.owner)) continue;
            const std::size_t r = row(R.owner), o = static_cast<std::size_t>(R.owner);
            owns[r] = 1;
            air_volume_[r] += R.volume;
            add_wall(R.owner, R.wall);
            const std::size_t distant = distance(o, R.cell);
            stats_.farthest = std::max(stats_.farthest, distant);
            if (distant > 1) stats_.far_volume += R.volume;
            const int k = direction(o, R.cell);
            if (k >= 0 && k < 6) sliver_[r * 6 + static_cast<std::size_t>(k)] += R.volume;
        }
        // A row that owns no region (an air cell next to an edge cell, its cube and every
        // neighbour's without faces) holds its own volume.
        for (std::size_t r = 0; r < n; ++r)
            if (!owns[r]) air_volume_[r] = cell_volume(static_cast<std::size_t>(row_cell_[r]));
        for (const Square &S : squares_) {
            for (const Piece &P : S.pieces) {
                const std::int64_t oa = owner_of(P.low), ob = owner_of(P.high);
                if (is_row(oa)) add_wall(oa, P.wall_low);
                if (is_row(ob)) add_wall(ob, P.wall_high);
                if (!P.is_open) continue;
                const double open = std::max(P.open, 0.0);
                if (oa == kEdge || ob == kEdge) {
                    const std::int64_t o = oa == kEdge ? ob : oa;
                    if (is_row(o)) {
                        wall_area_[row(o) * w + materials_] += open;
                        stats_.edge_area += open;
                    }
                    continue;
                }
                if (oa < 0 || ob < 0) {
                    if ((oa >= 0) != (ob >= 0)) stats_.dropped_area += open;
                    continue;
                }
                if (oa == ob) {
                    const Region &A = regions_[static_cast<std::size_t>(P.low)];
                    const Region &B = regions_[static_cast<std::size_t>(P.high)];
                    const std::size_t o = static_cast<std::size_t>(oa);
                    if ((A.cell == o) != (B.cell == o) && is_row(oa)) {
                        const int k = direction(o, A.cell == o ? B.cell : A.cell);
                        if (k >= 0 && k < 6) sliver_area[row(oa) * 6 + static_cast<std::size_t>(k)] += open;
                    }
                    continue;
                }
                const int k = direction(static_cast<std::size_t>(oa), static_cast<std::size_t>(ob));
                const bool both = is_row(oa) && is_row(ob);
                if (k < 0 || (k >= 6 && !both)) {
                    stats_.dropped_area += open;
                    continue;
                }
                if (k >= 6) stats_.diagonal_area += open;
                if (is_row(oa)) open_area_[row(oa) * 18 + static_cast<std::size_t>(k)] += open;
                if (is_row(ob)) open_area_[row(ob) * 18 + static_cast<std::size_t>(opposite(k))] += open;
            }
        }
        // A row's faces that no square was cut for (its cube and the one across both without
        // faces): whole, to the edge column when the cell across is an edge cell.
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t o = static_cast<std::size_t>(row_cell_[r]);
            if (cube_at(o)) continue;
            for (int k = 0; k < 6; ++k) {
                const std::size_t j = neighbour(o, k);
                if (cube_at(j)) continue;
                const double whole = face_area_of(o, k / 2);
                if (edge(j)) {
                    wall_area_[r * w + materials_] += whole;
                    stats_.edge_area += whole;
                } else if (type_[j] != kSolid) {
                    open_area_[r * 18 + static_cast<std::size_t>(k)] += whole;
                } else {
                    stats_.leaks += 1;  // cannot happen: a cell without faces next to air is in the air
                }
            }
        }
        // The cut air's thickness through each face: its volume over the open area it was
        // taken through (0 where that area is not above 1e-9 cell faces: a measure only).
        for (std::size_t q = 0; q < n * 6; ++q) {
            if (std::abs(sliver_area[q] - 1e-9) <= 1e-12) stats_.sliver_near_gate += 1;
            sliver_[q] = sliver_area[q] > 1e-9 ? sliver_[q] / sliver_area[q] : 0.0;
        }
        check_rows(sliver_area);
        // The air's total (cells^3, a measure on Q): the rows, and each air cell that owns
        // nothing cut, at its own volume between its planes.
        // (Neumaier's compensated sum, in cell order: a measure.)
        double total = 0.0, lost = 0.0;
        auto add = [&](double x) {
            const double t = total + x;
            lost += std::abs(total) >= std::abs(x) ? (total - t) + x : (x - t) + total;
            total = t;
        };
        for (std::size_t r = 0; r < n; ++r) add(air_volume_[r]);
        for (std::size_t i = 0; i < type_.size(); ++i)
            if (type_[i] == kAir && row_index_[i] < 0)
                add(cell_volume(i));
        stats_.air_volume = total + lost;
        row_index_.clear();
        row_index_.shrink_to_fit();
    }

    // Marks the open sides of the grid (open_sides_, -x, +x, -y, +y, -z, +z) from the cells'
    // kinds and the pieces' exact open flags, never from an area. Side 2a (2a + 1) is open when
    // an air cell lies next to the edge cell across its - (+) face along axis a, or when an open
    // piece of a square in the grid's edge plane on that side joins a region of the source's air
    // to the edge cell (the air of a cube next to the edge whose own centre is solid meets the
    // edge there). The layer's cells mark their sides where they are found (a cell of the
    // matched layer within the layer's width of a side). A closed piece (a wall in the edge
    // plane) and a component of air the source does not reach (an edge pocket among them) mark
    // no side.
    void sides() {
        for_depth_one([&](std::size_t i) {
            if (type_[i] != kAir) return;
            const auto p = at(i);
            for (std::size_t a = 0; a < 3; ++a) {
                if (p[a] == 1) open_sides_[2 * a] = true;
                if (p[a] + 2 == n_[a]) open_sides_[2 * a + 1] = true;
            }
        });
        for (const Square &S : squares_)
            for (const Piece &P : S.pieces) {
                if (!P.is_open) continue;
                const std::size_t a = static_cast<std::size_t>(S.axis);
                if (P.low == kEdge && P.high >= 0 && regions_[static_cast<std::size_t>(P.high)].in_air) open_sides_[2 * a] = true;
                if (P.high == kEdge && P.low >= 0 && regions_[static_cast<std::size_t>(P.low)].in_air) open_sides_[2 * a + 1] = true;
            }
    }

    // ---- the self-checks of the owners and the rows ------------------------------------------

    // K8: every owned region of the air is the region holding its owner's centre, or its
    // owner's whole cell, or shares an open piece with a region of the same owner.
    void check_owners() {
        std::vector<char> base(regions_.size(), 0), linked(regions_.size(), 0);
        for (std::size_t k = 0; k < cubes_.size(); ++k)
            base[static_cast<std::size_t>(cube_first_[k] + cubes_[k].centre)] = 1;
        for (std::size_t w = 0; w < whole_.size(); ++w) base[static_cast<std::size_t>(whole_first_) + w] = 1;
        for (const Square &S : squares_)
            for (const Piece &P : S.pieces) {
                if (!P.is_open || P.low < 0 || P.high < 0) continue;
                const Region &A = regions_[static_cast<std::size_t>(P.low)];
                const Region &B = regions_[static_cast<std::size_t>(P.high)];
                if (A.owner >= 0 && A.owner == B.owner) linked[static_cast<std::size_t>(P.low)] = linked[static_cast<std::size_t>(P.high)] = 1;
            }
        SelfCheck k8;
        for (std::size_t g = 0; g < regions_.size(); ++g) {
            const Region &R = regions_[g];
            if (R.owner < 0) continue;
            const bool own = base[g] && R.cell == static_cast<std::size_t>(R.owner);
            k8.compare(own || linked[g] ? 0.0 : 1.0, 0.0, -1.0, [&]() {
                return "region " + std::to_string(g) + " of " + cell_text(R.cell) + ", owned by " + cell_text(static_cast<std::size_t>(R.owner));
            });
        }
        finish(k8, "check_owners", stats_.check_owners, stats_.check_owners_failed);
    }

    // K4, K5 and K6: the rows as sum_up stores them, against a pass over the regions and the
    // squares' pieces written apart from it. K4: each row's air volume (the same terms in the
    // same order: exact; a row that owns no region, its cell's volume) and its wall area per
    // column (the edge column holding the open pieces to edge cells of the regions it owns and
    // its whole faces to edge cells), and that every air cell next to an edge cell is a row and
    // every row owning no region is such a cell. K5: the open area between two rows, each seen
    // from both. K6: every open piece falls in exactly one of the cases sum_up distinguishes,
    // and what each case adds is found where it belongs: the open areas by row and direction,
    // the open area of each row's slivers, and Stats' edge, dropped and diagonal areas.
    void check_rows(const std::vector<double> &sliver_area) {
        const std::size_t n = row_cell_.size(), w = materials_ + 1;
        // Each cell's row, read from row_cell_ here (not sum_up's index), or -1.
        std::vector<std::int32_t> row_at(type_.size(), -1);
        for (std::size_t r = 0; r < n; ++r) row_at[static_cast<std::size_t>(row_cell_[r])] = static_cast<std::int32_t>(r);
        auto row_of = [&](std::int64_t o) -> std::int64_t {
            if (o < 0 || type_[static_cast<std::size_t>(o)] != kAir) return -1;
            return row_at[static_cast<std::size_t>(o)];
        };
        std::vector<double> vol(n, 0.0), wall(n * w, 0.0), wall_scale(n * w, 0.0);
        std::vector<double> open(n * 18, 0.0), open_scale(n * 18, 0.0), same(n * 6, 0.0);
        auto add_walls = [&](std::int64_t r, const Walls &walls) {
            for (const auto &[m, a] : walls) {
                const std::size_t q = static_cast<std::size_t>(r) * w + (m < 0 ? materials_ : static_cast<std::size_t>(m));
                wall[q] += a;
                wall_scale[q] += std::abs(a);
            }
        };
        std::vector<char> owned(n, 0);
        for (const Region &R : regions_) {
            const std::int64_t r = row_of(R.owner);
            if (r < 0) continue;
            owned[static_cast<std::size_t>(r)] = 1;
            vol[static_cast<std::size_t>(r)] += R.volume;
            add_walls(r, R.wall);
        }
        SelfCheck k4, k5, k6;
        // The rows next to the grid's edge: every air cell across a face from an edge cell is a
        // row (found from each edge cell's neighbour inwards), and a row that owns no region is
        // such a cell, holding its own volume.
        for (std::size_t a = 0; a < 3; ++a) {
            const std::size_t b = (a + 1) % 3, c = (a + 2) % 3;
            if (n_[a] < 3 || n_[b] < 3 || n_[c] < 3) break;
            for (std::size_t v = 1; v + 1 < n_[c]; ++v)
                for (std::size_t u = 1; u + 1 < n_[b]; ++u)
                    for (const std::size_t t : {std::size_t{1}, n_[a] - 2}) {
                        std::array<std::size_t, 3> p{};
                        p[a] = t;
                        p[b] = u;
                        p[c] = v;
                        const std::size_t i = index(p[0], p[1], p[2]);
                        if (type_[i] == kAir)
                            k4.compare(row_at[i] >= 0 ? 0.0 : 1.0, 0.0, -1.0,
                                       [&]() { return cell_text(i) + ", an air cell next to the grid's edge, missing from the rows"; });
                    }
        }
        for (std::size_t r = 0; r < n; ++r) {
            if (owned[r]) continue;
            const std::size_t o = static_cast<std::size_t>(row_cell_[r]);
            const auto p = at(o);
            bool next = false;
            for (std::size_t a = 0; a < 3; ++a) next = next || p[a] == 1 || p[a] + 2 == n_[a];
            k4.compare(next ? 0.0 : 1.0, 0.0, -1.0,
                       [&]() { return "the row of " + cell_text(o) + ", which owns no region and does not lie next to the grid's edge"; });
            vol[r] = cell_volume(o);
        }
        // The cases of an open piece of area a between regions owned by oa and ob, each piece in
        // exactly one: to the grid's edge, one or both owners missing, one owner, not adjacent,
        // and adjacent (the open areas of the rows).
        double edge_rows = 0.0, dropped = 0.0, diagonal = 0.0;
        for (const Square &S : squares_) {
            for (const Piece &P : S.pieces) {
                const std::int64_t oa = owner_of(P.low), ob = owner_of(P.high);
                const std::int64_t ra = row_of(oa), rb = row_of(ob);
                if (ra >= 0) add_walls(ra, P.wall_low);
                if (rb >= 0) add_walls(rb, P.wall_high);
                if (!P.is_open) continue;
                const double a = (std::max)(P.open, 0.0);
                if (oa == kEdge || ob == kEdge) {
                    const std::int64_t r = oa == kEdge ? rb : ra;
                    if (r >= 0) {
                        const std::size_t q = static_cast<std::size_t>(r) * w + materials_;
                        wall[q] += a;
                        wall_scale[q] += a;
                        edge_rows += a;
                    }
                } else if (oa < 0 && ob < 0) {
                } else if (oa < 0 || ob < 0) {
                    dropped += a;
                } else if (oa == ob) {
                    const std::size_t o = static_cast<std::size_t>(oa);
                    const std::size_t ca = regions_[static_cast<std::size_t>(P.low)].cell, cb = regions_[static_cast<std::size_t>(P.high)].cell;
                    if (ra >= 0 && (ca == o) != (cb == o)) {
                        const int k = direction(o, ca == o ? cb : ca);
                        if (k >= 0 && k < 6) same[static_cast<std::size_t>(ra) * 6 + static_cast<std::size_t>(k)] += a;
                    }
                } else {
                    const int k = direction(static_cast<std::size_t>(oa), static_cast<std::size_t>(ob));
                    if (k < 0 || (k >= 6 && (ra < 0 || rb < 0))) {
                        dropped += a;
                    } else {
                        if (k >= 6) diagonal += a;
                        if (ra >= 0) {
                            open[static_cast<std::size_t>(ra) * 18 + static_cast<std::size_t>(k)] += a;
                            open_scale[static_cast<std::size_t>(ra) * 18 + static_cast<std::size_t>(k)] += a;
                        }
                        if (rb >= 0) {
                            open[static_cast<std::size_t>(rb) * 18 + static_cast<std::size_t>(opposite(k))] += a;
                            open_scale[static_cast<std::size_t>(rb) * 18 + static_cast<std::size_t>(opposite(k))] += a;
                        }
                    }
                }
            }
        }
        // The whole faces of a row whose cube and neighbour hold no faces, the rows next to the
        // edge included: to the edge column when the cell across is an edge cell, else to the
        // open area when it is not solid.
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t o = static_cast<std::size_t>(row_cell_[r]);
            if (cube_at(o)) continue;
            for (int k = 0; k < 6; ++k) {
                const std::size_t j = neighbour(o, k);
                if (cube_at(j)) continue;
                const double face = face_area_of(o, k / 2);
                if (edge(j)) {
                    wall[r * w + materials_] += face;
                    wall_scale[r * w + materials_] += face;
                    edge_rows += face;
                } else if (type_[j] != kSolid) {
                    open[r * 18 + static_cast<std::size_t>(k)] += face;
                    open_scale[r * 18 + static_cast<std::size_t>(k)] += face;
                }
            }
        }
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t o = static_cast<std::size_t>(row_cell_[r]);
            k4.compare(air_volume_[r], vol[r], -1.0, [&]() { return "the row of " + cell_text(o) + ", its air volume"; });
            for (std::size_t c = 0; c < w; ++c)
                k4.compare(wall_area_[r * w + c], wall[r * w + c], wall_scale[r * w + c],
                           [&]() { return "the row of " + cell_text(o) + ", wall column " + std::to_string(c); });
            for (int k = 0; k < 18; ++k) {
                const std::size_t q = r * 18 + static_cast<std::size_t>(k);
                k6.compare(open_area_[q], open[q], open_scale[q], [&]() { return "the row of " + cell_text(o) + ", open area in direction " + std::to_string(k); });
                const std::int64_t b = row_of(static_cast<std::int64_t>(neighbour(o, k)));
                if (b >= 0)
                    k5.compare(open_area_[q], open_area_[static_cast<std::size_t>(b) * 18 + static_cast<std::size_t>(opposite(k))], -1.0,
                               [&]() { return "the rows of " + cell_text(o) + " and " + cell_text(row_cell_[static_cast<std::size_t>(b)]) + ", direction " + std::to_string(k); });
            }
            for (std::size_t k = 0; k < 6; ++k)
                k6.compare(sliver_area[r * 6 + k], same[r * 6 + k], same[r * 6 + k],
                           [&]() { return "the row of " + cell_text(o) + ", the open area of its sliver in direction " + std::to_string(k); });
        }
        k6.compare(stats_.edge_area, edge_rows, edge_rows, []() { return std::string("Stats::edge_area"); });
        k6.compare(stats_.dropped_area, dropped, dropped, []() { return std::string("Stats::dropped_area"); });
        k6.compare(stats_.diagonal_area, diagonal, diagonal, []() { return std::string("Stats::diagonal_area"); });
        finish(k4, "check_rows", stats_.check_rows, stats_.check_rows_failed);
        finish(k5, "check_open_symmetry", stats_.check_open_symmetry, stats_.check_open_symmetry_failed);
        finish(k6, "check_open", stats_.check_open, stats_.check_open_failed);
    }

    // ---- state --------------------------------------------------------------------------------

    Vec3 origin_;
    double cell_;
    std::array<std::size_t, 3> n_;
    std::size_t sy_, sz_;
    std::size_t layer_;
    std::size_t materials_;
    // The frame: X0 (m) per axis, s, and S = h 2^s quanta per cell.
    double X0_[3] = {0.0, 0.0, 0.0};
    int s_ = 0;
    double S_ = 1.0;
    std::array<std::vector<I128>, 3> G_, Cq_;   // grid planes and centres on the lattice
    std::array<std::vector<double>, 3> width_;  // each cell's width on the lattice (cells)
    std::array<std::vector<int>, 3> grid_plane_;
    std::vector<Plane> planes_;
    std::unordered_map<PlaneKey, int, PlaneHash> plane_map_;
    std::size_t groups_ = 0;
    std::vector<char> given_;
    std::vector<FaceX> faces_;
    std::vector<std::size_t> through_start_;
    std::vector<int> through_;
    std::vector<std::array<std::int64_t, 3>> weld_q_;  // each welded vertex's lattice point
    std::vector<std::array<std::int64_t, 3>> line_;  // (pair of planes, weld a, weld b), sorted
    std::vector<std::int64_t> line_hash_;  // open-addressed: the first entry of line_ for each key
    std::vector<std::array<std::size_t, 3>> membrane_;  // (square's - cell, axis, face), by square then key
    double membrane_face_area_ = 0.0;
    std::vector<Cube> cubes_;
    mutable std::atomic<int> underflow_{0};  // the checking build: an underflow raised in a worker
    std::vector<std::int32_t> row_index_;    // while sum_up runs: each cell's row, or -1
    std::vector<std::size_t> cube_cells_;
    std::vector<std::int64_t> cube_first_;
    std::vector<std::size_t> whole_;
    std::int64_t whole_first_ = 0;
    // Refusal (i): the boundary's odd intervals, the faces with an edge on each line that
    // holds one, the zones (one per interval), R in quanta, and the regions' numbering
    // without the zones.
    std::vector<Interval> intervals_;
    std::vector<std::vector<std::size_t>> line_faces_;
    std::vector<Zone> zones_;
    std::int64_t R_ = 0;
    std::vector<LayerPair> layer_pairs_;  // refusal (ii)'s candidates, by (f1, f2)
    std::vector<std::int64_t> cube_first_z_;
    std::int64_t whole_first_z_ = 0;
    std::vector<Region> regions_;
    std::vector<Square> squares_;
    std::vector<std::uint8_t> type_;
    std::vector<std::int64_t> row_cell_;
    std::vector<double> air_volume_, open_area_, wall_area_, leak_area_row_, sliver_;
    std::array<bool, 6> open_sides_{};  // the open sides of the grid (see open_sides())
    Stats stats_;
    // Counts that every table adds to when it ends (cubes, squares and location, all threads).
    mutable std::atomic<std::size_t> exact_e_{0}, exact_t_{0}, refined_{0};
#ifdef ACOUSTIC_SPACES_CUT_CHECK
public:
    mutable std::atomic<std::size_t> check_contradictions_{0};
#endif
};

}  // namespace acoustic_spaces
