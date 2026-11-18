// Tests of the arithmetic that the fit (cut_cells.hpp) rests on, run in the compiled code of each
// platform: the wide integers' primitives against their portable code, the classes U128 and I128
// against a reference in 32-bit limbs (and against the compiler's 128-bit integers where it has
// them), exact division, the correctly rounded conversions to double, the compiler's conversion
// of 64-bit integers of 2^53 and more to double against the correctly rounded one, the stencil's
// quotient, the double-double operations against exact values in Big, and the arithmetic guard. Each check
// counts its cases and its mismatches; a correct build has no mismatch. The package's tests call
// it through _native._fit_selftest.
#pragma once

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "acoustic_spaces/cut_cells.hpp"

namespace acoustic_spaces {
namespace cut_selftest {

using cut_detail::Big;
using cut_detail::DD;
using cut_detail::I128;
using cut_detail::I256;
using cut_detail::U128;

// splitmix64: a deterministic stream of 64-bit values.
struct Random {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    // A value with a random number of low bits set at random (so that small and large values,
    // and long runs of ones and zeros, all occur).
    std::uint64_t varied() {
        const std::uint64_t x = next(), k = next() % 65;
        return k == 64 ? x : x & ((std::uint64_t{1} << k) - 1);
    }
    double uniform() { return static_cast<double>(static_cast<std::int64_t>(next() >> 11)) * 0x1p-53; }
};

// The edge values of 64-bit words: 0, 1, the powers of two and their neighbours, all ones.
inline std::vector<std::uint64_t> edges64() {
    std::vector<std::uint64_t> v = {0, 1, ~std::uint64_t{0}};
    for (int k = 1; k < 64; ++k) {
        const std::uint64_t p = std::uint64_t{1} << k;
        v.push_back(p - 1);
        v.push_back(p);
        v.push_back(p + 1);
    }
    return v;
}

// A 128-bit value as four 32-bit limbs, least significant first: the reference.
struct L4 {
    std::uint32_t w[4] = {0, 0, 0, 0};
    static L4 of(const U128 &x) {
        L4 r;
        r.w[0] = static_cast<std::uint32_t>(x.lo);
        r.w[1] = static_cast<std::uint32_t>(x.lo >> 32);
        r.w[2] = static_cast<std::uint32_t>(x.hi);
        r.w[3] = static_cast<std::uint32_t>(x.hi >> 32);
        return r;
    }
    bool same(const U128 &x) const {
        const L4 y = of(x);
        return w[0] == y.w[0] && w[1] == y.w[1] && w[2] == y.w[2] && w[3] == y.w[3];
    }
};
inline L4 ref_add(const L4 &a, const L4 &b) {
    L4 r;
    std::uint64_t c = 0;
    for (int i = 0; i < 4; ++i) {
        c += static_cast<std::uint64_t>(a.w[i]) + b.w[i];
        r.w[i] = static_cast<std::uint32_t>(c);
        c >>= 32;
    }
    return r;
}
inline L4 ref_sub(const L4 &a, const L4 &b) {
    L4 nb;
    for (int i = 0; i < 4; ++i) nb.w[i] = ~b.w[i];
    L4 one;
    one.w[0] = 1;
    return ref_add(ref_add(a, nb), one);
}
inline L4 ref_mul(const L4 &a, const L4 &b) {
    std::uint64_t acc[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        std::uint64_t c = 0;
        for (int j = 0; i + j < 4; ++j) {
            const std::uint64_t t = static_cast<std::uint64_t>(a.w[i]) * b.w[j] + (acc[i + j] & 0xffffffffu) + c;
            acc[i + j] = t & 0xffffffffu;
            c = t >> 32;
        }
    }
    L4 r;
    for (int i = 0; i < 4; ++i) r.w[i] = static_cast<std::uint32_t>(acc[i]);
    return r;
}
inline L4 ref_shl(const L4 &a, int k) {
    L4 r;
    for (int i = 0; i < 128; ++i) {
        const int j = i - k;
        if (j >= 0 && j < 128 && ((a.w[j / 32] >> (j % 32)) & 1u)) r.w[i / 32] |= std::uint32_t{1} << (i % 32);
    }
    return r;
}
inline L4 ref_shr(const L4 &a, int k, bool arithmetic) {
    L4 r;
    const bool top = (a.w[3] >> 31) != 0;
    for (int i = 0; i < 128; ++i) {
        const int j = i + k;
        const bool bit = j < 128 ? ((a.w[j / 32] >> (j % 32)) & 1u) != 0 : (arithmetic && top);
        if (bit) r.w[i / 32] |= std::uint32_t{1} << (i % 32);
    }
    return r;
}
inline int ref_cmp(const L4 &a, const L4 &b) {
    for (int i = 3; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}

// 2^k as a Big.
inline Big pow2(int k) {
    Big r(1);
    while (k >= 100) {
        r = r * Big(I128(1) << 100);
        k -= 100;
    }
    return r * Big(I128(1) << k);
}
// v 2^s exactly, for a v whose last bit lies at 2^-s or above.
inline Big big_of(double v, int s) {
    if (v == 0.0) return Big(0);
    int e = 0;
    const double m = std::frexp(v, &e);
    const auto mant = static_cast<std::int64_t>(std::ldexp(m, 53));
    return Big(mant) * pow2(e - 53 + s);
}
inline Big big_abs(const Big &x) { return x.sign() < 0 ? -x : x; }
inline Big big_of(const DD &x, int s) { return big_of(x.hi, s) + big_of(x.lo, s); }
// |r - e| 2^106 <= c |e|: r within c 2^-106 relative of e.
inline bool within(const Big &r, const Big &e, int c) {
    return Big::compare(big_abs(r - e) * pow2(106), big_abs(e) * Big(c)) <= 0;
}
// Whether d is the double nearest the integer x (ties to even), x of at most 600 bits.
inline bool nearest(const Big &x, double d) {
    if (x.sign() == 0) return d == 0.0 && !std::signbit(d);
    if (d == 0.0 || (d < 0.0) != (x.sign() < 0)) return false;
    int e = 0;
    const double m = std::frexp(std::abs(d), &e);
    const auto mant = static_cast<std::int64_t>(std::ldexp(m, 53));
    const int shift = e - 53;
    if (shift < 0) {
        // |d| < 2^53: an integer x is converted exactly.
        return std::abs(d) == std::floor(std::abs(d)) && Big::compare(big_abs(x), Big(static_cast<std::int64_t>(std::abs(d)))) == 0;
    }
    // 2 |x| - 2 D against the unit 2^shift (half of it below a power of two, from below).
    const Big D = Big(mant) * pow2(shift);
    const Big diff = big_abs(x) - D;
    const bool below_power = mant == (std::int64_t{1} << 52) && diff.sign() < 0;
    const Big lhs = big_abs(diff) * Big(below_power ? 4 : 2), unit = pow2(shift);
    const int c = Big::compare(lhs, unit);
    return c < 0 || (c == 0 && (mant % 2) == 0);
}
// Whether q is the double nearest n / d (0 <= n <= d), ties to even.
inline bool nearest_quotient(std::int64_t n, std::int64_t d, double q) {
    if (n == 0) return q == 0.0;
    int e = 0;
    const double m = std::frexp(q, &e);
    const auto mant = static_cast<std::int64_t>(std::ldexp(m, 53));
    const int k = 53 - e;  // q = mant 2^-k
    // |mant d - n 2^k| against d / 2 (d / 4 below a power of two, from below).
    const Big diff = Big(mant) * Big(d) - Big(n) * pow2(k);
    const bool below_power = mant == (std::int64_t{1} << 52) && diff.sign() > 0;
    const int c = Big::compare(big_abs(diff) * Big(below_power ? 4 : 2), Big(d));
    return c < 0 || (c == 0 && (mant % 2) == 0);
}

struct Counts {
    std::vector<std::pair<std::string, std::uint64_t>> items;
    void add(const std::string &name, std::uint64_t cases, std::uint64_t mismatches) {
        items.push_back({name + "_cases", cases});
        items.push_back({name + "_mismatches", mismatches});
    }
};

// Runs every check on its edge values and `cases` random values from `seed`.
inline std::vector<std::pair<std::string, std::uint64_t>> run(std::size_t cases, std::uint64_t seed) {
    Counts out;
    Random rnd{seed};
    const std::vector<std::uint64_t> E = edges64();
    std::vector<std::uint64_t> words = E;
    for (std::size_t i = 0; i < cases; ++i) words.push_back(rnd.varied());

    // The primitives against their portable code.
    {
        std::uint64_t n = 0, bad = 0, nc = 0, badc = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint64_t a = words[i], b = words[(i * 7 + 3) % words.size()];
            std::uint64_t h1 = 0, h2 = 0;
            const std::uint64_t l1 = cut_detail::mul64(a, b, h1), l2 = cut_detail::mul64_portable(a, b, h2);
            ++n;
            bad += (l1 != l2 || h1 != h2) ? 1 : 0;
            if (a != 0) {
                nc += 2;
                badc += cut_detail::clz64(a) != cut_detail::clz64_portable(a) ? 1 : 0;
                badc += cut_detail::ctz64(a) != cut_detail::ctz64_portable(a) ? 1 : 0;
            }
        }
        for (const std::uint64_t a : E)
            for (const std::uint64_t b : E) {
                std::uint64_t h1 = 0, h2 = 0;
                const std::uint64_t l1 = cut_detail::mul64(a, b, h1), l2 = cut_detail::mul64_portable(a, b, h2);
                ++n;
                bad += (l1 != l2 || h1 != h2) ? 1 : 0;
            }
        // The portable counts against their definition, bit by bit.
        for (int k = 0; k < 64; ++k) {
            const std::uint64_t p = std::uint64_t{1} << k;
            nc += 2;
            badc += cut_detail::ctz64_portable(p) != k ? 1 : 0;
            badc += cut_detail::clz64_portable(p | (p - 1)) != 63 - k ? 1 : 0;
        }
        out.add("mul64", n, bad);
        out.add("zero_counts", nc, badc);
    }

    // U128 and I128 against the 32-bit limbs (and the compiler's __int128 where it has one).
    {
        std::uint64_t n = 0, bad = 0;
        auto u128 = [&](std::size_t i) { return U128::of(words[i % words.size()], words[(i * 13 + 5) % words.size()]); };
        const std::size_t N = words.size() * 2;
        for (std::size_t i = 0; i < N; ++i) {
            const U128 a = u128(i), b = u128(i * 31 + 17);
            const L4 ra = L4::of(a), rb = L4::of(b);
            const int k = static_cast<int>(rnd.next() % 130);
            const I128 sa(a), sb(b);
            bool ok = ref_add(ra, rb).same(a + b) && ref_sub(ra, rb).same(a - b) && ref_mul(ra, rb).same(a * b) &&
                      ref_shl(ra, k).same(a << k) && ref_shr(ra, k, false).same(a >> k) &&
                      ref_shr(ra, k, true).same((sa >> k).u) && ((a < b) == (ref_cmp(ra, rb) < 0)) && ((a == b) == (ref_cmp(ra, rb) == 0));
            // Signed order: flip the top bits and compare unsigned.
            L4 fa = ra, fb = rb;
            fa.w[3] ^= 0x80000000u;
            fb.w[3] ^= 0x80000000u;
            ok = ok && ((sa < sb) == (ref_cmp(fa, fb) < 0));
            if (b != U128(0)) {
                // a = q b + r with r < b, by the reference's product and sum.
                const U128 q = a / b, r = a % b;
                ok = ok && ref_add(ref_mul(L4::of(q), rb), L4::of(r)).same(a) && ref_cmp(L4::of(r), rb) < 0;
            }
            if (sb != I128(0) && !(sa == (I128(1) << 127) && sb == I128(-1))) {
                const I128 q = sa / sb, r = sa % sb;
                ok = ok && (q * sb + r == sa) && (r == I128(0) || r.negative() == sa.negative());
            }
#if defined(__SIZEOF_INT128__)
            __extension__ typedef unsigned __int128 NU;
            const NU na = (static_cast<NU>(a.hi) << 64) | a.lo, nb = (static_cast<NU>(b.hi) << 64) | b.lo;
            auto eq = [](NU x, const U128 &y) { return static_cast<std::uint64_t>(x) == y.lo && static_cast<std::uint64_t>(x >> 64) == y.hi; };
            ok = ok && eq(na + nb, a + b) && eq(na - nb, a - b) && eq(na * nb, a * b);
            if (k < 128) ok = ok && eq(na << k, a << k) && eq(na >> k, a >> k);
            if (nb != 0) ok = ok && eq(na / nb, a / b) && eq(na % nb, a % b);
#endif
            ++n;
            bad += ok ? 0 : 1;
        }
        out.add("wide_integers", n, bad);
    }

    // Exact division of random multiples.
    {
        std::uint64_t n = 0, bad = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            U128 g = U128::of(0, words[i] | 1u) << static_cast<int>(rnd.next() % 40);
            g = g >> static_cast<int>(rnd.next() % 8);
            if (g == U128(0)) continue;
            const I128 y = I128(static_cast<std::int64_t>(rnd.next())) >> static_cast<int>(rnd.next() % 64);
            const I128 x = y * I128(g);
            if (I128(g) != I128(0) && x / I128(g) != y) continue;  // the product wrapped
            ++n;
            bad += cut_detail::exact_div(x, g) != y ? 1 : 0;
        }
        out.add("exact_division", n, bad);
    }

    // The conversions to double, correctly rounded, with halfway cases built on purpose.
    {
        std::uint64_t n = 0, bad = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            // A 54-bit odd value (a midpoint between two doubles once shifted), its neighbours, and random values.
            const std::uint64_t mid = (words[i] & ((std::uint64_t{1} << 54) - 1)) | (std::uint64_t{1} << 53) | 1u;
            const int k = static_cast<int>(rnd.next() % 70);
            for (int d = -1; d <= 1; ++d) {
                const U128 x = (U128::of(0, mid) << k) + U128(static_cast<std::int64_t>(d));
                ++n;
                bad += nearest(Big(I128(x)), static_cast<double>(x)) ? 0 : 1;
                const I128 neg = -I128(x);
                ++n;
                bad += nearest(Big(neg), static_cast<double>(neg)) ? 0 : 1;
                const I256 w = I256::mul(I128(x), static_cast<std::int64_t>(words[(i * 3 + 1) % words.size()] >> 2));
                ++n;
                bad += nearest(Big::of_limbs(w.w), w.to_double()) ? 0 : 1;
                const Big b = Big(I128(x)) * Big(I128(x)) + Big(d);
                ++n;
                bad += nearest(b, b.to_double()) ? 0 : 1;
            }
            const U128 r = U128::of(words[i] >> 1, words[(i + 1) % words.size()]);
            ++n;
            bad += nearest(Big(I128(r)), static_cast<double>(r)) ? 0 : 1;
        }
        out.add("conversion", n, bad);
    }

    // The compiler's conversion of 64-bit integers of 2^53 and more in magnitude to double (which the fit uses
    // for lattice coordinates) against the correctly rounded conversion of I128: random values of every
    // magnitude from 2^53 to 2^63, values halfway between two doubles and their neighbours, both signs.
    {
        std::uint64_t n = 0, bad = 0;
        auto one = [&](std::int64_t v) {
            const volatile std::int64_t vv = v;
            const double compiler = static_cast<double>(vv);
            ++n;
            bad += (compiler == static_cast<double>(I128(v)) && nearest(Big(I128(v)), compiler)) ? 0 : 1;
        };
        one(std::numeric_limits<std::int64_t>::min());
        one(std::numeric_limits<std::int64_t>::max());
        for (std::size_t i = 0; i < words.size(); ++i) {
            const int top = 53 + static_cast<int>(rnd.next() % 10);  // the highest bit, 53 to 62
            const std::uint64_t mask = (std::uint64_t{1} << top) - 1;
            const auto v = static_cast<std::int64_t>((words[i] & mask) | (std::uint64_t{1} << top));
            one(v);
            one(-v);
            // Halfway: 53 bits of mantissa, then a one, then zeros, at 2^top.
            const int below = top - 53;  // the bits below the halfway bit
            const std::uint64_t mant = (words[(i + 1) % words.size()] >> 11) | (std::uint64_t{1} << 52);
            const auto half = static_cast<std::int64_t>(((mant << 1) | 1u) << below);
            for (int d = -1; d <= 1; ++d) {
                one(half + d);
                one(-(half + d));
            }
        }
        out.add("int64_conversion", n, bad);
    }

    // The stencil's quotient n / d (0 <= n <= d < 2^63), near midpoints and at random.
    {
        std::uint64_t n = 0, bad = 0;
        for (std::size_t i = 0; i < words.size(); ++i) {
            std::int64_t d = static_cast<std::int64_t>(words[i] >> 1);
            if (d == 0) d = 1 + static_cast<std::int64_t>(i);
            // x is one of 0 to d. The modulus d + 1 is formed without a sign, because d reaches 2^63 - 1.
            const std::uint64_t modulus = static_cast<std::uint64_t>(d) + 1;
            const std::int64_t x = static_cast<std::int64_t>((rnd.next() >> 1) % modulus);
            for (const std::int64_t num : {x, d - x, x / 2, d / 3, d}) {
                if (num < 0 || num > d) continue;
                ++n;
                bad += nearest_quotient(num, d, CutCells::quotient(num, d)) ? 0 : 1;
            }
        }
        out.add("quotient", n, bad);
    }

    // The double-double operations against exact values: the sum within 4 2^-106 relative (its
    // bound is 3 u^2 + 13 u^3, u = 2^-53), the product within 7, the quotient's residual z y - x
    // within 16 |x| (15 u^2 + 56 u^3), the square root's residual z^2 - a within 9 a (twice 4.2 u^2).
    {
        std::uint64_t ns = 0, bs = 0, np = 0, bp = 0, nq = 0, bq = 0, nr = 0, br = 0;
        auto value = [&]() {
            const double hi = std::ldexp(1.0 + rnd.uniform(), static_cast<int>(rnd.next() % 41) - 20) * (rnd.next() & 1 ? -1.0 : 1.0);
            const double lo = hi * (rnd.uniform() - 0.5) * 0x1p-52;
            return DD::quick(hi, lo);
        };
        const int s = 300;
        for (std::size_t i = 0; i < cases / 4 + 64; ++i) {
            DD a = value(), b = value();
            if (i % 3 == 0) b = DD(-a.hi, -a.lo * (1.0 + rnd.uniform() * 0x1p-20));  // a sum that cancels
            const Big A = big_of(a, s), B = big_of(b, s);
            const Big sum = A + B;
            ++ns;
            bs += within(big_of(a + b, s), sum, 4) ? 0 : 1;
            ++np;
            bp += within(big_of(a * b, s) * pow2(s), A * B, 7) ? 0 : 1;
            if (b.hi != 0.0) {
                const DD z = a / b;
                ++nq;
                bq += Big::compare(big_abs(big_of(z, s) * B - A * pow2(s)) * pow2(106), big_abs(A) * pow2(s) * Big(16)) <= 0 ? 0 : 1;
            }
            const DD p = cut_detail::dd_abs(a);
            const DD z = cut_detail::dd_sqrt(p);
            const Big Z = big_of(z, s), P = big_of(p, s);
            ++nr;
            br += Big::compare(big_abs(Z * Z - P * pow2(s)) * pow2(106), P * pow2(s) * Big(9)) <= 0 ? 0 : 1;
        }
        out.add("dd_sum", ns, bs);
        out.add("dd_product", np, bp);
        out.add("dd_quotient", nq, bq);
        out.add("dd_sqrt", nr, br);
    }
    return out.items;
}

// The arithmetic guard's three results, reported rather than thrown: whether the compile-time
// constant finds nothing, the name of the probe's first failing test (empty when it passes), and
// whether the rounding mode is to nearest.
struct Guard {
    bool arithmetic_constant = false;
    std::string probe_failed;
    bool rounding_to_nearest = false;
};
inline Guard guard() {
    Guard g;
    g.arithmetic_constant = cut_detail::kArithmeticRefusal == nullptr;
    const cut_detail::ProbeResult r = cut_detail::arithmetic_probe();
    g.probe_failed = r.name ? r.name : "";
    g.rounding_to_nearest = std::fegetround() == FE_TONEAREST;
    return g;
}

}  // namespace cut_selftest
}  // namespace acoustic_spaces
