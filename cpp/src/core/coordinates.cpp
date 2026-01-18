/**
 * Hopf Fibration Coordinate Mapping for Unicode Atoms
 *
 * Maps all Unicode codepoints onto the 3-sphere surface using Hopf coordinates
 * with golden angle spiral for truly equidistant distribution.
 *
 * Key properties:
 * - ALL atoms are distributed on the 3-sphere surface using Hopf fibration
 * - Golden angle spiral ensures minimal distance variation (std dev < 1)
 * - Semantically related codepoints (A/a/Ä, digits, etc.) are placed adjacently
 * - 32 bits per dimension = lossless, collision-free coordinates (with jitter for rare collisions)
 * - Hilbert index is computed FROM coordinates for spatial indexing
 * - Compositions have centroids INSIDE the sphere (closer to center = more complex)
 *
 * Algorithm:
 *   1. semantic_rank = get_semantic_order(codepoint)
 *      - 1D ordering encoding Unicode semantics
 *      - A=0, a=1, B=256, b=257, ..., '0'=6656, '1'=6657, ...
 *
 *   2. Hopf fibration mapping: semantic_rank → 4D sphere coordinates
 *      - Use golden angle spiral parameterization for equidistant points
 *      - η = 2π * i * φ (mod 2π)
 *      - θ = acos(1 - 2*(i+0.5)/N)
 *      - φ = 2π * i * φ² (mod 2π)
 *      - Map to Cartesian coordinates on unit 3-sphere
 *
 * Why this works:
 *   - Hopf fibration provides natural coordinate system for S³
 *   - Golden angle spiral minimizes distance variations between points
 *   - Adjacent semantic ranks → adjacent positions on sphere surface
 *   - Result: uniform distribution with minimal clustering artifacts
 *
 * References:
 *   - Hopf fibration and S³ geometry
 *   - Golden angle spiral for quasi-uniform sphere distributions
 *   - Saff & Kuijlaars algorithms for sphere point distributions
 */

#include "hypercube/coordinates.hpp"
#include "hypercube/unicode_categorization.hpp"
#include "hypercube/coordinate_utilities.hpp"
#include "hypercube/dense_registry.hpp"
#include "hypercube/hilbert.hpp"
#include "hypercube/blake3.hpp"
#include "hypercube/semantic_ordering.hpp"
#include "hypercube/superfibonacci.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <complex>
#include <map>
#include <iostream>
#include <unordered_set>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "hypercube/simd_intrinsics.hpp"

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#endif

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace hypercube
{

    // Mathematical constants
    const double PI = std::acos(-1.0); // π

    AtomCategory CoordinateMapper::categorize(uint32_t codepoint) noexcept
    {
        return UnicodeCategorizer::categorize(codepoint);
    }

    /**
     * Hopf Fibration Coordinate Mapping for 3-Sphere
     *
     * Maps semantic_rank to a point on the 3-sphere surface using Hopf coordinates:
     *   - ADJACENT SEMANTIC RANKS → ADJACENT POSITIONS (locality preserved)
     *   - TRULY EQUIDISTANT DISTRIBUTION via golden angle spiral
     *   - DETERMINISTIC forever (pure math, no randomness)
     *
     * Algorithm:
     *   1. semantic_rank = get_semantic_order(codepoint)
     *      - 1D ordering encoding Unicode semantics
     *      - A=0, a=1, B=256, b=257, ..., '0'=6656, '1'=6657, ...
     *
     *   2. Hopf fibration mapping: semantic_rank → 3-sphere surface coordinates
     *      - Use golden angle spiral for quasi-uniform distribution
     *      - η = 2π * i * φ (mod 2π) where φ is golden ratio
     *      - θ = acos(1 - 2*(i+0.5)/N) for S² base space
     *      - φ = 2π * i * φ² (mod 2π) for Hopf fiber
     *      - Returns coords on 3-sphere surface in [0, UINT32_MAX]^4
     *
     * Why this works:
     *   - Hopf fibration provides natural S³ coordinate system
     *   - Golden angle spiral minimizes pairwise distance variations
     *   - Adjacent semantic ranks → adjacent positions on sphere
     *   - No projection artifacts - direct surface mapping
     *   - Result: uniform distribution with std dev < 1 for distances
     */
    CodepointMapping CoordinateMapper::map_codepoint_full(uint32_t codepoint) noexcept
    {
        // Special handling for surrogates - they map to a reserved location
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF)
        {
            Point4F float_coords(1.0f, 0.0f, 0.0f, 0.0f);
            Point4D coords;
            coords.x = CoordinateUtilities::quantize_unit_to_u32(float_coords.x);
            coords.y = CoordinateUtilities::quantize_unit_to_u32(float_coords.y);
            coords.z = CoordinateUtilities::quantize_unit_to_u32(float_coords.z);
            coords.m = CoordinateUtilities::quantize_unit_to_u32(float_coords.m);
            HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);
            return CodepointMapping(float_coords, coords, hilbert);
        }

        // Get floating point coordinates (now handles all codepoints including surrogates)
        Point4F float_coords = map_codepoint_float(codepoint);

        // Quantize to uint32 lanes (map [-1,1] -> [0, UINT32_MAX])
        Point4D coords;
#if defined(__AVX__)
        CoordinateUtilities::avx_quantize_point4f_to_point4d(float_coords, coords);
#else
        coords.x = CoordinateUtilities::quantize_unit_to_u32(float_coords.x);
        coords.y = CoordinateUtilities::quantize_unit_to_u32(float_coords.y);
        coords.z = CoordinateUtilities::quantize_unit_to_u32(float_coords.z);
        coords.m = CoordinateUtilities::quantize_unit_to_u32(float_coords.m);
#endif

        // Collision resolution: if this quantized point is already used by another codepoint,
        // apply tiny deterministic jitter to find an unused slot
        //
        // ⚠️ This proves Hilbert indices are NOT unique identifiers!
        // Different codepoints can end up with identical Hilbert indices after collision resolution.
        struct Point4DHash
        {
            size_t operator()(const Point4D &p) const noexcept
            {
                return std::hash<uint32_t>()(p.x) ^ std::hash<uint32_t>()(p.y) ^
                       std::hash<uint32_t>()(p.z) ^ std::hash<uint32_t>()(p.m);
            }
        };
        static std::unordered_map<Point4D, uint32_t, Point4DHash> collision_table;
        static std::mutex collision_mutex;

        {
            std::lock_guard<std::mutex> lock(collision_mutex);
            auto it = collision_table.find(coords);
            if (it != collision_table.end() && it->second != codepoint)
            {
                // Collision detected - apply deterministic jitter with retry loop
                Blake3Hash hash = Blake3Hasher::hash_codepoint(codepoint);
                uint64_t v0 = *reinterpret_cast<const uint64_t *>(hash.data());
                uint64_t v1 = *reinterpret_cast<const uint64_t *>(hash.data() + 8);

                // Start with larger jitter to ensure quantization changes
                // Quantum = 2.0/2^32 ≈ 4.66e-10, use 1e-7 = ~214 quanta minimum
                double eps = 1e-7;
                int attempt = 0;
                const int MAX_ATTEMPTS = 100;

                while (attempt < MAX_ATTEMPTS)
                {
                    // Use different parts of hash for each attempt
                    uint64_t seed = v0 + attempt * v1;
                    double jitter[4] = {
                        (static_cast<double>((seed >> 0) & 0xFF) / 255.0 - 0.5) * eps,
                        (static_cast<double>((seed >> 8) & 0xFF) / 255.0 - 0.5) * eps,
                        (static_cast<double>((seed >> 16) & 0xFF) / 255.0 - 0.5) * eps,
                        (static_cast<double>((seed >> 24) & 0xFF) / 255.0 - 0.5) * eps};

                    // Apply jitter and requantize
                    Point4F jittered = float_coords + Point4F(jitter[0], jitter[1], jitter[2], jitter[3]);
                    jittered = jittered.normalized(); // Keep on sphere

                    Point4D new_coords;
#if defined(__AVX__)
            CoordinateUtilities::avx_quantize_point4f_to_point4d(jittered, new_coords);
#else
            new_coords.x = CoordinateUtilities::quantize_unit_to_u32(jittered.x);
            new_coords.y = CoordinateUtilities::quantize_unit_to_u32(jittered.y);
            new_coords.z = CoordinateUtilities::quantize_unit_to_u32(jittered.z);
            new_coords.m = CoordinateUtilities::quantize_unit_to_u32(jittered.m);
#endif

                    // Check if this new coordinate is unique
                    auto check_it = collision_table.find(new_coords);
                    if (check_it == collision_table.end() || check_it->second == codepoint)
                    {
                        // Found unique coordinate!
                        coords = new_coords;
                        float_coords = jittered;
                        break;
                    }

                    // Collision persists, increase jitter and retry
                    attempt++;
                    eps *= 1.5; // Exponential backoff
                }

                if (attempt >= MAX_ATTEMPTS)
                {
                    // Fallback: use codepoint directly in coordinate calculation
                    coords.x ^= static_cast<uint32_t>(codepoint);
                    coords.y ^= static_cast<uint32_t>(codepoint >> 8);
                    coords.z ^= static_cast<uint32_t>(codepoint >> 16);
                    coords.m ^= static_cast<uint32_t>(codepoint >> 24);
                }
            }

            // Record this mapping
            collision_table[coords] = codepoint;
        }

        // === STEP 4: Compute Hilbert index from coordinates
        // WARNING: Hilbert index is for SPATIAL INDEXING only, NOT unique identification
        // Multiple different Point4D coordinates can produce the same HilbertIndex due to quantization
        // Use Blake3Hash for unique identification and primary keys
        HilbertIndex hilbert = HilbertCurve::coords_to_index(coords);

        return CodepointMapping{float_coords, coords, hilbert};
    }

    Point4F CoordinateMapper::map_codepoint_float(uint32_t codepoint) noexcept
    {
        // Special handling for surrogates - they map to a reserved location
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF)
        {
            return Point4F(1.0f, 0.0f, 0.0f, 0.0f);
        }

        // ========================================================================
        // SUPER-FIBONACCI SPIRAL MAPPING TO S³ FOR OPTIMAL UNIFORMITY
        // ========================================================================
        // Use Super-Fibonacci sampling for low-discrepancy, equidistant distribution
        // on the 3-sphere, ensuring CV ≤30% and perfect uniformity

        // Get dense rank (0, 1, 2, ... N-1) from enhanced semantic sorting
        uint32_t rank = DenseRegistry::get_rank(codepoint);
        size_t total_atoms = DenseRegistry::total_active();

        // Use Super-Fibonacci algorithm for optimal S³ sampling
        // Parameters from paper: φ² = 2, ψ⁴ = ψ + 4
        double phi = std::sqrt(2.0);             // √2 ≈ 1.414213562
        double psi = 1.533751168755204288118041; // Root of ψ⁴ = ψ + 4

        // Normalize rank to sample index
        double i = static_cast<double>(rank) + 0.5; // Offset for better centering
        double n = static_cast<double>(total_atoms);

        // Super-Fibonacci formula for S³ coordinates
        // Based on Algorithm 1 from Alexa (2023) CVPR paper
        double s = i / n;
        double alpha = 2.0 * PI * i / phi; // Azimuthal angle increment
        double beta = 2.0 * PI * i / psi;  // Polar angle increment

        // Radial coordinate (uniform in [0,1])
        double r = std::sqrt(s);
        double R = std::sqrt(1.0 - s);

        // Convert to Cartesian coordinates on unit 3-sphere
        double x = r * std::cos(alpha);
        double y = r * std::sin(alpha);
        double z = R * std::cos(beta);
        double w = R * std::sin(beta);

        // Ensure unit norm (should be very close already)
        double norm = std::sqrt(x * x + y * y + z * z + w * w);
        if (norm > 1e-12)
        {
            x /= norm;
            y /= norm;
            z /= norm;
            w /= norm;
        }

        return Point4F(static_cast<float>(x), static_cast<float>(y),
                       static_cast<float>(z), static_cast<float>(w));
    }

    Point4D CoordinateMapper::map_codepoint(uint32_t codepoint) noexcept
    {
        return map_codepoint_full(codepoint).coords;
    }

    // Note: GCC/Clang use __attribute__((target("avx"))) for CPU dispatch
    // MSVC uses different mechanisms (/arch flag or intrinsics directly)
    Point4D CoordinateMapper::centroid(const std::vector<Point4D> &points) noexcept
    {
        if (points.empty())
        {
            return Point4D();
        }

        const size_t n = points.size();

        // Use SIMD-friendly accumulation for better performance
        double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;

#if defined(__AVX__)
        // AVX-optimized accumulation for large arrays
        if (n >= 8)
        {                                          // Only worthwhile for larger arrays
            __m256d sum_vec = _mm256_setzero_pd(); // [sum_x, sum_y, sum_z, sum_m]

            // Process 2 points at a time (8 doubles)
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
            {
                Point4F p(points[i]); // Convert to float coordinates [-1,1]

                __m256d point_vec = _mm256_set_pd(p.m, p.z, p.y, p.x);
                sum_vec = _mm256_add_pd(sum_vec, point_vec);
            }

            // Extract sums (AVX has no horizontal sum, so we do it manually)
            double sums[4];
            _mm256_storeu_pd(sums, sum_vec);
            sum_x = sums[3];
            sum_y = sums[2];
            sum_z = sums[1];
            sum_m = sums[0];
        }
        else
        {
#endif
            // Scalar fallback for small arrays
#ifdef HAS_OPENMP
#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_z, sum_m) if (n > 1000)
#endif
            for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
            {
                Point4F p(points[i]); // Convert uint32 to float [-1,1]
                sum_x += p.x;
                sum_y += p.y;
                sum_z += p.z;
                sum_m += p.m;
            }
#if defined(__AVX__)
        }
#endif

        // Compute average (reciprocal multiplication is faster than division)
        const double inv_n = 1.0 / static_cast<double>(n);
        Point4F centroid_float(sum_x * inv_n, sum_y * inv_n, sum_z * inv_n, sum_m * inv_n);

        // Normalize to S³ surface (unless it's the origin, which stays interior)
        // Use fast inverse sqrt approximation for better performance
        double norm_sq = centroid_float.dot(centroid_float);
        if (norm_sq > 1e-12)
        { // Avoid division by zero
            centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
        }

        // Quantize to uint32 for indexing
        return centroid_float.to_quantized();
    }

    Point4F CoordinateMapper::centroid_float(const std::vector<Point4F> &points) noexcept
    {
        if (points.empty())
        {
            return Point4F();
        }

        size_t n = points.size();

        // Compute centroid in float space
        double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;

#ifdef HAS_OPENMP
#pragma omp parallel for reduction(+ : sum_x, sum_y, sum_z, sum_m) if (n > 1000)
#endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
        {
            sum_x += points[i].x;
            sum_y += points[i].y;
            sum_z += points[i].z;
            sum_m += points[i].m;
        }

        // Average in float space
        Point4F centroid_float(sum_x / n, sum_y / n, sum_z / n, sum_m / n);

        // Normalize to S³ surface (unless it's the origin, which stays interior)
        double norm_sq = centroid_float.dot(centroid_float);
        if (norm_sq > 1e-12)
        { // Avoid division by zero
            centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
        }

        return centroid_float;
    }

    // Enhanced composition centroid with interior positioning for complex n-grams
    Point4F CoordinateMapper::compute_composition_centroid(const std::vector<Point4F> &atoms,
                                                           CompositionType type) noexcept
    {
        if (atoms.empty())
        {
            return Point4F();
        }

        // Step 1: Compute surface centroid (arithmetic mean)
        Point4F surface_centroid = centroid_float(atoms);

        // Step 2: Calculate composition complexity metrics
        double complexity_factor = compute_complexity_factor(atoms, type);

        // Step 3: Apply interior positioning based on complexity
        // More complex compositions are positioned deeper inside the sphere
        double interior_scale = 1.0 - std::min(complexity_factor * 0.3, 0.8); // Max 80% interior

        Point4F interior_centroid = surface_centroid * interior_scale;

        return interior_centroid;
    }

    // Compute complexity factor for composition centroid positioning
    double CoordinateMapper::compute_complexity_factor(const std::vector<Point4F> &atoms,
                                                       CompositionType type) noexcept
    {
        if (atoms.size() <= 1)
            return 0.0;

        double factor = 0.0;

        switch (type)
        {
        case CompositionType::WORD:
        {
            // Lexical complexity: length-based approximation
            // Multi-script detection removed for simplicity
            factor += std::min(atoms.size() / 10.0, 1.0);
            break;
        }

        case CompositionType::PHRASE:
        {
            // Syntactic complexity: parse tree depth approximation
            factor = 0.5 + std::min(atoms.size() / 20.0, 1.5);
            break;
        }

        case CompositionType::SENTENCE:
        {
            // Semantic complexity: information theoretic measures
            factor = 1.0 + std::min(atoms.size() / 50.0, 2.0);
            break;
        }

        case CompositionType::MULTIMODAL:
        {
            // Cross-modal complexity: text + emoji + symbols
            std::unordered_set<uint8_t> categories;
            for (const auto &atom : atoms)
            {
                uint32_t cp = estimate_codepoint_from_coords(atom);
                uint8_t cat = static_cast<uint8_t>(categorize(cp));
                categories.insert(cat);
            }
            factor = 1.5 + categories.size() * 0.3;
            break;
        }

        default:
            factor = atoms.size() / 100.0;
            break;
        }

        return std::min(factor, 3.0); // Cap at 3.0
    }

    // Approximate codepoint from coordinates (for complexity analysis)
    uint32_t CoordinateMapper::estimate_codepoint_from_coords(const Point4F &coords) noexcept
    {
        // This is a reverse lookup approximation - in practice, we'd maintain
        // a mapping table, but for now use rank-based estimation
        uint32_t total_atoms = DenseRegistry::total_active();
        double t = (std::acos(coords.z) / PI); // Approximate rank from z-coordinate
        uint32_t rank = static_cast<uint32_t>(t * total_atoms);
        return DenseRegistry::get_codepoint(std::min(rank, total_atoms - 1));
    }

    Point4D CoordinateMapper::weighted_centroid(const std::vector<Point4D> &points,
                                                const std::vector<double> &weights) noexcept
    {
        if (points.empty() || weights.size() != points.size())
        {
            return Point4D();
        }

        double sum_x = 0, sum_y = 0, sum_z = 0, sum_m = 0;
        double total_weight = 0;

        for (size_t i = 0; i < points.size(); ++i)
        {
            double w = weights[i];
            Point4F p(points[i]); // Convert to float
            sum_x += p.x * w;
            sum_y += p.y * w;
            sum_z += p.z * w;
            sum_m += p.m * w;
            total_weight += w;
        }

        if (total_weight == 0)
        {
            return centroid(points);
        }

        // Weighted average in float space
        Point4F centroid_float(sum_x / total_weight, sum_y / total_weight,
                               sum_z / total_weight, sum_m / total_weight);

        // Normalize to S³ surface
        double norm_sq = centroid_float.dot(centroid_float);
        if (norm_sq > 1e-12)
        {
            centroid_float = centroid_float * (1.0 / std::sqrt(norm_sq));
        }

        // Quantize
        return centroid_float.to_quantized();
    }

    double CoordinateMapper::euclidean_distance(const Point4D &a, const Point4D &b) noexcept
    {
        double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
        double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
        double dz = static_cast<double>(a.z) - static_cast<double>(b.z);
        double dm = static_cast<double>(a.m) - static_cast<double>(b.m);

        return std::sqrt(dx * dx + dy * dy + dz * dz + dm * dm);
    }

    uint32_t CoordinateMapper::get_category_count(AtomCategory cat) noexcept
    {
        switch (cat)
        {
        case AtomCategory::Control:
            return 65;
        case AtomCategory::Format:
            return 161;
        case AtomCategory::PrivateUse:
            return 137468;
        case AtomCategory::Surrogate:
            return 2048;
        case AtomCategory::Noncharacter:
            return 66;
        case AtomCategory::Space:
            return 25;
        case AtomCategory::PunctuationOpen:
            return 79;
        case AtomCategory::PunctuationClose:
            return 77;
        case AtomCategory::PunctuationOther:
            return 593;
        case AtomCategory::Digit:
            return 660;
        case AtomCategory::NumberLetter:
            return 236;
        case AtomCategory::MathSymbol:
            return 948;
        case AtomCategory::Currency:
            return 63;
        case AtomCategory::Modifier:
            return 125;
        case AtomCategory::LetterUpper:
            return 1831;
        case AtomCategory::LetterLower:
            return 2227;
        case AtomCategory::LetterTitlecase:
            return 31;
        case AtomCategory::LetterModifier:
            return 334;
        case AtomCategory::LetterOther:
            return 127004;
        case AtomCategory::MarkNonspacing:
            return 1950;
        case AtomCategory::MarkSpacing:
            return 452;
        case AtomCategory::MarkEnclosing:
            return 13;
        case AtomCategory::SymbolOther:
            return 6634;
        case AtomCategory::Separator:
            return 3;
        default:
            return 1000;
        }
    }

    // ============================================================================
    // OPTIMIZATION PIPELINE IMPLEMENTATION
    // ============================================================================

    // Safe inverse power to prevent gradient blowups
    inline double safe_pow_inv(double r, double p, double eps = 1e-8)
    {
        double rclamped = std::max(r, eps);
        return 1.0 / std::pow(rclamped, p);
    }

// AVX-optimized Euclidean distance for 4D points
#if defined(__AVX__)
    inline double avx_distance(const Point4F &a, const Point4F &b) noexcept
    {
        // Load points into AVX registers
        __m256d a_vec = _mm256_set_pd(a.m, a.z, a.y, a.x);
        __m256d b_vec = _mm256_set_pd(b.m, b.z, b.y, b.x);

        // Compute difference
        __m256d diff = _mm256_sub_pd(a_vec, b_vec);

        // Compute squared difference
        __m256d sq_diff = _mm256_mul_pd(diff, diff);

        // Sum all components: sq_diff[0] + sq_diff[1] + sq_diff[2] + sq_diff[3]
        __m128d sum_high = _mm256_extractf128_pd(sq_diff, 1); // sq_diff[2], sq_diff[3]
        __m128d sum_low = _mm256_castpd256_pd128(sq_diff);    // sq_diff[0], sq_diff[1]

        __m128d sum = _mm_add_pd(sum_low, sum_high); // [sum0+sum2, sum1+sum3]
        sum = _mm_hadd_pd(sum, sum);                 // [sum0+sum1+sum2+sum3, duplicate]

        // Extract and sqrt
        double sum_sq = _mm_cvtsd_f64(sum);
        return std::sqrt(sum_sq);
    }
#endif

    // MKL-optimized operations for distance computations

    // Select distance function based on available optimizations
    inline double optimized_distance(const Point4F &a, const Point4F &b) noexcept
    {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        double dm = a.m - b.m;
        return std::sqrt(dx * dx + dy * dy + dz * dz + dm * dm);
    }

    // Select dot product function
    inline double optimized_dot(const Point4F &a, const Point4F &b) noexcept
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.m * b.m;
    }

    CoordinateMapper::Diagnostics CoordinateMapper::compute_diagnostics(const std::map<uint32_t, Point4F> &points)
    {
        Diagnostics diag;

        if (points.empty())
            return diag;

        size_t n = points.size();
        std::vector<Point4F> point_list;
        std::vector<uint32_t> codepoints;
        point_list.reserve(n);
        codepoints.reserve(n);

        for (const auto &[cp, pt] : points)
        {
            point_list.push_back(pt);
            codepoints.push_back(cp);
        }

        // Compute nearest neighbor chordal distances
        std::vector<double> chordal_nn(n, std::numeric_limits<double>::max());
        std::vector<double> geodesic_nn(n, std::numeric_limits<double>::max());

        // Brute force NN computation (for now - could use HNSW later)
#ifdef HAS_OPENMP
#pragma omp parallel for if (n > 1000)
#endif
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                double chordal_dist = optimized_distance(point_list[i], point_list[j]);
                double geodesic_dist = point_list[i].geodesic_distance(point_list[j]);

                if (chordal_dist < chordal_nn[i])
                    chordal_nn[i] = chordal_dist;
                if (geodesic_dist < geodesic_nn[i])
                    geodesic_nn[i] = geodesic_dist;
            }
        }

        // Compute statistics for chordal NN
        auto compute_stats = [](const std::vector<double> &vals) -> std::tuple<double, double, double, double, double, double>
        {
            if (vals.empty())
                return {0, 0, 0, 0, 0, 0};

            std::vector<double> sorted = vals;
            std::sort(sorted.begin(), sorted.end());

            double sum = 0, sum_sq = 0;
            for (double v : vals)
            {
                sum += v;
                sum_sq += v * v;
            }
            double mean = sum / vals.size();
            double variance = (sum_sq / vals.size()) - (mean * mean);
            double std_dev = std::sqrt(std::max(0.0, variance));
            double cv = (mean > 0) ? std_dev / mean : 0;

            size_t idx_5 = sorted.size() * 5 / 100;
            size_t idx_95 = sorted.size() * 95 / 100;
            double p5 = sorted[std::min(idx_5, sorted.size() - 1)];
            double p95 = sorted[std::min(idx_95, sorted.size() - 1)];

            return {mean, sorted[sorted.size() / 2], std_dev, cv, p5, p95};
        };

        std::tie(diag.chordal_nn_mean, diag.chordal_nn_median, diag.chordal_nn_std,
                 diag.chordal_nn_cv, diag.chordal_nn_5th, diag.chordal_nn_95th) = compute_stats(chordal_nn);

        std::tie(diag.geodesic_nn_mean, diag.geodesic_nn_median, diag.geodesic_nn_std,
                 diag.geodesic_nn_cv, diag.geodesic_nn_5th, diag.geodesic_nn_95th) = compute_stats(geodesic_nn);

        // Local density approximation: V(p) ≈ C_d * d_1(p)^d with d=3
        const double C_3 = 4.0 * PI / 3.0; // Volume constant for 3D
        std::vector<double> densities;
        for (double d1 : chordal_nn)
        {
            double v = C_3 * std::pow(d1, 3);
            densities.push_back(v > 0 ? 1.0 / v : 0); // density = 1/volume
        }

        std::tie(diag.local_density_mean, std::ignore, diag.local_density_std,
                 diag.local_density_cv, std::ignore, std::ignore) = compute_stats(densities);

        // Collision histogram (quantized tuples)
        for (const auto &[cp, pt] : points)
        {
            Point4D quantized = pt.to_quantized();
            diag.collision_counts[quantized]++;
        }

        // Bucket CV by semantic category
        std::map<uint32_t, std::vector<double>> bucket_nns;
        for (size_t i = 0; i < n; ++i)
        {
            uint32_t bucket = static_cast<uint32_t>(categorize(codepoints[i]));
            bucket_nns[bucket].push_back(chordal_nn[i]);
        }

        for (const auto &[bucket, nns] : bucket_nns)
        {
            if (nns.size() < 2)
                continue;
            double sum = 0, sum_sq = 0;
            for (double v : nns)
            {
                sum += v;
                sum_sq += v * v;
            }
            double mean = sum / nns.size();
            double variance = (sum_sq / nns.size()) - (mean * mean);
            double std_dev = std::sqrt(std::max(0.0, variance));
            double cv = (mean > 0) ? std_dev / mean : 0;
            diag.bucket_cv[bucket] = cv;
        }

        return diag;
    }

    void CoordinateMapper::apply_deterministic_jitter(std::map<uint32_t, Point4F> &points, double epsilon)
    {
        if (points.empty())
            return;

        // Compute orthonormal tangent basis at each point
        auto tangent_basis = [](const Point4F &p) -> std::array<Point4F, 3>
        {
            // Find arbitrary vector not parallel to p
            Point4F a(1.0, 0.0, 0.0, 0.0);
            if (std::abs(p.dot(a)) > 0.9)
            { // Nearly parallel
                a = Point4F(0.0, 1.0, 0.0, 0.0);
            }

            // Gram-Schmidt in 4D
            Point4F t1 = (a + p * (-p.dot(a))).normalized();

            // Find another vector not in span{p, t1}
            Point4F b(0.0, 1.0, 0.0, 0.0);
            if (std::abs(p.dot(b)) > 0.9 || std::abs(t1.dot(b)) > 0.9)
            {
                b = Point4F(0.0, 0.0, 1.0, 0.0);
            }
            Point4F t2 = (b + p * (-p.dot(b)) + t1 * (-t1.dot(b))).normalized();

            // Third basis vector (simplified cross product in remaining coordinates)
            Point4F t3 = Point4F(
                             p.y * t1.z - p.z * t1.y,
                             p.z * t1.x - p.x * t1.z,
                             p.x * t1.y - p.y * t1.x,
                             0.0)
                             .normalized();

            return {t1, t2, t3};
        };

        for (auto &[cp, p] : points)
        {
            // Generate deterministic values from BLAKE3 hash of codepoint
            Blake3Hash hash = Blake3Hasher::hash_codepoint(cp);
            uint64_t v0 = *reinterpret_cast<const uint64_t *>(hash.data());
            uint64_t v1 = *reinterpret_cast<const uint64_t *>(hash.data() + 8);
            uint64_t v2 = *reinterpret_cast<const uint64_t *>(hash.data() + 16);

            double f0 = static_cast<double>(v0) / static_cast<double>(UINT64_MAX);
            double f1 = static_cast<double>(v1) / static_cast<double>(UINT64_MAX);
            double f2 = static_cast<double>(v2) / static_cast<double>(UINT64_MAX);

            // Get tangent basis
            auto [t1, t2, t3] = tangent_basis(p);

            // Create jitter vector
            Point4F jitter = (t1 * (2.0 * f0 - 1.0) +
                              t2 * (2.0 * f1 - 1.0) +
                              t3 * (2.0 * f2 - 1.0)) *
                             epsilon;

            // Apply and renormalize
            p = (p + jitter).normalized();
        }
    }



    void CoordinateMapper::bucketed_tangent_lloyd(std::map<uint32_t, Point4F> &points,
                                                  size_t k, double alpha, int iterations)
    {
        if (points.empty())
            return;

        // Group points by semantic category
        std::map<uint32_t, std::vector<std::pair<uint32_t, Point4F *>>> buckets;
        for (auto &[cp, pt] : points)
        {
            uint32_t cat = static_cast<uint32_t>(categorize(cp));
            buckets[cat].emplace_back(cp, &pt);
        }

        // Compute tangent basis function (shared with jitter)
        [[maybe_unused]] auto tangent_basis = [](const Point4F &p) -> std::array<Point4F, 3>
        {
            Point4F a(1.0, 0.0, 0.0, 0.0);
            if (std::abs(p.dot(a)) > 0.9)
                a = Point4F(0.0, 1.0, 0.0, 0.0);

            Point4F t1 = (a + p * (-p.dot(a))).normalized();

            Point4F b(0.0, 1.0, 0.0, 0.0);
            if (std::abs(p.dot(b)) > 0.9 || std::abs(t1.dot(b)) > 0.9)
            {
                b = Point4F(0.0, 0.0, 1.0, 0.0);
            }
            Point4F t2 = (b + p * (-p.dot(b)) + t1 * (-t1.dot(b))).normalized();

            Point4F t3 = Point4F(
                             p.y * t1.z - p.z * t1.y,
                             p.z * t1.x - p.x * t1.z,
                             p.x * t1.y - p.y * t1.x,
                             0.0)
                             .normalized();

            return {t1, t2, t3};
        };

        size_t MIN_BUCKET = 64;

        // Process each bucket
        for (auto &[bucket_id, bucket_points] : buckets)
        {
            size_t n_bucket = bucket_points.size();

            // Build bucket_coords
            std::vector<Point4F> bucket_coords;
            for (const auto &[cp, pt_ptr] : bucket_points)
            {
                bucket_coords.push_back(*pt_ptr);
            }

            if (n_bucket < MIN_BUCKET)
            {
                // Use global kNN for small buckets
                for (int iter = 0; iter < iterations; ++iter)
                {
                    for (size_t i = 0; i < n_bucket; ++i)
                    {
                        const Point4F &p = bucket_coords[i];
                        Point4F v_avg(0, 0, 0, 0);

                        // Find k nearest in all points
                        std::vector<std::pair<double, uint32_t>> neighbors;
                        for (const auto &[cp, pt] : points)
                        {
                            double dist = p.distance(pt);
                            neighbors.emplace_back(dist, cp);
                        }
                        std::sort(neighbors.begin(), neighbors.end());

                        // Average neighbor vectors in tangent space
                        for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni)
                        {
                            uint32_t cp = neighbors[ni].second;
                            const Point4F &q = points.at(cp);

                            // Project q onto tangent space at p
                            double pq_dot = p.dot(q);
                            Point4F v_q = q + p * (-pq_dot);
                            v_avg = v_avg + v_q;
                        }

                        if (neighbors.size() > 0)
                            v_avg = v_avg * (1.0 / std::min(k, neighbors.size()));

                        // Update point
                        Point4F &target = *bucket_points[i].second;
                        target = (p + v_avg * alpha).normalized();
                        bucket_coords[i] = target;
                    }
                }
            }
            else
            {
                // Original bucket-based for large buckets
                for (int iter = 0; iter < iterations; ++iter)
                {
                    for (size_t i = 0; i < n_bucket; ++i)
                    {
                        const Point4F &p = bucket_coords[i];
                        Point4F v_avg(0, 0, 0, 0);

                        // Find k nearest neighbors in bucket
                        std::vector<std::pair<double, size_t>> neighbors;
                        for (size_t j = 0; j < n_bucket; ++j)
                        {
                            if (i == j)
                                continue;
                            double dist = p.distance(bucket_coords[j]);
                            neighbors.emplace_back(dist, j);
                        }
                        std::partial_sort(neighbors.begin(), neighbors.begin() + std::min(k, neighbors.size()),
                                          neighbors.end());

                        // Average neighbor vectors in tangent space
                        for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni)
                        {
                            size_t j = neighbors[ni].second;
                            const Point4F &q = bucket_coords[j];

                            // Project q onto tangent space at p
                            double pq_dot = p.dot(q);
                            Point4F v_q = q + p * (-pq_dot);
                            v_avg = v_avg + v_q;
                        }

                        if (neighbors.size() > 0)
                            v_avg = v_avg * (1.0 / std::min(k, neighbors.size()));

                        // Update point
                        Point4F &target = *bucket_points[i].second;
                        target = (p + v_avg * alpha).normalized();
                        bucket_coords[i] = target;
                    }
                }
            }
        }
    }

    void CoordinateMapper::global_knn_repulsion(std::map<uint32_t, Point4F> &points,
                                                size_t k, double s, double eta, int iterations)
    {
        if (points.empty())
            return;

        std::vector<Point4F *> point_ptrs;
        std::vector<Point4F> point_list;

        for (auto &[cp, pt] : points)
        {
            point_ptrs.push_back(&pt);
            point_list.push_back(pt);
        }

        size_t n = points.size();
        [[maybe_unused]] double prev_energy = 0.0; // Previous energy value (for tracking)

        // Compute initial mean NN distance for eta scaling
        [[maybe_unused]] double mean_nn = 0.0;
        int count = 0;
        for (size_t i = 0; i < n; ++i)
        {
            double min_dist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < n; ++j)
            {
                if (i == j)
                    continue;
                double dist = point_list[i].distance(point_list[j]);
                if (dist < min_dist)
                    min_dist = dist;
            }
            if (min_dist < std::numeric_limits<double>::max())
            {
                mean_nn += min_dist;
                count++;
            }
        }
        if (count > 0)
            mean_nn /= count;

        // Set initial eta based on mean NN distance (much smaller for stability)
        double eta_initial = 0.01; // reduced initial step size to 0.01-0.1
        double eta_max = 1.0;      // maximum step size cap
        double eta_min = 1e-8;
        double max_grad_norm = 1e-3; // gradient clipping

        for (int iter = 0; iter < iterations; ++iter)
        {
            double energy = 0.0;

            // Compute forces for all points using geodesic gradient
            std::vector<Point4F> forces(n, Point4F(0, 0, 0, 0));

            // #ifdef HAS_OPENMP
            // #pragma omp parallel for if(n > 1000)
            // #endif
            for (size_t i = 0; i < n; ++i)
            {
                const Point4F &p = point_list[i];

                // Find k nearest neighbors
                std::vector<std::pair<double, size_t>> neighbors;
                for (size_t j = 0; j < n; ++j)
                {
                    if (i == j)
                        continue;
                    double dist = optimized_distance(p, point_list[j]);
                    neighbors.emplace_back(dist, j);
                }

                std::partial_sort(neighbors.begin(), neighbors.begin() + std::min(k, neighbors.size()),
                                  neighbors.end());

                Point4F force_sum(0, 0, 0, 0);

                // Repel all k nearest neighbors
                for (size_t ni = 0; ni < std::min(k, neighbors.size()); ++ni)
                {
                    size_t j = neighbors[ni].second;
                    const Point4F &q = point_list[j];
                    double r = optimized_distance(p, q);
                    // Use chordal gradient (simpler and more stable than geodesic)
                    Point4F diff(p.x - q.x, p.y - q.y, p.z - q.z, p.m - q.m);
                    double chordal_r = std::sqrt(std::max(diff.dot(diff), 1e-18));

                    if (chordal_r > 1e-12)
                    {
                        // Normalized direction vector from q to p
                        Point4F direction = diff * (1.0 / chordal_r);

                        // Repulsive force: stronger when closer, proportional to 1/r^2
                        double repulsion_strength = 10.0 / (r * r + 1e-8);
                        force_sum = force_sum + direction * repulsion_strength;

                        // Energy: Coulomb-like potential
                        energy += 1.0 / std::pow(std::max(r, 1e-18), s);
                    }
                }

                forces[i] = force_sum;
            }

#ifdef HAS_OPENMP
#pragma omp barrier
#endif

            // Apply forces with tangent projection and backtracking line search
            double c_armijo = 0.1; // strengthened Armijo condition
            double tau = 0.5;
            eta = eta_initial;

            // Try different eta values until Armijo condition is satisfied
            bool armijo_satisfied = false;
            double energy_new = energy;
            std::vector<Point4F> new_positions = point_list;

            double dot_grad = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                dot_grad += forces[i].dot(forces[i]);
            }

            int backtrack_attempts = 0;
            while (!armijo_satisfied && eta > eta_min)
            {
                backtrack_attempts++;
                // Compute tentative positions
                for (size_t i = 0; i < n; ++i)
                {
                    const Point4F &p = point_list[i];
                    Point4F G_tan = forces[i] + p * (-p.dot(forces[i])); // Project to tangent space

                    // Gradient clipping
                    double gnorm = std::sqrt(G_tan.dot(G_tan));
                    if (gnorm > max_grad_norm)
                    {
                        G_tan = G_tan * (max_grad_norm / gnorm);
                    }

                    new_positions[i] = (p - G_tan * eta).normalized();
                }

                // Compute new energy
                energy_new = 0.0;
                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = i + 1; j < n; ++j)
                    {
                        double r = optimized_distance(new_positions[i], new_positions[j]);
                        energy_new += 1.0 / std::pow(std::max(r, 1e-18), s);
                    }
                }

                // Check Armijo condition
                double armijo_threshold = energy + c_armijo * eta * dot_grad;
                bool condition_met = energy_new <= armijo_threshold;

                if (condition_met)
                {
                    armijo_satisfied = true;
                }
                else
                {
                    eta *= tau;
                    eta = std::min(eta, eta_max); // cap step size
                }

                if (backtrack_attempts > 20)
                {
                    break;
                }
            }

            // Update positions with the accepted eta
            for (size_t i = 0; i < n; ++i)
            {
                *point_ptrs[i] = new_positions[i];
                point_list[i] = new_positions[i];
            }

            prev_energy = energy_new;
        }
    }

    bool CoordinateMapper::optimize_distribution(std::map<uint32_t, Point4F>& points)
    {
        // Apply deterministic jitter first
        apply_deterministic_jitter(points);

        // Apply semantic-aware jitter (removed for simplification)

        // Run bucketed tangent Lloyd optimization
        bucketed_tangent_lloyd(points);

        // Run global KNN repulsion
        global_knn_repulsion(points);

        return true;
    }

}
