#pragma once

#include "hypercube/types.hpp"
#include "hypercube/hilbert.hpp"
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>

namespace hypercube {

/**
 * Types of text compositions for centroid calculation
 */
enum class CompositionType {
    ATOM,           // Single codepoint
    WORD,           // Lexical unit
    PHRASE,         // Syntactic phrase
    SENTENCE,       // Complete sentence
    MULTIMODAL      // Mixed text/emoji/symbols
};

/**
 * Combined result of codepoint mapping - coords AND hilbert index
 * Avoids redundant Hilbert encode/decode roundtrip
 */
struct CodepointMapping {
    Point4F coords_float;  // Canonical floating-point coordinates on S³
    Point4D coords;        // Quantized coordinates for indexing
    HilbertIndex hilbert;
};

/**
 * Hopf Fibration Coordinate Mapper for Unicode Atoms
 *
 * Maps all Unicode codepoints onto the surface of a 3-sphere (hypersphere in 4D)
 * using Hopf fibration with golden angle spiral for truly equidistant distribution.
 *
 * Key properties:
 * - ALL atoms are evenly distributed on the 3-sphere surface using Hopf coordinates
 * - Golden angle spiral ensures minimal distance variation between points (std dev < 1)
 * - Semantically related codepoints (A, a, Ä) are placed nearby via sequential ordering
 * - 32 bits per dimension = lossless, collision-free coordinates
 * - Hilbert index is derived from coords for spatial indexing
 * - Compositions (n-grams, words, etc.) have centroids INSIDE the sphere
 */
class CoordinateMapper {
public:
    /**
     * Map a Unicode codepoint to its 4D coordinates on the 3-sphere surface
     * Uses Fibonacci lattice for even distribution with semantic clustering
     * @param codepoint Unicode codepoint (0 to 0x10FFFF)
     * @return 4D point on 3-sphere surface (32 bits per dimension)
     */
    static Point4D map_codepoint(uint32_t codepoint) noexcept;
    
    /**
     * Map a Unicode codepoint to BOTH coords AND hilbert index
     * This is the efficient version - computes hilbert once during mapping
     * @param codepoint Unicode codepoint (0 to 0x10FFFF)
     * @return Combined coords + hilbert index
     */
    static CodepointMapping map_codepoint_full(uint32_t codepoint) noexcept;

    /**
     * Map a Unicode codepoint to floating-point coordinates on the 3-sphere
     * Useful for optimization and diagnostics before quantization
     * @param codepoint Unicode codepoint (0 to 0x10FFFF)
     * @return Point4F on unit 3-sphere surface
     */
    static Point4F map_codepoint_float(uint32_t codepoint) noexcept;
    
    /**
     * Determine category of a Unicode codepoint
     * Uses Unicode general categories for classification
     * @param codepoint Unicode codepoint
     * @return Category enum value
     */
    static AtomCategory categorize(uint32_t codepoint) noexcept;
    
    /**
     * Calculate centroid of multiple points
     * Used for composition coordinates - will be INSIDE the sphere
     * @param points Vector of 4D points
     * @return Centroid point (interior for surface points)
     */
    static Point4D centroid(const std::vector<Point4D>& points) noexcept;

    /**
     * Calculate centroid of multiple float points (canonical for geometry)
     * @param points Vector of 4F points
     * @return Centroid point (normalized to S³ surface)
     */
    static Point4F centroid_float(const std::vector<Point4F>& points) noexcept;
    
    /**
     * Calculate weighted centroid
     * @param points Vector of 4D points
     * @param weights Weights for each point (must be same size as points)
     * @return Weighted centroid
     */
    static Point4D weighted_centroid(const std::vector<Point4D>& points,
                                      const std::vector<double>& weights) noexcept;
    

    
    /**
     * Calculate Euclidean distance in 4D space
     */
    static double euclidean_distance(const Point4D& a, const Point4D& b) noexcept;
    
    /**
     * Get the count of codepoints in a category
     */
    static uint32_t get_category_count(AtomCategory cat) noexcept;

    // ============================================================================
    // OPTIMIZATION PIPELINE FUNCTIONS
    // ============================================================================

    /**
     * Compute diagnostics for a set of points
     */
    struct Diagnostics {
        double chordal_nn_mean;
        double chordal_nn_median;
        double chordal_nn_std;
        double chordal_nn_cv;
        double chordal_nn_5th;
        double chordal_nn_95th;

        double geodesic_nn_mean;
        double geodesic_nn_median;
        double geodesic_nn_std;
        double geodesic_nn_cv;
        double geodesic_nn_5th;
        double geodesic_nn_95th;

        double local_density_mean;
        double local_density_std;
        double local_density_cv;

        std::map<Point4D, size_t> collision_counts;
        std::map<uint32_t, double> bucket_cv;

        Diagnostics() : chordal_nn_mean(0), chordal_nn_median(0), chordal_nn_std(0), chordal_nn_cv(0),
                       chordal_nn_5th(0), chordal_nn_95th(0),
                       geodesic_nn_mean(0), geodesic_nn_median(0), geodesic_nn_std(0), geodesic_nn_cv(0),
                       geodesic_nn_5th(0), geodesic_nn_95th(0),
                       local_density_mean(0), local_density_std(0), local_density_cv(0) {}
    };

    /**
     * Compute comprehensive diagnostics for a mapping of codepoints to points
     * @param points Map from codepoint to Point4F (floating point coordinates)
     * @return Diagnostics struct with all metrics
     */
    static Diagnostics compute_diagnostics(const std::map<uint32_t, Point4F>& points);

    /**
     * Apply deterministic jitter to break quantization collisions
     * @param points Input/output map of codepoints to points
     * @param epsilon Jitter magnitude (default 1e-7)
     */
    static void apply_deterministic_jitter(std::map<uint32_t, Point4F>& points,
                                           double epsilon = 1e-7);

    /**
     * Perform bucketed tangent Lloyd relaxation
     * @param points Input/output map of codepoints to points
     * @param k Number of neighbors per iteration (default 32)
     * @param alpha Step size (default 0.25)
     * @param iterations Number of iterations (default 4)
     */
    static void bucketed_tangent_lloyd(std::map<uint32_t, Point4F>& points,
                                       size_t k = 32, double alpha = 0.25, int iterations = 4);

    /**
     * Perform global KNN repulsion
     * @param points Input/output map of codepoints to points
     * @param k Number of neighbors for repulsion (default 64)
     * @param s Repulsion exponent (default 1.0 for Coulomb-like)
     * @param eta Initial step size (default 0.001)
     * @param iterations Number of iterations (default 10)
     */
    static void global_knn_repulsion(std::map<uint32_t, Point4F>& points,
                                     size_t k = 64, double s = 1.0,
                                     double eta = 0.001, int iterations = 10);

    /**
     * Run the complete optimization pipeline
     * @param points Input/output map of codepoints to points
     * @return True if successful
     */
    static bool optimize_distribution(std::map<uint32_t, Point4F>& points);

    /**
     * Apply semantic-aware jitter that preserves case variant proximity
     */
    static void apply_semantic_aware_jitter(std::map<uint32_t, Point4F>& points,
                                            double epsilon = 1e-7);

    /**
     * Perform geodesic repulsion optimization using Riemannian gradients
     */
    static void geodesic_repulsion_optimization(std::map<uint32_t, Point4F>& points,
                                               size_t k = 32, double eta = 0.001, int iterations = 8);

    /**
     * Perform adaptive tangent space optimization with convergence criteria
     */
    static void adaptive_tangent_optimization(std::map<uint32_t, Point4F>& points,
                                            size_t k = 48, double eta = 0.01, int iterations = 6);

    /**
     * Perform convergence-driven optimization targeting specific CV
     */
    static void convergence_driven_optimization(std::map<uint32_t, Point4F>& points,
                                               double target_cv = 0.30);

    /**
     * Calculate composition centroid with interior positioning
     * @param atoms Vector of atom coordinates
     * @param type Type of composition
     * @return Centroid point (interior for complex compositions)
     */
    static Point4F compute_composition_centroid(const std::vector<Point4F>& atoms,
                                               CompositionType type) noexcept;

    /**
     * Compute complexity factor for composition positioning
     * @param atoms Vector of atom coordinates
     * @param type Type of composition
     * @return Complexity factor (0.0 to 3.0)
     */
    static double compute_complexity_factor(const std::vector<Point4F>& atoms,
                                           CompositionType type) noexcept;

    /**
     * Estimate codepoint from coordinates (reverse lookup approximation)
     * @param coords Point coordinates
     * @return Estimated codepoint
     */
    static uint32_t estimate_codepoint_from_coords(const Point4F& coords) noexcept;
};

/**
 * Generate 64-bit semantic key for dense ranking
 * Bit-packed by semantic gravity: Script (63-56), Category (55-48), Base (47-16), Variant (15-0)
 */
uint64_t get_semantic_key(uint32_t cp) noexcept;

/**
 * Dense registry for mapping codepoints to dense ranks
 */
class DenseRegistry {
private:
    static std::unordered_map<uint32_t, uint32_t> codepoint_to_rank;
    static std::vector<uint32_t> rank_to_codepoint;
    static bool initialized;
    static std::mutex init_mutex;

    static void initialize();

public:
    static uint32_t get_rank(uint32_t cp);
    static uint32_t total_active();
    static uint32_t get_codepoint(uint32_t rank);
};

} // namespace hypercube
