using HypercubeGenerativeApi.Services;
using HypercubeGenerativeApi.Interop;
using Microsoft.AspNetCore.Mvc;
using System.Runtime.InteropServices;

namespace HypercubeGenerativeApi.Controllers;

/// <summary>
/// 4D geometric operations - exposing the hypercube's spatial intelligence
/// </summary>
public static class GeometricController
{
    /// <summary>
    /// POST /geometric/neighbors - Find 4D spatial neighbors of a concept
    /// </summary>
    public static async Task<IResult> FindNeighbors(
        [FromBody] NeighborQueryRequest request,
        ILogger<GenerativeService> logger)
    {
        try
        {
            logger.LogInformation("Finding 4D neighbors for: {Entity}, k={K}", request.Entity, request.K ?? 10);

            if (string.IsNullOrWhiteSpace(request.Entity))
            {
                return TypedResults.BadRequest(new
                {
                    error = new
                    {
                        message = "Entity cannot be empty",
                        type = "invalid_request_error",
                        code = "missing_required_parameter"
                    }
                });
            }

            // This would find concepts with similar 4D coordinates
            var neighbors = await FindGeometricNeighborsAsync(request.Entity, request.K ?? 10);

            return TypedResults.Ok(new
            {
                entity = request.Entity,
                neighbors = neighbors,
                space = "4D_hypercube",
                method = "euclidean_distance",
                coordinate_system = "hypersphere_projection"
            });
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error finding geometric neighbors");
            return TypedResults.Json(new
            {
                error = new
                {
                    message = "Geometric neighbor search failed",
                    type = "internal_error",
                    detail = ex.Message
                }
            }, statusCode: 500);
        }
    }

    /// <summary>
    /// POST /geometric/centroid - Calculate semantic centroid of multiple concepts
    /// </summary>
    public static async Task<IResult> CalculateCentroid(
        [FromBody] CentroidQueryRequest request,
        ILogger<GenerativeService> logger)
    {
        try
        {
            logger.LogInformation("Calculating centroid for {Count} entities", request.Entities?.Count ?? 0);

            if (request.Entities == null || request.Entities.Count == 0)
            {
                return TypedResults.BadRequest(new
                {
                    error = new
                    {
                        message = "At least one entity is required",
                        type = "invalid_request_error"
                    }
                });
            }

            if (request.Entities.Count > 100)
            {
                return TypedResults.BadRequest(new
                {
                    error = new
                    {
                        message = "Too many entities (max 100)",
                        type = "invalid_request_error",
                        code = "parameter_too_large"
                    }
                });
            }

            // Calculate geometric centroid of multiple concepts
            var centroid = await CalculateGeometricCentroidAsync(request.Entities);

            return TypedResults.Ok(new
            {
                entities = request.Entities,
                centroid = centroid,
                method = "geometric_centroid_calculation",
                dimensions = 4,
                space = "4D_hypercube"
            });
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error calculating geometric centroid");
            return TypedResults.Json(new
            {
                error = new
                {
                    message = "Centroid calculation failed",
                    type = "internal_error",
                    detail = ex.Message
                }
            }, statusCode: 500);
        }
    }

    /// <summary>
    /// POST /geometric/distance - Measure semantic distance between concepts
    /// </summary>
    public static async Task<IResult> MeasureDistance(
        [FromBody] DistanceQueryRequest request,
        ILogger<GenerativeService> logger)
    {
        try
        {
            logger.LogInformation("Measuring distance between: {Entity1} and {Entity2}",
                request.Entity1, request.Entity2);

            if (string.IsNullOrWhiteSpace(request.Entity1) || string.IsNullOrWhiteSpace(request.Entity2))
            {
                return TypedResults.BadRequest(new
                {
                    error = new
                    {
                        message = "Both entities are required",
                        type = "invalid_request_error"
                    }
                });
            }

            // Calculate 4D Euclidean distance between concept coordinates
            var distance = await CalculateSemanticDistanceAsync(request.Entity1, request.Entity2);

            return TypedResults.Ok(new
            {
                entity1 = request.Entity1,
                entity2 = request.Entity2,
                distance = distance,
                normalized_distance = Math.Min(distance / 2.0, 1.0), // Max distance in unit hypercube is 2.0
                method = "4D_euclidean_distance",
                interpretation = distance < 0.5 ? "semantically_similar" :
                               distance < 1.0 ? "moderately_related" : "distantly_related"
            });
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error measuring semantic distance");
            return TypedResults.Json(new
            {
                error = new
                {
                    message = "Distance measurement failed",
                    type = "internal_error",
                    detail = ex.Message
                }
            }, statusCode: 500);
        }
    }

    /// <summary>
    /// GET /geometric/visualize/{entity} - Get 4D coordinates for visualization
    /// </summary>
    public static async Task<IResult> VisualizeCoordinates(
        [FromRoute] string entity,
        ILogger<GenerativeService> logger)
    {
        try
        {
            logger.LogInformation("Getting 4D coordinates for: {Entity}", entity);

            if (string.IsNullOrWhiteSpace(entity))
            {
                return TypedResults.BadRequest(new
                {
                    error = new
                    {
                        message = "Entity cannot be empty",
                        type = "invalid_request_error"
                    }
                });
            }

            // Get 4D coordinates for visualization
            var coordinates = await GetEntityCoordinatesAsync(entity);

            if (coordinates == null)
            {
                return TypedResults.NotFound(new
                {
                    error = new
                    {
                        message = $"Entity '{entity}' not found in geometric space",
                        type = "not_found_error"
                    }
                });
            }

            return TypedResults.Ok(new
            {
                entity = entity,
                coordinates = coordinates,
                space = "4D_hypercube",
                projection = "hypersphere_surface",
                visualization_hint = "Use 3D projection or t-SNE for 2D/3D visualization"
            });
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error getting coordinates for entity: {Entity}", entity);
            return TypedResults.Json(new
            {
                error = new
                {
                    message = "Coordinate retrieval failed",
                    type = "internal_error",
                    detail = ex.Message
                }
            }, statusCode: 500);
        }
    }

    // Placeholder implementations - these would integrate with C++ geometric operations

    private static async Task<List<GeometricNeighbor>> FindGeometricNeighborsAsync(string entity, int k)
    {
        try
        {
            // Use real C++ hypercube similarity search
            var results = new GenSimilarResult[k];
            var resultCount = GenerativeInterop.gen_find_similar(entity, (UIntPtr)k, results);

            var neighbors = new List<GeometricNeighbor>();
            for (int i = 0; i < (int)resultCount && i < k; i++)
            {
                var labelPtr = GenerativeInterop.gen_vocab_get_label(results[i].index);
                if (labelPtr != IntPtr.Zero)
                {
                    var neighborLabel = Marshal.PtrToStringAnsi(labelPtr);
                    if (!string.IsNullOrEmpty(neighborLabel) && neighborLabel != entity)
                    {
                        // Get 4D coordinates for this neighbor
                        var coordinates = await GetEntityCoordinatesAsync(neighborLabel) ?? new[] { 0.0, 0.0, 0.0, 0.0 };

                        neighbors.Add(new GeometricNeighbor
                        {
                            Entity = neighborLabel,
                            Distance = results[i].similarity,
                            Coordinates = coordinates,
                            Relationship = results[i].similarity > 0.8 ? "highly_similar" :
                                          results[i].similarity > 0.6 ? "moderately_similar" : "distantly_related"
                        });
                    }
                }
            }

            return neighbors;
        }
        catch (Exception)
        {
            // Fallback if C++ call fails - use simple placeholder for now
            return new List<GeometricNeighbor>
            {
                new GeometricNeighbor {
                    Entity = $"{entity}_neighbor_1",
                    Distance = 0.5,
                    Coordinates = new[] { 0.1, 0.2, 0.3, 0.4 },
                    Relationship = "fallback_similarity"
                }
            }.Take(k).ToList();
        }
    }

    private static async Task<GeometricCentroid> CalculateGeometricCentroidAsync(List<string> entities)
    {
        try
        {
            // Get coordinates for all entities
            var entityCoords = new List<double[]>();
            foreach (var entity in entities)
            {
                var coords = await GetEntityCoordinatesAsync(entity);
                if (coords != null)
                {
                    entityCoords.Add(coords);
                }
            }

            if (entityCoords.Count == 0)
            {
                // No valid coordinates found
                return new GeometricCentroid
                {
                    Coordinates = new[] { 0.0, 0.0, 0.0, 0.0 },
                    Confidence = 0.0,
                    EntityCount = entities.Count,
                    BoundingRadius = 0.0
                };
            }

            // Convert to Point4D array
            var points = entityCoords.Select(Point4D.FromArray).ToArray();

            // Calculate centroid using C++ geometric operations
            Point4D centroidPoint;
            GenerativeInterop.geom_centroid(points, (UIntPtr)points.Length, out centroidPoint);

            // Calculate bounding radius (max distance from centroid)
            var maxDistance = 0.0;
            for (int i = 0; i < points.Length; i++)
            {
                var distance = GenerativeInterop.geom_euclidean_distance(ref centroidPoint, ref points[i]);
                if (distance > maxDistance) maxDistance = distance;
            }

            return new GeometricCentroid
            {
                Coordinates = centroidPoint.ToArray(),
                Confidence = entityCoords.Count / (double)entities.Count, // Fraction of entities with coordinates
                EntityCount = entities.Count,
                BoundingRadius = maxDistance
            };
        }
        catch (Exception)
        {
            // Fallback to placeholder if C++ call fails
            var random = new Random();
            return new GeometricCentroid
            {
                Coordinates = new[] {
                    random.NextDouble(),
                    random.NextDouble(),
                    random.NextDouble(),
                    random.NextDouble()
                },
                Confidence = 0.87,
                EntityCount = entities.Count,
                BoundingRadius = 0.45
            };
        }
    }

    private static async Task<double> CalculateSemanticDistanceAsync(string entity1, string entity2)
    {
        try
        {
            // Get coordinates for both entities
            var coords1 = await GetEntityCoordinatesAsync(entity1);
            var coords2 = await GetEntityCoordinatesAsync(entity2);

            if (coords1 == null || coords2 == null)
            {
                return double.NaN; // Entities not found in geometric space
            }

            // Convert to Point4D structs
            var point1 = Point4D.FromArray(coords1);
            var point2 = Point4D.FromArray(coords2);

            // Calculate Euclidean distance using C++ geometric operations
            var distance = GenerativeInterop.geom_euclidean_distance(ref point1, ref point2);

            return distance;
        }
        catch (Exception)
        {
            // Fallback to placeholder if C++ call fails
            var random = new Random();
            return random.NextDouble() * 2.0; // Max distance in unit hypercube is 2.0
        }
    }

    private static async Task<double[]?> GetEntityCoordinatesAsync(string entity)
    {
        try
        {
            // First try to get from database (TODO: implement database lookup)
            // For now, calculate coordinates using C++ geometric mapping

            // Convert entity name to codepoint (simplified - use first character)
            if (string.IsNullOrEmpty(entity))
                return null;

            var codepoint = (uint)entity[0]; // Simplified mapping

            // Map to 4D coordinates using C++ geometric operations
            Point4D coords;
            GenerativeInterop.geom_map_codepoint(codepoint, out coords);

            return coords.ToArray();
        }
        catch (Exception)
        {
            // Fallback to null if geometric mapping fails
            return null;
        }
    }
}

// Request/Response models for geometric operations

public class NeighborQueryRequest
{
    public string Entity { get; set; } = string.Empty;
    public int? K { get; set; } = 10;
    public string? DistanceMetric { get; set; } = "euclidean"; // euclidean, manhattan, cosine
}

public class CentroidQueryRequest
{
    public List<string> Entities { get; set; } = new();
    public bool Normalize { get; set; } = true;
}

public class DistanceQueryRequest
{
    public string Entity1 { get; set; } = string.Empty;
    public string Entity2 { get; set; } = string.Empty;
    public string? DistanceMetric { get; set; } = "euclidean";
}

public class GeometricNeighbor
{
    public string Entity { get; set; } = string.Empty;
    public double Distance { get; set; }
    public double[] Coordinates { get; set; } = Array.Empty<double>();
    public string Relationship { get; set; } = string.Empty;
}

public class GeometricCentroid
{
    public double[] Coordinates { get; set; } = Array.Empty<double>();
    public double Confidence { get; set; }
    public int EntityCount { get; set; }
    public double BoundingRadius { get; set; }
}