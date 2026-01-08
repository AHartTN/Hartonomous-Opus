using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace HypercubeGenerativeApi;

/// <summary>
/// Configures OpenAPI/Swagger documentation for the Hypercube Generative API
/// </summary>
public class SwaggerConfig
{
    /// <summary>
    /// Configure Swagger generation options
    /// </summary>
    public static void ConfigureSwagger(SwaggerGenOptions options)
    {
        options.SwaggerDoc("v1", new OpenApiInfo
        {
            Title = "Hypercube Generative API",
            Version = "v1",
            Description = @"
# Hartonomous-Opus: 4D Hypercube AI Substrate

**Revolutionary AI that understands concepts geometrically in 4D space**

This API provides both OpenAI-compatible endpoints and groundbreaking hypercube intelligence capabilities. Unlike traditional AI that treats language as flat token sequences, the hypercube approach maps all digital content to semantic units in 4D geometric space, enabling unprecedented understanding and query capabilities.

## Key Innovations

### 🎯 **4D Geometric Intelligence**
- Concepts exist as coordinates in 4D hypersphere space
- Semantic relationships are geometric distances
- Hilbert curve indexing enables O(log n) spatial queries
- Multi-modal content unified in single geometric substrate

### 🔍 **Universal Content Ingestion**
- Ingest any digital content: code, documents, images, audio, databases
- Automatic semantic unit extraction and 4D mapping
- Cross-modal semantic search and relationships
- Billion-scale vocabulary support with instant loading

### ⚡ **C++ Accelerated Operations**
- SIMD vector operations for geometric calculations
- Memory-mapped vocabularies for instant access
- Parallel geometric processing across CPU cores
- Real-time 4D spatial queries and transformations

### 🌐 **API Architecture**
- **OpenAI Compatibility**: Drop-in replacement for existing AI workflows
- **Hypercube Extensions**: Revolutionary geometric and semantic operations
- **Multi-Modal Queries**: Search across all ingested content types
- **Real-Time Performance**: C++ offloading for computational intensive tasks

## Quick Start

### OpenAI-Compatible Usage
```bash
curl -X POST http://localhost:5000/v1/completions \
  -H ""Content-Type: application/json"" \
  -d '{
    ""model"": ""hypercube-generative"",
    ""prompt"": ""Explain quantum computing"",
    ""max_tokens"": 100
  }'
```

### Hypercube-Specific Queries
```bash
# Find geometric neighbors in 4D space
curl -X POST http://localhost:5000/geometric/neighbors \
  -H ""Content-Type: application/json"" \
  -d '{
    ""entity"": ""machine learning"",
    ""k"": 10
  }'

# Measure semantic distance
curl -X POST http://localhost:5000/geometric/distance \
  -H ""Content-Type: application/json"" \
  -d '{
    ""entity1"": ""quantum physics"",
    ""entity2"": ""classical physics""
  }'

# Ingest any content
curl -X POST http://localhost:5000/ingest/document \
  -F ""file=@research_paper.pdf"" \
  -F ""title=Quantum Computing Advances""
```
",
            Contact = new OpenApiContact
            {
                Name = "Hartonomous-Opus Project",
                Url = new Uri("https://github.com/AHartTN/Hartonomous-Opus")
            },
            License = new OpenApiLicense
            {
                Name = "MIT License"
            }
        });

        // Group endpoints by functionality
        options.TagActionsBy(api => api.GroupName ?? api.ActionDescriptor.RouteValues["controller"]);

        options.DocInclusionPredicate((name, api) => true);

        // Add operation filters for better documentation
        options.OperationFilter<HypercubeOperationFilter>();
    }
}

/// <summary>
/// Custom operation filter to enhance endpoint documentation
/// </summary>
public class HypercubeOperationFilter : IOperationFilter
{
    public void Apply(OpenApiOperation operation, OperationFilterContext context)
    {
        var endpointMetadata = context.ApiDescription.ActionDescriptor.EndpointMetadata;

        // Add hypercube-specific documentation based on endpoint
        switch (operation.OperationId)
        {
            case "FindNeighbors":
                operation.Summary = "Find 4D geometric neighbors of a concept";
                operation.Description = @"
**Revolutionary geometric search beyond token similarity**

Unlike traditional AI that finds similar content through statistical co-occurrence, this endpoint locates concepts that are geometrically proximate in 4D semantic space. This reveals true semantic relationships that go beyond surface-level text matching.

**Use Cases:**
- Discover conceptually related ideas that traditional search misses
- Build semantic knowledge graphs
- Find cross-domain analogies
- Navigate concept spaces like physical landscapes

**Performance:** O(log n) spatial indexing via Hilbert curves
";
                break;

            case "MeasureDistance":
                operation.Summary = "Measure semantic distance between concepts";
                operation.Description = @"
**Quantify conceptual separation in 4D space**

Returns the normalized Euclidean distance between concept coordinates in 4D hypersphere space. This provides a precise, geometric measure of semantic relatedness that correlates with human understanding better than traditional similarity scores.

**Interpretation:**
- 0.0-0.3: Highly related concepts
- 0.3-0.7: Moderately related
- 0.7-1.0: Distantly related
- 1.0+: Unrelated conceptual domains

**Advantages over traditional AI:** Geometric precision vs statistical approximation
";
                break;

            case "CalculateCentroid":
                operation.Summary = "Calculate semantic centroid of multiple concepts";
                operation.Description = @"
**Find the geometric center of multiple concepts in 4D space**

Computes the centroid (average position) of multiple semantic units in 4D space. This reveals the 'essence' or 'average meaning' of a collection of concepts - the geometric heart of a set of related ideas.

**Applications:**
- Understand category prototypes (what is the 'average' programming language?)
- Find conceptual midpoints between domains
- Generate novel concept blends
- Analyze semantic drift over time

**Mathematics:** 4D vector averaging with geometric normalization
";
                break;

            case "QuerySemantic":
                operation.Summary = "Query semantic relationships across all ingested content";
                operation.Description = @"
**Universal semantic search across all content types and modalities**

Unlike traditional search that operates on text similarity, this performs geometric queries across the entire hypercube substrate. Find relevant information regardless of whether it came from code, documents, images, or other digital content.

**Capabilities:**
- Cross-modal semantic search
- Multi-document synthesis
- Concept relationship discovery
- Domain-agnostic knowledge retrieval

**Powered by:** Unified 4D geometric indexing of all digital content
";
                break;

            case "IngestDocument":
                operation.Summary = "Ingest any document into the hypercube substrate";
                operation.Description = @"
**Transform any digital document into geometric semantic units**

The ingestion process extracts semantic atoms from documents and maps them to 4D coordinates in the hypercube space. This enables geometric querying and relationships across all ingested content.

**Supported Formats:**
- PDF documents
- Word documents (DOCX)
- Text files
- HTML content
- Markdown files
- And more...

**Process:**
1. Content extraction and parsing
2. Semantic unit identification
3. 4D coordinate mapping via hypercube algorithms
4. Hilbert curve spatial indexing
5. Relationship computation with existing substrate

**Result:** Document becomes part of the universal semantic space
";
                break;

            case "CreateCompletion":
                operation.Summary = "Generate text completions (OpenAI-compatible)";
                operation.Description = @"
**OpenAI-compatible text generation powered by hypercube intelligence**

While maintaining full API compatibility with OpenAI, this endpoint uses revolutionary hypercube generation that combines:
- **Geometric proximity**: Concept relationships in 4D space
- **Pointwise mutual information**: Statistical co-occurrence patterns
- **Attention networks**: Contextual relationship understanding
- **Hilbert ordering**: Spatial coherence preservation

**Advantages over traditional models:**
- Better semantic coherence
- Cross-domain knowledge integration
- Geometric context awareness
- Multi-modal knowledge foundation

**Compatibility:** Drop-in replacement for OpenAI clients
";
                break;
        }

        // Add common parameters and responses
        if (operation.Parameters != null)
        {
            foreach (var parameter in operation.Parameters)
            {
                if (parameter.Name == "entity" || parameter.Name == "entity1" || parameter.Name == "entity2")
                {
                    parameter.Description = "Semantic unit identifier (word, phrase, or concept identifier)";
                    parameter.Schema.Example = new Microsoft.OpenApi.Any.OpenApiString("machine learning");
                }
                else if (parameter.Name == "k")
                {
                    parameter.Description = "Number of results to return (1-100)";
                    parameter.Schema.Example = new Microsoft.OpenApi.Any.OpenApiInteger(10);
                }
            }
        }
    }
}