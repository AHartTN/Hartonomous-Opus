using HypercubeGenerativeApi.Models;

namespace HypercubeGenerativeApi.Controllers;

/// <summary>
/// OpenAI-compatible models endpoints
/// </summary>
public static class ModelsEndpoints
{
    /// <summary>
    /// GET /v1/models - List available models
    /// </summary>
    public static IResult ListModels()
    {
        var response = new ModelsResponse
        {
            Data = new[]
            {
                new ModelInfo
                {
                    Id = "hypercube-generative",
                    Created = 1677652288, // 2023-03-01 approximate
                    OwnedBy = "hartonomous-opus"
                }
            }
        };

        return Results.Ok(response);
    }
}