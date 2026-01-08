using HypercubeGenerativeApi.Models;
using HypercubeGenerativeApi.Services;

namespace HypercubeGenerativeApi.Controllers;

/// <summary>
/// OpenAI-compatible completions endpoints
/// </summary>
public static class CompletionsEndpoints
{
    /// <summary>
    /// POST /v1/completions - Create a completion
    /// </summary>
    public static async Task<IResult> CreateCompletion(
        CompletionRequest request,
        GenerativeService generativeService,
        ILogger<GenerativeService> logger)
    {
        try
        {
            logger.LogInformation("Received completion request for model {Model}", request.Model);

            // Validate request model
            if (request.Model != "hypercube-generative")
            {
                logger.LogWarning("Invalid model requested: {Model}", request.Model);
                return Results.NotFound(ErrorResponseFactory.NotFound($"Model '{request.Model}' not found"));
            }

            // Validate prompt
            if (string.IsNullOrWhiteSpace(request.Prompt))
            {
                logger.LogWarning("Empty prompt in completion request");
                return Results.BadRequest(ErrorResponseFactory.BadRequest(
                    "Prompt cannot be empty",
                    ErrorCodes.MissingRequiredParameter,
                    "prompt"));
            }

            // Validate prompt length
            if (request.Prompt.Length > 10000) // Reasonable limit
            {
                logger.LogWarning("Prompt too large: {Length} characters", request.Prompt.Length);
                return Results.BadRequest(ErrorResponseFactory.BadRequest(
                    "Prompt is too large",
                    ErrorCodes.ParameterTooLarge,
                    "prompt"));
            }

            // Validate max_tokens
            if (request.MaxTokens <= 0 || request.MaxTokens > 2048)
            {
                logger.LogWarning("Invalid max_tokens: {MaxTokens}", request.MaxTokens);
                return Results.BadRequest(ErrorResponseFactory.BadRequest(
                    "max_tokens must be between 1 and 2048",
                    ErrorCodes.InvalidParameter,
                    "max_tokens"));
            }

            // Validate temperature
            if (request.Temperature < 0 || request.Temperature > 2)
            {
                logger.LogWarning("Invalid temperature: {Temperature}", request.Temperature);
                return Results.BadRequest(ErrorResponseFactory.BadRequest(
                    "temperature must be between 0 and 2",
                    ErrorCodes.InvalidParameter,
                    "temperature"));
            }

            // Check if service is ready
            if (!generativeService.IsInitialized)
            {
                logger.LogWarning("Generation service not initialized");
                return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
                    "Service is initializing, please try again later",
                    ErrorCodes.ServiceUnavailable), statusCode: 503);
            }

            // Generate completion
            var response = await generativeService.GenerateCompletionAsync(request);

            logger.LogInformation("Completed generation request {Id}", response.Id);
            return Results.Ok(response);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("not initialized"))
        {
            logger.LogWarning(ex, "Service not initialized");
            return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
                "Service is not ready",
                ErrorCodes.ModelNotLoaded), statusCode: 503);
        }
        catch (ArgumentException ex)
        {
            logger.LogWarning(ex, "Invalid completion request parameter");
            return Results.BadRequest(ErrorResponseFactory.BadRequest(
                ex.Message,
                ErrorCodes.InvalidParameter));
        }
        catch (Npgsql.NpgsqlException ex)
        {
            logger.LogError(ex, "Database error during completion generation");
            return TypedResults.Json(ErrorResponseFactory.ServiceUnavailable(
                "Database temporarily unavailable",
                ErrorCodes.DatabaseUnavailable), statusCode: 503);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Unexpected error processing completion request");
            return TypedResults.Json(ErrorResponseFactory.InternalError(
                "An unexpected error occurred",
                ex), statusCode: 500);
        }
    }
}