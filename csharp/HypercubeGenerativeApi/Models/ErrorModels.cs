namespace HypercubeGenerativeApi.Models;

/// <summary>
/// OpenAI-compatible error response
/// </summary>
public class ErrorResponse
{
    /// <summary>
    /// Error details
    /// </summary>
    public ErrorDetails Error { get; set; } = new();
}

/// <summary>
/// Error details structure
/// </summary>
public class ErrorDetails
{
    /// <summary>
    /// Human-readable error message
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Error type/category
    /// </summary>
    public string Type { get; set; } = "internal_error";

    /// <summary>
    /// OpenAI-compatible error code
    /// </summary>
    public string? Code { get; set; }

    /// <summary>
    /// Parameter name that caused the error (for validation errors)
    /// </summary>
    public string? Param { get; set; }

    /// <summary>
    /// Additional error context (development only)
    /// </summary>
    public object? InternalMessage { get; set; }
}

/// <summary>
/// Common error types matching OpenAI
/// </summary>
public static class ErrorTypes
{
    public const string InvalidRequest = "invalid_request_error";
    public const string Authentication = "authentication_error";
    public const string Permission = "permission_error";
    public const string NotFound = "not_found_error";
    public const string Conflict = "conflict_error";
    public const string Unprocessable = "unprocessable_entity_error";
    public const string RateLimit = "rate_limit_error";
    public const string Internal = "internal_error";
    public const string ServiceUnavailable = "service_unavailable_error";
}

/// <summary>
/// Common error codes
/// </summary>
public static class ErrorCodes
{
    // Request validation
    public const string MissingRequiredParameter = "missing_required_parameter";
    public const string InvalidParameter = "invalid_parameter";
    public const string ParameterTooLarge = "parameter_too_large";
    public const string InvalidModel = "model_not_found";

    // Authentication/Authorization
    public const string InvalidApiKey = "invalid_api_key";
    public const string InsufficientPermissions = "insufficient_permissions";

    // Service state
    public const string ServiceUnavailable = "service_unavailable";
    public const string ModelNotLoaded = "model_not_loaded";
    public const string DatabaseUnavailable = "database_unavailable";

    // Generation errors
    public const string GenerationFailed = "generation_failed";
    public const string ContentFilter = "content_filter";
    public const string LengthExceeded = "maximum_length_exceeded";
}

/// <summary>
/// Error response factory methods
/// </summary>
public static class ErrorResponseFactory
{
    public static ErrorResponse BadRequest(string message, string code, string? param = null)
    {
        return new ErrorResponse
        {
            Error = new ErrorDetails
            {
                Message = message,
                Type = ErrorTypes.InvalidRequest,
                Code = code,
                Param = param
            }
        };
    }

    public static ErrorResponse NotFound(string message, string code = ErrorCodes.InvalidModel)
    {
        return new ErrorResponse
        {
            Error = new ErrorDetails
            {
                Message = message,
                Type = ErrorTypes.NotFound,
                Code = code
            }
        };
    }

    public static ErrorResponse ServiceUnavailable(string message, string code = ErrorCodes.ServiceUnavailable)
    {
        return new ErrorResponse
        {
            Error = new ErrorDetails
            {
                Message = message,
                Type = ErrorTypes.ServiceUnavailable,
                Code = code
            }
        };
    }

    public static ErrorResponse InternalError(string message, Exception? exception = null)
    {
        return new ErrorResponse
        {
            Error = new ErrorDetails
            {
                Message = message,
                Type = ErrorTypes.Internal,
                Code = "internal_error",
                InternalMessage = exception?.Message
            }
        };
    }
}