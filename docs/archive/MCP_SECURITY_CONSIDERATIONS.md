# MCP Security Considerations

## Overview
This document outlines the security implications, requirements, and implementation strategies for MCP server and client functionality within the Hartonomous-Opus system. Security is paramount when exposing hypercube capabilities and connecting to external MCP servers.

## Threat Model

### Server-Side Threats

#### 1. Unauthorized Access
**Description**: Malicious clients attempting to invoke tools without proper authentication.
**Impact**: Exposure of sensitive hypercube data, unauthorized semantic operations.
**Likelihood**: High
**Mitigation Priority**: Critical

#### 2. Data Exfiltration
**Description**: Legitimate or compromised clients extracting sensitive information through tool responses.
**Impact**: Loss of intellectual property, privacy violations.
**Likelihood**: Medium-High
**Mitigation Priority**: Critical

#### 3. Resource Exhaustion
**Description**: Clients performing expensive operations repeatedly to consume system resources.
**Impact**: Denial of service, degraded performance for legitimate users.
**Likelihood**: High
**Mitigation Priority**: High

#### 4. Injection Attacks
**Description**: Malicious input in tool parameters causing unintended behavior.
**Impact**: Data corruption, system compromise.
**Likelihood**: Medium
**Mitigation Priority**: High

#### 5. Protocol Abuse
**Description**: Exploiting MCP protocol weaknesses or implementation bugs.
**Impact**: System instability, data manipulation.
**Likelihood**: Low-Medium
**Mitigation Priority**: Medium

### Client-Side Threats

#### 6. Malicious Servers
**Description**: External MCP servers providing compromised tools or responses.
**Impact**: Data leakage, system compromise through client-side execution.
**Likelihood**: Medium
**Mitigation Priority**: High

#### 7. Man-in-the-Middle Attacks
**Description**: Interception of MCP communications over insecure channels.
**Impact**: Credential theft, request/response manipulation.
**Likelihood**: Medium
**Mitigation Priority**: High

#### 8. Resource Consumption
**Description**: External servers causing excessive resource usage through responses.
**Impact**: Performance degradation, cost overruns.
**Likelihood**: Low-Medium
**Mitigation Priority**: Medium

#### 9. Self-Connection Loops
**Description**: Recursive tool calls leading to infinite loops or stack exhaustion.
**Impact**: System unavailability, resource exhaustion.
**Likelihood**: Medium
**Mitigation Priority**: High

## Authentication Framework

### Server Authentication

#### JWT Token Authentication
```csharp
public class JwtAuthenticationHandler : IMcpAuthenticationHandler
{
    private readonly IJwtTokenValidator _tokenValidator;
    private readonly IUserRepository _userRepository;

    public async Task<AuthenticationResult> AuthenticateAsync(McpRequestContext context)
    {
        var authHeader = context.Request.Headers.GetValueOrDefault("Authorization");

        if (string.IsNullOrEmpty(authHeader) || !authHeader.StartsWith("Bearer "))
        {
            return AuthenticationResult.Failed("Missing or invalid token");
        }

        var token = authHeader.Substring("Bearer ".Length);
        var validationResult = await _tokenValidator.ValidateAsync(token);

        if (!validationResult.IsValid)
        {
            return AuthenticationResult.Failed(validationResult.Error);
        }

        var user = await _userRepository.GetByIdAsync(validationResult.UserId);
        context.User = user;

        return AuthenticationResult.Success(user);
    }
}

public class AuthenticationResult
{
    public bool IsAuthenticated { get; private set; }
    public IUser User { get; private set; }
    public string Error { get; private set; }

    public static AuthenticationResult Success(IUser user) =>
        new() { IsAuthenticated = true, User = user };

    public static AuthenticationResult Failed(string error) =>
        new() { IsAuthenticated = false, Error = error };
}
```

#### API Key Authentication
```csharp
public class ApiKeyAuthenticationHandler : IMcpAuthenticationHandler
{
    private readonly IApiKeyValidator _apiKeyValidator;

    public async Task<AuthenticationResult> AuthenticateAsync(McpRequestContext context)
    {
        var apiKey = context.Request.Headers.GetValueOrDefault("X-API-Key") ??
                    context.Request.QueryParameters.GetValueOrDefault("api_key");

        if (string.IsNullOrEmpty(apiKey))
        {
            return AuthenticationResult.Failed("Missing API key");
        }

        var user = await _apiKeyValidator.ValidateAsync(apiKey);
        if (user == null)
        {
            return AuthenticationResult.Failed("Invalid API key");
        }

        context.User = user;
        return AuthenticationResult.Success(user);
    }
}
```

#### Multi-Factor Authentication (Future)
```csharp
public class MfaAuthenticationHandler : IMcpAuthenticationHandler
{
    private readonly IMfaService _mfaService;

    public async Task<AuthenticationResult> AuthenticateAsync(McpRequestContext context)
    {
        // Primary authentication (JWT/API key) already passed
        var user = context.User;

        // Check if MFA is required for this user/operation
        if (await _mfaService.IsMfaRequiredAsync(user, context.Request.ToolName))
        {
            var mfaToken = context.Request.Headers.GetValueOrDefault("X-MFA-Token");
            if (string.IsNullOrEmpty(mfaToken))
            {
                return AuthenticationResult.Failed("MFA token required", MfaChallenge.Required);
            }

            var mfaResult = await _mfaService.ValidateMfaTokenAsync(user, mfaToken);
            if (!mfaResult.IsValid)
            {
                return AuthenticationResult.Failed("Invalid MFA token");
            }
        }

        return AuthenticationResult.Success(user);
    }
}
```

### Client Authentication

#### External Server Authentication
```csharp
public class McpClientAuthenticationService
{
    private readonly Dictionary<string, IAuthenticationProvider> _providers = new();

    public void RegisterProvider(string serverId, IAuthenticationProvider provider)
    {
        _providers[serverId] = provider;
    }

    public async Task<AuthenticationHeaders> GetAuthenticationHeadersAsync(string serverId)
    {
        if (_providers.TryGetValue(serverId, out var provider))
        {
            return await provider.GetHeadersAsync();
        }

        return new AuthenticationHeaders(); // No authentication
    }
}

public interface IAuthenticationProvider
{
    Task<AuthenticationHeaders> GetHeadersAsync();
}

public class BearerTokenProvider : IAuthenticationProvider
{
    private readonly string _token;

    public BearerTokenProvider(string token) => _token = token;

    public Task<AuthenticationHeaders> GetHeadersAsync() =>
        Task.FromResult(new AuthenticationHeaders
        {
            { "Authorization", $"Bearer {_token}" }
        });
}

public class ApiKeyProvider : IAuthenticationProvider
{
    private readonly string _apiKey;
    private readonly string _headerName;

    public ApiKeyProvider(string apiKey, string headerName = "X-API-Key")
    {
        _apiKey = apiKey;
        _headerName = headerName;
    }

    public Task<AuthenticationHeaders> GetHeadersAsync() =>
        Task.FromResult(new AuthenticationHeaders
        {
            { _headerName, _apiKey }
        });
}
```

## Authorization Framework

### Role-Based Access Control (RBAC)
```csharp
public class McpAuthorizationService
{
    private readonly IPermissionRepository _permissionRepository;

    public async Task<bool> CheckPermissionAsync(IUser user, string toolName, object parameters = null)
    {
        var requiredPermissions = await GetRequiredPermissionsAsync(toolName);

        foreach (var permission in requiredPermissions)
        {
            if (!await user.HasPermissionAsync(permission))
            {
                return false;
            }
        }

        // Parameter-based authorization (e.g., user can only access their own data)
        if (parameters != null)
        {
            return await CheckParameterAuthorizationAsync(user, toolName, parameters);
        }

        return true;
    }

    private async Task<string[]> GetRequiredPermissionsAsync(string toolName)
    {
        return toolName switch
        {
            "SemanticSearch" => new[] { "semantic:read" },
            "IngestContent" => new[] { "content:write", "semantic:write" },
            "GenerateText" => new[] { "generation:execute" },
            "GeometricNeighbors" => new[] { "geometry:read" },
            _ => new[] { "tool:execute" }
        };
    }

    private async Task<bool> CheckParameterAuthorizationAsync(IUser user, string toolName, object parameters)
    {
        // Implement parameter-based authorization logic
        // e.g., check if user owns the data being accessed
        return true; // Placeholder
    }
}
```

### Attribute-Based Access Control (ABAC)
```csharp
[McpServerTool, RequiresPermission("semantic:search")]
[ParameterConstraint("limit", MaxValue = 100)]
[ParameterConstraint("query", MaxLength = 1000)]
public async Task<SemanticSearchResult> SemanticSearch(
    [Inject] IUser user,
    string query, int limit = 10)
{
    // Tool implementation
}

public class ParameterConstraintAttribute : Attribute
{
    public string ParameterName { get; }
    public object MaxValue { get; set; }
    public object MinValue { get; set; }
    public int MaxLength { get; set; }

    public ParameterConstraintAttribute(string parameterName)
    {
        ParameterName = parameterName;
    }
}

public class ParameterValidator
{
    public ValidationResult ValidateParameters(
        MethodInfo method,
        Dictionary<string, object> parameters)
    {
        var constraints = method.GetCustomAttributes<ParameterConstraintAttribute>();
        var errors = new List<string>();

        foreach (var constraint in constraints)
        {
            if (parameters.TryGetValue(constraint.ParameterName, out var value))
            {
                if (!ValidateConstraint(value, constraint, out var error))
                {
                    errors.Add(error);
                }
            }
        }

        return errors.Any()
            ? ValidationResult.Invalid(errors)
            : ValidationResult.Valid;
    }

    private bool ValidateConstraint(object value, ParameterConstraintAttribute constraint, out string error)
    {
        error = null;

        if (constraint.MaxValue != null && Comparer.Default.Compare(value, constraint.MaxValue) > 0)
        {
            error = $"{constraint.ParameterName} exceeds maximum value {constraint.MaxValue}";
            return false;
        }

        if (constraint.MinValue != null && Comparer.Default.Compare(value, constraint.MinValue) < 0)
        {
            error = $"{constraint.ParameterName} below minimum value {constraint.MinValue}";
            return false;
        }

        if (constraint.MaxLength > 0 && value is string str && str.Length > constraint.MaxLength)
        {
            error = $"{constraint.ParameterName} exceeds maximum length {constraint.MaxLength}";
            return false;
        }

        return true;
    }
}
```

## Input Validation and Sanitization

### Comprehensive Input Validation
```csharp
public class McpInputValidator
{
    private readonly Dictionary<string, ValidationRule[]> _toolValidationRules;

    public McpInputValidator()
    {
        _toolValidationRules = new Dictionary<string, ValidationRule[]>
        {
            ["SemanticSearch"] = new[]
            {
                new ValidationRule
                {
                    Parameter = "query",
                    Required = true,
                    Type = typeof(string),
                    MinLength = 1,
                    MaxLength = 1000,
                    Pattern = @"^[\w\s\p{P}]+$", // Word chars, spaces, punctuation
                    Sanitizers = new[] { "trim", "normalize-whitespace" }
                },
                new ValidationRule
                {
                    Parameter = "limit",
                    Type = typeof(int),
                    MinValue = 1,
                    MaxValue = 100,
                    DefaultValue = 10
                }
            },
            ["GenerateText"] = new[]
            {
                new ValidationRule
                {
                    Parameter = "prompt",
                    Required = true,
                    Type = typeof(string),
                    MinLength = 1,
                    MaxLength = 5000,
                    Sanitizers = new[] { "trim", "html-encode" },
                    ContentFilters = new[] { "profanity", "personal-info" }
                }
            }
        };
    }

    public ValidationResult ValidateToolCall(string toolName, Dictionary<string, object> parameters)
    {
        if (!_toolValidationRules.TryGetValue(toolName, out var rules))
        {
            return ValidationResult.Valid; // No specific rules
        }

        var errors = new List<string>();
        var sanitizedParameters = new Dictionary<string, object>();

        foreach (var rule in rules)
        {
            if (!parameters.TryGetValue(rule.Parameter, out var value))
            {
                if (rule.Required)
                {
                    errors.Add($"{rule.Parameter} is required");
                    continue;
                }
                if (rule.DefaultValue != null)
                {
                    value = rule.DefaultValue;
                }
            }

            if (value != null)
            {
                var validation = ValidateValue(value, rule);
                if (!validation.IsValid)
                {
                    errors.Add(validation.Error);
                    continue;
                }

                sanitizedParameters[rule.Parameter] = validation.SanitizedValue;
            }
        }

        return errors.Any()
            ? ValidationResult.Invalid(errors)
            : ValidationResult.ValidWithSanitized(sanitizedParameters);
    }

    private ValidationResult ValidateValue(object value, ValidationRule rule)
    {
        // Type validation
        if (rule.Type != null && value.GetType() != rule.Type)
        {
            return ValidationResult.Invalid($"Invalid type for {rule.Parameter}");
        }

        // String validations
        if (value is string str)
        {
            if (rule.MinLength > 0 && str.Length < rule.MinLength)
                return ValidationResult.Invalid($"{rule.Parameter} too short");

            if (rule.MaxLength > 0 && str.Length > rule.MaxLength)
                return ValidationResult.Invalid($"{rule.Parameter} too long");

            if (!string.IsNullOrEmpty(rule.Pattern) && !Regex.IsMatch(str, rule.Pattern))
                return ValidationResult.Invalid($"{rule.Parameter} contains invalid characters");

            // Apply sanitizers
            var sanitized = ApplySanitizers(str, rule.Sanitizers);
            return ValidationResult.ValidWithSanitized(sanitized);
        }

        // Numeric validations
        if (value is int intValue)
        {
            if (rule.MinValue != null && intValue < (int)rule.MinValue)
                return ValidationResult.Invalid($"{rule.Parameter} below minimum");

            if (rule.MaxValue != null && intValue > (int)rule.MaxValue)
                return ValidationResult.Invalid($"{rule.Parameter} above maximum");
        }

        return ValidationResult.Valid;
    }

    private string ApplySanitizers(string value, string[] sanitizers)
    {
        if (sanitizers == null) return value;

        var result = value;
        foreach (var sanitizer in sanitizers)
        {
            result = sanitizer switch
            {
                "trim" => result.Trim(),
                "normalize-whitespace" => Regex.Replace(result, @"\s+", " "),
                "html-encode" => WebUtility.HtmlEncode(result),
                _ => result
            };
        }
        return result;
    }
}

public class ValidationRule
{
    public string Parameter { get; set; }
    public bool Required { get; set; }
    public Type Type { get; set; }
    public int MinLength { get; set; }
    public int MaxLength { get; set; }
    public object MinValue { get; set; }
    public object MaxValue { get; set; }
    public string Pattern { get; set; }
    public object DefaultValue { get; set; }
    public string[] Sanitizers { get; set; }
    public string[] ContentFilters { get; set; }
}

public class ValidationResult
{
    public bool IsValid { get; private set; }
    public List<string> Errors { get; private set; } = new();
    public object SanitizedValue { get; private set; }

    public static ValidationResult Valid => new() { IsValid = true };
    public static ValidationResult ValidWithSanitized(object sanitizedValue) =>
        new() { IsValid = true, SanitizedValue = sanitizedValue };
    public static ValidationResult Invalid(string error) =>
        new() { IsValid = false, Errors = new List<string> { error } };
    public static ValidationResult Invalid(List<string> errors) =>
        new() { IsValid = false, Errors = errors };
}
```

### Content Filtering
```csharp
public class ContentFilterService
{
    private readonly Dictionary<string, IContentFilter> _filters = new();

    public ContentFilterService()
    {
        _filters["profanity"] = new ProfanityFilter();
        _filters["personal-info"] = new PersonalInfoFilter();
        _filters["malicious-code"] = new MaliciousCodeFilter();
    }

    public ContentFilterResult Filter(string content, string[] filterNames)
    {
        var filteredContent = content;
        var triggeredFilters = new List<string>();
        var severity = ContentFilterSeverity.Allow;

        foreach (var filterName in filterNames)
        {
            if (_filters.TryGetValue(filterName, out var filter))
            {
                var result = filter.Filter(filteredContent);
                if (result.Severity > severity)
                {
                    severity = result.Severity;
                }
                if (result.Modified)
                {
                    filteredContent = result.Content;
                    triggeredFilters.Add(filterName);
                }
            }
        }

        return new ContentFilterResult
        {
            Content = filteredContent,
            Severity = severity,
            TriggeredFilters = triggeredFilters.ToArray()
        };
    }
}

public enum ContentFilterSeverity
{
    Allow,
    Warn,
    Block
}

public class ContentFilterResult
{
    public string Content { get; set; }
    public ContentFilterSeverity Severity { get; set; }
    public string[] TriggeredFilters { get; set; }
}

public interface IContentFilter
{
    ContentFilterResult Filter(string content);
}

public class ProfanityFilter : IContentFilter
{
    private readonly HashSet<string> _profanityWords = new()
    {
        // Profanity list would be populated here
    };

    public ContentFilterResult Filter(string content)
    {
        var words = content.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var modified = false;

        for (int i = 0; i < words.Length; i++)
        {
            if (_profanityWords.Contains(words[i].ToLower()))
            {
                words[i] = "***";
                modified = true;
            }
        }

        return new ContentFilterResult
        {
            Content = string.Join(" ", words),
            Severity = modified ? ContentFilterSeverity.Warn : ContentFilterSeverity.Allow,
            TriggeredFilters = modified ? new[] { "profanity" } : Array.Empty<string>()
        };
    }
}
```

## Rate Limiting and Throttling

### Multi-Level Rate Limiting
```csharp
public class McpRateLimiter
{
    private readonly ConcurrentDictionary<string, UserRateLimit> _userLimits = new();
    private readonly ConcurrentDictionary<string, ToolRateLimit> _toolLimits = new();
    private readonly ConcurrentDictionary<string, IpRateLimit> _ipLimits = new();

    public async Task<bool> CheckRateLimitAsync(McpRequestContext context)
    {
        var userId = context.User?.Id ?? "anonymous";
        var toolName = context.Request.ToolName;
        var ipAddress = context.Connection.RemoteIpAddress?.ToString();

        // User-level rate limiting
        var userLimit = _userLimits.GetOrAdd(userId, _ => new UserRateLimit());
        if (!await userLimit.CheckLimitAsync())
            return false;

        // Tool-level rate limiting
        var toolLimit = _toolLimits.GetOrAdd(toolName, _ => new ToolRateLimit(toolName));
        if (!await toolLimit.CheckLimitAsync())
            return false;

        // IP-level rate limiting (for anonymous users)
        if (context.User == null && !string.IsNullOrEmpty(ipAddress))
        {
            var ipLimit = _ipLimits.GetOrAdd(ipAddress, _ => new IpRateLimit());
            if (!await ipLimit.CheckLimitAsync())
                return false;
        }

        return true;
    }
}

public class UserRateLimit
{
    private readonly TokenBucket _requestsPerMinute = new(60, TimeSpan.FromMinutes(1));
    private readonly TokenBucket _requestsPerHour = new(1000, TimeSpan.FromHours(1));

    public async Task<bool> CheckLimitAsync()
    {
        return await _requestsPerMinute.TryConsumeAsync(1) &&
               await _requestsPerHour.TryConsumeAsync(1);
    }
}

public class ToolRateLimit
{
    private readonly TokenBucket _bucket;

    public ToolRateLimit(string toolName)
    {
        // Different limits for different tools
        var (capacity, period) = toolName switch
        {
            "GenerateText" => (10, TimeSpan.FromMinutes(1)), // Expensive operation
            "SemanticSearch" => (100, TimeSpan.FromMinutes(1)), // Less expensive
            _ => (50, TimeSpan.FromMinutes(1))
        };

        _bucket = new TokenBucket(capacity, period);
    }

    public async Task<bool> CheckLimitAsync() => await _bucket.TryConsumeAsync(1);
}

public class IpRateLimit
{
    private readonly TokenBucket _bucket = new(20, TimeSpan.FromMinutes(1));

    public async Task<bool> CheckLimitAsync() => await _bucket.TryConsumeAsync(1);
}

public class TokenBucket
{
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private double _tokens;
    private DateTime _lastRefill;
    private readonly double _capacity;
    private readonly TimeSpan _refillPeriod;

    public TokenBucket(int capacity, TimeSpan refillPeriod)
    {
        _capacity = capacity;
        _refillPeriod = refillPeriod;
        _tokens = capacity;
        _lastRefill = DateTime.UtcNow;
    }

    public async Task<bool> TryConsumeAsync(int tokens)
    {
        await _semaphore.WaitAsync();
        try
        {
            Refill();
            if (_tokens >= tokens)
            {
                _tokens -= tokens;
                return true;
            }
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    private void Refill()
    {
        var now = DateTime.UtcNow;
        var timePassed = now - _lastRefill;
        var refillAmount = (_capacity / _refillPeriod.TotalSeconds) * timePassed.TotalSeconds;

        _tokens = Math.Min(_capacity, _tokens + refillAmount);
        _lastRefill = now;
    }
}
```

## Transport Security

### TLS Configuration
```csharp
public class McpSecurityConfig
{
    public TlsConfig Tls { get; set; } = new();
}

public class TlsConfig
{
    public bool Enabled { get; set; } = true;
    public string CertificatePath { get; set; }
    public string CertificatePassword { get; set; }
    public SslProtocols Protocols { get; set; } = SslProtocols.Tls12 | SslProtocols.Tls13;
    public CipherSuitePolicy CipherSuites { get; set; } = CipherSuitePolicy.Restricted;
}

public class TlsTransportDecorator : IMcpTransport
{
    private readonly IMcpTransport _innerTransport;
    private readonly TlsConfig _tlsConfig;

    public TlsTransportDecorator(IMcpTransport innerTransport, TlsConfig tlsConfig)
    {
        _innerTransport = innerTransport;
        _tlsConfig = tlsConfig;
    }

    public async Task<McpResponse> SendRequestAsync(McpRequest request)
    {
        // TLS handshake and encryption would be handled by underlying transport
        // This decorator ensures proper TLS configuration
        return await _innerTransport.SendRequestAsync(request);
    }
}
```

### Certificate Management
```csharp
public class CertificateManager
{
    private readonly X509Certificate2 _serverCertificate;
    private readonly Dictionary<string, X509Certificate2> _clientCertificates = new();

    public CertificateManager(TlsConfig config)
    {
        if (!string.IsNullOrEmpty(config.CertificatePath))
        {
            _serverCertificate = new X509Certificate2(
                config.CertificatePath,
                config.CertificatePassword);
        }
    }

    public X509Certificate2 GetServerCertificate() => _serverCertificate;

    public void AddClientCertificate(string clientId, X509Certificate2 certificate)
    {
        _clientCertificates[clientId] = certificate;
    }

    public X509Certificate2 GetClientCertificate(string clientId)
    {
        return _clientCertificates.TryGetValue(clientId, out var cert) ? cert : null;
    }

    public bool ValidateClientCertificate(X509Certificate2 certificate, X509Chain chain)
    {
        // Custom certificate validation logic
        // Check certificate validity, chain trust, revocation, etc.
        return true; // Placeholder
    }
}
```

## Audit Logging and Monitoring

### Comprehensive Audit Logging
```csharp
public class McpAuditLogger
{
    private readonly ILogger _logger;
    private readonly IAuditLogRepository _auditRepository;

    public async Task LogRequestAsync(McpRequestContext context)
    {
        var auditEntry = new AuditEntry
        {
            Timestamp = DateTime.UtcNow,
            UserId = context.User?.Id,
            IpAddress = context.Connection.RemoteIpAddress?.ToString(),
            ToolName = context.Request.ToolName,
            Parameters = SanitizeParameters(context.Request.Parameters),
            UserAgent = context.Request.Headers.GetValueOrDefault("User-Agent"),
            SessionId = context.SessionId
        };

        await _auditRepository.SaveAsync(auditEntry);

        _logger.LogInformation(
            "MCP Request: User={UserId}, Tool={ToolName}, IP={IpAddress}, Session={SessionId}",
            auditEntry.UserId, auditEntry.ToolName, auditEntry.IpAddress, auditEntry.SessionId);
    }

    public async Task LogResponseAsync(McpRequestContext context, McpResponse response, TimeSpan duration)
    {
        var auditEntry = new AuditEntry
        {
            Timestamp = DateTime.UtcNow,
            UserId = context.User?.Id,
            ToolName = context.Request.ToolName,
            Duration = duration,
            Success = response.Success,
            ErrorCode = response.Error?.Code,
            ResponseSize = CalculateResponseSize(response),
            SessionId = context.SessionId
        };

        await _auditRepository.SaveAsync(auditEntry);

        if (!response.Success)
        {
            _logger.LogWarning(
                "MCP Error Response: User={UserId}, Tool={ToolName}, Error={ErrorCode}, Duration={Duration}ms",
                auditEntry.UserId, auditEntry.ToolName, auditEntry.ErrorCode, duration.TotalMilliseconds);
        }
    }

    private Dictionary<string, object> SanitizeParameters(Dictionary<string, object> parameters)
    {
        // Remove or mask sensitive parameters like passwords, tokens, etc.
        var sanitized = new Dictionary<string, object>(parameters);

        // Mask sensitive fields
        if (sanitized.ContainsKey("password"))
            sanitized["password"] = "***";
        if (sanitized.ContainsKey("token"))
            sanitized["token"] = sanitized["token"].ToString()?.Substring(0, 8) + "***";

        return sanitized;
    }

    private long CalculateResponseSize(McpResponse response)
    {
        // Calculate approximate response size for monitoring
        return JsonSerializer.Serialize(response).Length;
    }
}

public class AuditEntry
{
    public DateTime Timestamp { get; set; }
    public string UserId { get; set; }
    public string IpAddress { get; set; }
    public string ToolName { get; set; }
    public Dictionary<string, object> Parameters { get; set; }
    public TimeSpan Duration { get; set; }
    public bool Success { get; set; }
    public int? ErrorCode { get; set; }
    public long ResponseSize { get; set; }
    public string UserAgent { get; set; }
    public string SessionId { get; set; }
}
```

### Real-Time Security Monitoring
```csharp
public class SecurityMonitor
{
    private readonly ILogger _logger;
    private readonly IAlertService _alertService;
    private readonly Dictionary<string, SecurityMetrics> _metrics = new();

    public void RecordSecurityEvent(SecurityEventType eventType, string userId, string details)
    {
        var metrics = _metrics.GetOrAdd(userId, _ => new SecurityMetrics());

        switch (eventType)
        {
            case SecurityEventType.FailedAuthentication:
                metrics.FailedAuthAttempts++;
                if (metrics.FailedAuthAttempts >= 5)
                {
                    _alertService.SendAlert($"Multiple failed auth attempts for user {userId}");
                }
                break;

            case SecurityEventType.RateLimitExceeded:
                metrics.RateLimitHits++;
                if (metrics.RateLimitHits >= 10)
                {
                    _alertService.SendAlert($"Rate limit abuse by user {userId}");
                }
                break;

            case SecurityEventType.SuspiciousRequest:
                metrics.SuspiciousRequests++;
                _alertService.SendAlert($"Suspicious request from user {userId}: {details}");
                break;
        }

        _logger.LogWarning("Security Event: {EventType}, User: {UserId}, Details: {Details}",
            eventType, userId, details);
    }

    public void ResetMetrics(string userId)
    {
        _metrics.Remove(userId);
    }
}

public enum SecurityEventType
{
    FailedAuthentication,
    RateLimitExceeded,
    SuspiciousRequest,
    InvalidInput,
    UnauthorizedAccess
}

public class SecurityMetrics
{
    public int FailedAuthAttempts { get; set; }
    public int RateLimitHits { get; set; }
    public int SuspiciousRequests { get; set; }
    public DateTime LastActivity { get; set; } = DateTime.UtcNow;
}
```

## Incident Response Plan

### Security Incident Classification
- **Critical**: Unauthorized access to sensitive data, system compromise
- **High**: Successful attacks, data breaches, service disruption
- **Medium**: Failed attack attempts, suspicious activities
- **Low**: Policy violations, minor security events

### Response Procedures

#### Critical Incident Response
1. **Immediate Actions**:
   - Isolate affected systems
   - Disable MCP endpoints
   - Notify security team and management
   - Preserve evidence and logs

2. **Investigation**:
   - Analyze audit logs and system logs
   - Identify attack vectors and compromised data
   - Assess damage and breach scope

3. **Recovery**:
   - Patch vulnerabilities
   - Restore from clean backups
   - Monitor for additional attacks
   - Update security measures

4. **Post-Incident**:
   - Document findings and lessons learned
   - Update security policies and procedures
   - Conduct security awareness training

#### Automated Response Actions
```csharp
public class IncidentResponseCoordinator
{
    private readonly ISecurityMonitor _securityMonitor;
    private readonly IMcpServerController _serverController;

    public async Task HandleSecurityIncidentAsync(SecurityIncident incident)
    {
        switch (incident.Severity)
        {
            case IncidentSeverity.Critical:
                await HandleCriticalIncidentAsync(incident);
                break;
            case IncidentSeverity.High:
                await HandleHighIncidentAsync(incident);
                break;
            case IncidentSeverity.Medium:
                await HandleMediumIncidentAsync(incident);
                break;
        }
    }

    private async Task HandleCriticalIncidentAsync(SecurityIncident incident)
    {
        // Immediate shutdown
        await _serverController.ShutdownAsync();

        // Alert all stakeholders
        await _securityMonitor.SendCriticalAlertAsync(incident);

        // Log incident details
        await LogIncidentAsync(incident, "Critical incident - system shutdown initiated");
    }

    private async Task HandleHighIncidentAsync(SecurityIncident incident)
    {
        // Temporary disable suspicious user
        await _securityMonitor.DisableUserAsync(incident.UserId);

        // Increase monitoring
        await _securityMonitor.EnableEnhancedMonitoringAsync(incident.UserId);

        // Send alerts
        await _securityMonitor.SendHighAlertAsync(incident);
    }

    private async Task HandleMediumIncidentAsync(SecurityIncident incident)
    {
        // Log and monitor
        await LogIncidentAsync(incident, "Medium incident - monitoring increased");

        // Send notifications
        await _securityMonitor.SendNotificationAsync(incident);
    }
}
```

This comprehensive security framework ensures that MCP server and client functionality is implemented with robust protection against various threats, comprehensive monitoring, and effective incident response capabilities.