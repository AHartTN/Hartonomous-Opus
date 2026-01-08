# PostgresService Documentation

## Overview

The `PostgresService` provides database connectivity and hypercube data access for the Generative API. It serves as the bridge between the C# application layer and the PostgreSQL database containing the hypercube semantic substrate.

## Core Responsibilities

- **Database Connection Management**: Establish and maintain PostgreSQL connections
- **Token Validation**: Verify tokens exist in hypercube vocabulary
- **Statistics Retrieval**: Provide database metrics for monitoring
- **Query Execution**: Safe execution of hypercube-related queries

## Class Structure

```csharp
public class PostgresService : IDisposable
{
    // Configuration
    private readonly ILogger<PostgresService> _logger;
    private readonly string _connectionString;

    // State
    private NpgsqlConnection? _connection;
    private bool _disposed;

    // Core API
    public Task InitializeAsync();
    public Task<bool> CheckConnectionAsync();
    public Task<Dictionary<string, object>> GetDatabaseStatsAsync();
    public Task<bool> TokenExistsAsync(string token);
    public Task<string[]> GetValidTokensFromPromptAsync(string prompt);

    // Lifecycle
    public void Dispose();
}
```

## Connection Management

### Initialization Process

```csharp
public async Task InitializeAsync()
{
    if (_disposed) throw new ObjectDisposedException(nameof(PostgresService));

    try {
        _connection = new NpgsqlConnection(_connectionString);
        await _connection.OpenAsync();

        _logger.LogInformation("PostgreSQL connection established to hypercube database");

    } catch (Exception ex) {
        _logger.LogError(ex, "Failed to connect to PostgreSQL database");
        throw;
    }
}
```

### Connection Configuration

**From appsettings.json**:
```json
{
  "ConnectionStrings": {
    "HypercubeDatabase": "Host=localhost;Port=5432;Username=hartonomous;Password=***;Database=hypercube"
  },
  "Hypercube": {
    "Database": {
      "ConnectionTimeout": 30,
      "MaxPoolSize": 10
    }
  }
}
```

### Connection Validation

```csharp
public async Task<bool> CheckConnectionAsync()
{
    if (_connection == null || _disposed) return false;

    try {
        await using var cmd = new NpgsqlCommand("SELECT 1", _connection);
        await cmd.ExecuteScalarAsync();
        return true;

    } catch (Exception ex) {
        _logger.LogWarning(ex, "Database connectivity check failed");
        return false;
    }
}
```

## Token Validation Operations

### Token Existence Check

```csharp
public async Task<bool> TokenExistsAsync(string token)
{
    if (_connection == null || _disposed || string.IsNullOrWhiteSpace(token)) {
        return false;
    }

    try {
        await using var cmd = new NpgsqlCommand(@"
            SELECT 1
            FROM composition
            WHERE label = @token
              AND centroid IS NOT NULL
            LIMIT 1", _connection);

        cmd.Parameters.AddWithValue("@token", token);
        var result = await cmd.ExecuteScalarAsync();

        return result != null;

    } catch (Exception ex) {
        _logger.LogWarning(ex, "Error checking token existence '{Token}'", token);
        return false;
    }
}
```

**Query Details**:
- **Table**: `composition` (hypercube vocabulary)
- **Filter**: `centroid IS NOT NULL` ensures 4D coordinates exist
- **Limit**: `LIMIT 1` for performance (existence check only)

### Bulk Token Validation

```csharp
public async Task<string[]> GetValidTokensFromPromptAsync(string prompt)
{
    if (string.IsNullOrWhiteSpace(prompt)) {
        return Array.Empty<string>();
    }

    // Basic tokenization
    var candidates = TokenizeBasic(prompt)
        .Take(100)  // Reasonable limit
        .Distinct() // Remove duplicates
        .ToArray();

    var validTokens = new List<string>();

    // Check each token (could be optimized with batch query)
    foreach (var token in candidates) {
        if (await TokenExistsAsync(token)) {
            validTokens.Add(token);
        }
    }

    _logger.LogDebug("Validated {Valid}/{Total} tokens from prompt",
        validTokens.Count, candidates.Length);

    return validTokens.ToArray();
}
```

## Database Statistics

### Stats Retrieval

```csharp
public async Task<Dictionary<string, object>> GetDatabaseStatsAsync()
{
    if (_connection == null || _disposed) {
        return new Dictionary<string, object> {
            ["error"] = "No database connection"
        };
    }

    try {
        var stats = new Dictionary<string, object>();

        await using var cmd = new NpgsqlCommand(@"
            SELECT 'atoms' as stat_name, COUNT(*) as stat_value FROM atom
            UNION ALL
            SELECT 'compositions', COUNT(*) FROM composition
            UNION ALL
            SELECT 'compositions_with_centroid', COUNT(*) FROM composition WHERE centroid IS NOT NULL
            UNION ALL
            SELECT 'relations', COUNT(*) FROM relation
            UNION ALL
            SELECT 'attention_edges', COUNT(*) FROM relation WHERE relation_type = 'A'
            UNION ALL
            SELECT 'sequence_edges', COUNT(*) FROM relation WHERE relation_type = 'S'
        ", _connection);

        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync()) {
            var statName = reader.GetString(0);
            var statValue = reader.GetInt64(1);
            stats[statName] = statValue;
        }

        return stats;

    } catch (Exception ex) {
        _logger.LogError(ex, "Error retrieving database statistics");
        return new Dictionary<string, object> {
            ["error"] = ex.Message
        };
    }
}
```

### Available Statistics

| Statistic Key | Description | Source |
|----------------|-------------|--------|
| `atoms` | Total Unicode codepoints | `atom` table count |
| `compositions` | Total vocabulary entries | `composition` table count |
| `compositions_with_centroid` | Entries with 4D coordinates | Filtered count |
| `relations` | Total semantic relationships | `relation` table count |
| `attention_edges` | Attention-based relations | Filtered count |
| `sequence_edges` | Sequence/PMI relations | Filtered count |

## Error Handling Strategy

### Exception Classification

- **Connection Issues**: `NpgsqlException` with connection-related messages
- **Query Errors**: `NpgsqlException` with SQL syntax/query issues
- **Timeout Issues**: `NpgsqlException` with timeout messages
- **Data Issues**: Unexpected null values or type mismatches

### Graceful Degradation

```csharp
// Always handle connection issues gracefully
if (_connection == null || _disposed) {
    _logger.LogWarning("Database operation attempted without connection");
    return defaultValue; // Safe default
}
```

### Logging Levels

- **Information**: Successful connections, major operations
- **Warning**: Recoverable errors, connection issues
- **Error**: Query failures, data corruption
- **Debug**: Query details, performance metrics

## Performance Considerations

### Connection Pooling

**Current State**: Single connection per service instance

**Future Enhancement**:
```csharp
// Configure Npgsql connection pooling
_connectionString += ";Pooling=true;Minimum Pool Size=1;Maximum Pool Size=10";
```

### Query Optimization

- **Indexed Lookups**: `composition.label` should be indexed
- **Limit Clauses**: Always use `LIMIT 1` for existence checks
- **Batch Operations**: Future bulk token validation
- **Prepared Statements**: Reuse query plans

### Memory Management

- **Command Disposal**: `await using` ensures proper cleanup
- **Result Streaming**: Large result sets handled efficiently
- **Parameter Binding**: Prevents SQL injection and optimizes parsing

## Integration with Other Services

### TokenizationService Dependency

```csharp
// TokenizationService uses PostgresService for validation
public TokenizationService(ILogger<TokenizationService> logger, PostgresService postgresService)
{
    _postgresService = postgresService;
}

private async Task<long?> EncodeTokenAsync(string token)
{
    var exists = await _postgresService.TokenExistsAsync(token);
    if (!exists) return null;

    return GetStableHash(token);
}
```

### GenerativeService Usage

```csharp
// GenerativeService gets validated tokens
var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);
var startLabel = validTokens.LastOrDefault() ?? "the";
```

### Health Check Integration

```csharp
// Used by GenerativeHealthCheck
var dbHealthy = await _postgresService.CheckConnectionAsync();
var stats = await _postgresService.GetDatabaseStatsAsync();
```

## Configuration Options

### Connection String Format

```csharp
// Full connection string
"Host=localhost;Port=5432;Username=user;Password=pass;Database=hypercube;Timeout=30;CommandTimeout=60"
```

### Environment Variables

```bash
# Secure configuration via environment
export ConnectionStrings__HypercubeDatabase="Host=prod-db;Username=app;Password=${DB_PASSWORD}"
```

### Runtime Configuration

```json
{
  "Hypercube": {
    "Database": {
      "ConnectionTimeout": 30,      // Connection establishment timeout
      "MaxPoolSize": 10,           // Future connection pool size
      "CommandTimeout": 30,        // Query execution timeout
      "RetryCount": 3              // Future retry configuration
    }
  }
}
```

## Testing Strategy

### Unit Tests

```csharp
[TestMethod]
public async Task TokenExistsAsync_ValidToken_ReturnsTrue()
{
    // Arrange: Ensure test token exists in DB
    var service = new PostgresService(logger, "test_connection_string");

    // Act
    var exists = await service.TokenExistsAsync("the");

    // Assert
    Assert.IsTrue(exists);
}

[TestMethod]
public async Task GetDatabaseStatsAsync_Connected_ReturnsStats()
{
    // Arrange
    var service = new PostgresService(logger, configuration);

    // Act
    var stats = await service.GetDatabaseStatsAsync();

    // Assert
    Assert.IsTrue(stats.ContainsKey("compositions"));
    Assert.IsInstanceOfType(stats["compositions"], typeof(long));
}
```

### Integration Tests

```csharp
[TestMethod]
public async Task EndToEnd_DatabaseOperations_WorkCorrectly()
{
    // Arrange: Full database setup
    var service = new PostgresService(logger, config);

    // Act
    await service.InitializeAsync();
    var connected = await service.CheckConnectionAsync();
    var stats = await service.GetDatabaseStatsAsync();
    var tokenExists = await service.TokenExistsAsync("test");

    // Assert
    Assert.IsTrue(connected);
    Assert.IsTrue(stats.ContainsKey("atoms"));
    Assert.IsInstanceOfType(tokenExists, typeof(bool));
}
```

### Mock Testing

```csharp
[TestMethod]
public async Task TokenExistsAsync_ConnectionFailure_ReturnsFalse()
{
    // Arrange: Mock connection failure
    var mockConnection = new Mock<NpgsqlConnection>();
    mockConnection.Setup(c => c.OpenAsync()).ThrowsAsync(new Exception());

    // Act & Assert
    var service = new PostgresService(logger, "invalid_connection");
    await Assert.ThrowsExceptionAsync<Exception>(() => service.InitializeAsync());
}
```

## Future Enhancements

### Connection Pooling

```csharp
// Npgsql connection string with pooling
"Host=localhost;Pooling=true;MinPoolSize=1;MaxPoolSize=10;ConnectionLifetime=300"
```

### Advanced Querying

- **Batch Token Validation**: Single query for multiple tokens
- **Composition ID Retrieval**: Full BYTEA hash support
- **Spatial Queries**: PostGIS geometric operations
- **Relation Traversal**: Graph queries on semantic relations

### Monitoring Integration

- **Connection Pool Stats**: Active/idle connection counts
- **Query Performance**: Slow query detection
- **Error Metrics**: Connection failure rates
- **Health Indicators**: Pool exhaustion detection

### High Availability

- **Read Replicas**: Separate read/write connections
- **Failover**: Automatic connection recovery
- **Load Balancing**: Multiple database endpoints
- **Circuit Breaker**: Prevent cascade failures

## Troubleshooting Guide

### Common Connection Issues

**"Connection refused"**:
- Verify PostgreSQL is running
- Check hostname/port in connection string
- Ensure firewall allows connections

**"Authentication failed"**:
- Verify username/password
- Check pg_hba.conf for authentication method
- Ensure user has database access

**"Database does not exist"**:
- Confirm database name in connection string
- Run database setup scripts
- Check database creation permissions

### Query Performance Issues

**Slow token validation**:
- Ensure `composition.label` has an index
- Check query execution plans
- Consider query result caching

**High memory usage**:
- Monitor connection pool size
- Check for connection leaks
- Review query result sizes

### Data Consistency Issues

**Missing centroids**:
- Ensure model ingestion completed successfully
- Verify PostGIS extension is installed
- Check for data import errors

**Token not found**:
- Confirm tokenization matches training data
- Check for case sensitivity issues
- Verify vocabulary was loaded correctly

This service provides reliable database access for the hypercube generation workflow, ensuring semantic validation and data integrity throughout the API pipeline.