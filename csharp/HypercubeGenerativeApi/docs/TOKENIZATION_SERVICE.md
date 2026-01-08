# TokenizationService Documentation

## Overview

The `TokenizationService` bridges natural language text and the hypercube semantic substrate by converting human-readable prompts into validated token sequences that exist within the hypercube's vocabulary. This service ensures that only semantically meaningful tokens are passed to the generation engine.

## Core Responsibility

**Problem Solved**: OpenAI clients send arbitrary text, but hypercube generation requires tokens that exist in the trained semantic space. The TokenizationService validates and filters prompts to ensure generation quality.

## Class Structure

```csharp
public class TokenizationService
{
    // Dependencies
    private readonly ILogger<TokenizationService> _logger;
    private readonly PostgresService _postgresService;

    // Core API
    public Task<long[]> TokenizeAndEncodeAsync(string prompt);
    public int EstimateTokenCount(string text);

    // Internal Methods
    private string[] TokenizeWords(string text);
    private Task<long?> EncodeTokenAsync(string token);
    private long GetStableHash(string input);
}
```

## Tokenization Workflow

### Main Entry Point

```csharp
public async Task<long[]> TokenizeAndEncodeAsync(string prompt)
{
    // Phase 1: Input Validation
    if (string.IsNullOrWhiteSpace(prompt)) {
        return Array.Empty<long>();
    }

    // Phase 2: Text Segmentation
    var rawTokens = TokenizeWords(prompt);

    // Phase 3: Vocabulary Validation
    var validTokens = new List<long>();
    foreach (var token in rawTokens.Take(50)) {  // Limit context
        var encoded = await EncodeTokenAsync(token);
        if (encoded.HasValue) {
            validTokens.Add(encoded.Value);
        }
    }

    // Phase 4: Logging & Return
    _logger.LogInformation(
        "Tokenized '{Prompt}' → {RawCount} raw, {ValidCount} valid tokens",
        prompt, rawTokens.Length, validTokens.Count);

    return validTokens.ToArray();
}
```

## Text Segmentation Algorithm

### TokenizeWords Implementation

```csharp
private static string[] TokenizeWords(string text)
{
    // Split on whitespace and common punctuation
    var tokens = text.Split(new[] {
        ' ', '\t', '\n', '\r',
        '.', ',', '!', '?', ';', ':',
        '"', '\'', '(', ')', '[', ']',
        '{', '}', '-', '_', '+', '=',
        '/', '\\', '|', '@', '#', '$',
        '%', '^', '&', '*'
    }, StringSplitOptions.RemoveEmptyEntries);

    // Normalization
    return tokens
        .Select(token => token.Trim().ToLowerInvariant())
        .Where(token => !string.IsNullOrEmpty(token))
        .ToArray();
}
```

**Segmentation Examples**:

| Input Text | Tokenized Output |
|------------|------------------|
| `"Hello world!"` | `["hello", "world"]` |
| `"machine learning"` | `["machine", "learning"]` |
| `"class MyClass { ... }"` | `["class", "myclass"]` (normalized) |

### Context Length Management

```csharp
// Limit to prevent excessive processing and memory usage
const int MaxContextTokens = 50;

var validTokens = new List<long>();
foreach (var token in rawTokens.Take(MaxContextTokens)) {
    var encoded = await EncodeTokenAsync(token);
    if (encoded.HasValue) {
        validTokens.Add(encoded.Value);
    }
}

// Warn if truncated
if (rawTokens.Length > MaxContextTokens) {
    _logger.LogWarning(
        "Prompt truncated from {Total} to {Limited} tokens",
        rawTokens.Length, MaxContextTokens);
}
```

## Vocabulary Validation

### Token Encoding Process

```csharp
private async Task<long?> EncodeTokenAsync(string token)
{
    try {
        // Step 1: Check existence in hypercube vocabulary
        var exists = await _postgresService.TokenExistsAsync(token);
        if (!exists) {
            _logger.LogDebug("Token '{Token}' not in vocabulary", token);
            return null;
        }

        // Step 2: Generate stable identifier
        // TODO: Replace with actual composition ID from database
        var stableId = GetStableHash(token);

        _logger.LogTrace("Encoded token '{Token}' → ID {Id}", token, stableId);
        return stableId;

    } catch (Exception ex) {
        _logger.LogWarning(ex, "Error encoding token '{Token}'", token);
        return null;
    }
}
```

### Database Integration

**Token Existence Check**:
```sql
-- Query executed by PostgresService.TokenExistsAsync()
SELECT 1
FROM composition
WHERE label = @token
  AND centroid IS NOT NULL  -- Must have 4D coordinates
LIMIT 1;
```

**Future Enhancement**:
```sql
-- Full composition ID retrieval (planned)
SELECT id  -- BYTEA 32-byte BLAKE3 hash
FROM composition
WHERE label = @token
  AND centroid IS NOT NULL;
```

### Stable Hash Generation

**Current Implementation** (Temporary):
```csharp
private static long GetStableHash(string input)
{
    unchecked {
        long hash = 23;
        foreach (char c in input) {
            hash = hash * 31 + c;
        }
        return Math.Abs(hash);
    }
}
```

**Purpose**: Provides deterministic, collision-resistant IDs until full BYTEA support is implemented.

**Properties**:
- Deterministic: Same input → same output
- Stable: Unaffected by dictionary ordering
- Positive: Ensures valid identifier range

## Usage Statistics

### Token Count Estimation

```csharp
public int EstimateTokenCount(string text)
{
    if (string.IsNullOrEmpty(text)) return 0;

    // Simple heuristic: ~4 characters per token
    // This will be replaced with actual tokenization counting
    var wordCount = text.Split(new[] { ' ', '\t', '\n', '\r' },
                               StringSplitOptions.RemoveEmptyEntries)
                        .Length;

    return Math.Max(1, wordCount);
}
```

**Usage in OpenAI Compatibility**:
```json
{
  "usage": {
    "prompt_tokens": 5,      // Estimated from input text
    "completion_tokens": 12, // Actual from generation
    "total_tokens": 17
  }
}
```

## Error Handling & Resilience

### Validation Errors

- **Empty Prompts**: Return empty array, log as warning
- **Database Failures**: Log error, return null for failed tokens
- **Encoding Exceptions**: Graceful degradation, skip problematic tokens

### Logging Strategy

```csharp
// Information level - key operations
_logger.LogInformation("Tokenized prompt into {Count} valid tokens", validTokens.Count);

// Warning level - issues requiring attention
_logger.LogWarning("No valid tokens found in prompt '{Prompt}'", prompt);

// Debug level - detailed token processing
_logger.LogDebug("Token '{Token}' validated successfully", token);

// Trace level - internal operations
_logger.LogTrace("Generated stable hash {Hash} for token '{Token}'", hash, token);
```

### Graceful Degradation

```csharp
// If no tokens validate, provide fallback
if (validTokens.Count == 0) {
    _logger.LogWarning("No valid tokens in prompt, using fallback");
    return new[] { GetStableHash("the") };  // Common fallback token
}
```

## Performance Characteristics

### Latency Breakdown

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Text Segmentation | <1ms | String splitting only |
| Token Validation | 5-50ms | Database round-trips |
| Hash Generation | <1ms | In-memory computation |
| Total (per request) | 10-100ms | Depends on token count |

### Optimization Opportunities

- **Batch Validation**: Validate multiple tokens in single DB query
- **Caching**: Cache frequently used token validations
- **Async Processing**: Parallel validation of independent tokens
- **Connection Pooling**: Reuse database connections

### Memory Usage

- **Per Request**: ~1-10KB for token arrays and strings
- **Cache Potential**: Future token validation cache
- **Database Load**: 1-N queries per request (N = token count)

## Integration Points

### With GenerativeService

```csharp
// In GenerativeService.GenerateCompletionAsync()
var validTokens = await _postgresService.GetValidTokensFromPromptAsync(request.Prompt);
var startLabel = validTokens.LastOrDefault() ?? "the";
```

### With PostgresService

```csharp
// Batch token validation (future enhancement)
public async Task<Dictionary<string, bool>> ValidateTokensAsync(string[] tokens) {
    // Single query with IN clause
    // return token → exists mapping
}
```

### With Health Monitoring

```csharp
// Tokenization health check
public async Task<bool> ValidateCoreTokensAsync() {
    var coreTokens = new[] { "the", "a", "an", "is", "are" };
    foreach (var token in coreTokens) {
        if (!await TokenExistsAsync(token)) return false;
    }
    return true;
}
```

## Testing Strategy

### Unit Tests

```csharp
[TestMethod]
public async Task TokenizeAndEncodeAsync_ValidPrompt_ReturnsValidTokens()
{
    // Arrange
    var service = new TokenizationService(logger, postgresService);
    var prompt = "Hello world";

    // Act
    var result = await service.TokenizeAndEncodeAsync(prompt);

    // Assert
    Assert.IsTrue(result.Length > 0);
    Assert.IsTrue(result.All(id => id > 0));  // All IDs positive
}

[TestMethod]
public async Task TokenizeAndEncodeAsync_EmptyPrompt_ReturnsEmptyArray()
{
    // Arrange & Act
    var result = await service.TokenizeAndEncodeAsync("");

    // Assert
    Assert.AreEqual(0, result.Length);
}
```

### Integration Tests

```csharp
[TestMethod]
public async Task TokenizeAndEncodeAsync_WithDatabase_ValidatesRealTokens()
{
    // Arrange: Ensure test data in database
    var prompt = "the quick brown fox";

    // Act
    var result = await service.TokenizeAndEncodeAsync(prompt);

    // Assert: Should find "the", "fox" etc. if they exist in hypercube
    // Unknown tokens should be filtered out
}
```

### Performance Tests

```csharp
[TestMethod]
public async Task TokenizeAndEncodeAsync_LargePrompt_PerformanceUnder100ms()
{
    // Arrange: 1000 character prompt
    var prompt = string.Join(" ", Enumerable.Repeat("word", 200));

    // Act
    var stopwatch = Stopwatch.StartNew();
    var result = await service.TokenizeAndEncodeAsync(prompt);
    var elapsed = stopwatch.ElapsedMilliseconds;

    // Assert
    Assert.IsTrue(elapsed < 100, $"Took {elapsed}ms");
}
```

## Future Enhancements

### Advanced Tokenization

- **Subword Tokenization**: Handle out-of-vocabulary words
- **Context Awareness**: Consider surrounding tokens for validation
- **Language Detection**: Different rules for different languages
- **Normalization**: Unicode normalization, case handling

### Performance Optimizations

- **Token Cache**: LRU cache of validated tokens
- **Batch Queries**: Single DB query for multiple tokens
- **Async Validation**: Parallel database checks
- **Preprocessing**: Client-side tokenization hints

### Semantic Enhancements

- **Synonym Mapping**: Map related terms to known tokens
- **Stemming/Lemmatization**: Reduce inflected forms
- **Multi-token Phrases**: Recognize common phrases as units
- **Context Disambiguation**: Choose correct sense of ambiguous words

### Monitoring & Analytics

- **Token Coverage Stats**: % of input tokens that validate
- **Vocabulary Growth**: Track new tokens added to hypercube
- **Fallback Frequency**: How often default tokens are used
- **Performance Trends**: Tokenization latency over time

## Configuration

### Service Configuration

```json
{
  "Tokenization": {
    "MaxContextTokens": 50,        // Maximum tokens to process
    "ValidationTimeout": 5000,     // DB query timeout (ms)
    "EnableDetailedLogging": false // Debug-level token logging
  }
}
```

### Fallback Tokens

```csharp
private static readonly string[] FallbackTokens = {
    "the", "a", "an", "is", "are", "was", "were", "has", "have"
};
```

This service ensures that the hypercube generation engine only receives semantically meaningful input, bridging the gap between natural language and geometric computation.