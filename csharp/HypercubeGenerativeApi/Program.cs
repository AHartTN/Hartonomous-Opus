using HypercubeGenerativeApi.Services;
using HypercubeGenerativeApi.Interop;
using HypercubeGenerativeApi.Controllers;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Swashbuckle.AspNetCore.SwaggerGen;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Add health checks
builder.Services.AddHealthChecks()
    .AddCheck<GenerativeHealthCheck>("generative", HealthStatus.Degraded, new[] { "hypercube" });

// Register application services
builder.Services.AddSingleton<GenerativeService>();
builder.Services.AddSingleton<PostgresService>();
builder.Services.AddSingleton<TokenizationService>();

// Configure JSON options for OpenAI compatibility
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = null; // Preserve property names
    options.SerializerOptions.WriteIndented = false;
});

// Configure logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

// Build the application
var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

// Map health check endpoint
app.MapHealthChecks("/health");

// Map API endpoints - OpenAI compatibility (legacy)
app.MapGet("/v1/models", ModelsEndpoints.ListModels);
app.MapPost("/v1/completions", CompletionsEndpoints.CreateCompletion);

// Map API endpoints - Hypercube semantic operations (primary)
app.MapPost("/query/semantic", SemanticQueryController.QuerySemantic);
app.MapPost("/query/analogies", SemanticQueryController.QueryAnalogies);
app.MapPost("/query/relationships", SemanticQueryController.QueryRelationships);

// Map API endpoints - 4D geometric operations (spatial intelligence)
app.MapPost("/geometric/neighbors", GeometricController.FindNeighbors);
app.MapPost("/geometric/centroid", GeometricController.CalculateCentroid);
app.MapPost("/geometric/distance", GeometricController.MeasureDistance);
app.MapGet("/geometric/visualize/{entity}", GeometricController.VisualizeCoordinates);

// Map API endpoints - Cross-content analysis (universal intelligence)
app.MapPost("/analyze/overlap", AnalysisController.AnalyzeOverlap);
app.MapPost("/analyze/concepts", AnalysisController.AnalyzeConcepts);
app.MapPost("/analyze/relationships", AnalysisController.AnalyzeRelationships);

// Map API endpoints - Content ingestion (revolutionary core)
app.MapPost("/ingest/document", IngestionController.IngestDocument);
app.MapPost("/ingest/codebase", IngestionController.IngestCodebase);
app.MapGet("/ingest/status/{id}", IngestionController.GetIngestionStatus);
app.MapGet("/ingest/stats", IngestionController.GetIngestionStats);

// Global exception handler
app.UseExceptionHandler(exceptionHandlerApp =>
{
    exceptionHandlerApp.Run(async context =>
    {
        var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>()?.Error;

        context.Response.StatusCode = StatusCodes.Status500InternalServerError;
        context.Response.ContentType = "application/json";

        var errorResponse = new
        {
            error = new
            {
                message = "An internal error occurred",
                type = "internal_error",
                details = app.Environment.IsDevelopment() ? exception?.Message : null
            }
        };

        await context.Response.WriteAsJsonAsync(errorResponse);
    });
});

// Initialize services on startup
var generativeService = app.Services.GetRequiredService<GenerativeService>();
var postgresService = app.Services.GetRequiredService<PostgresService>();

try
{
    await postgresService.InitializeAsync();
    await generativeService.InitializeAsync();
    app.Logger.LogInformation("Hypercube Generative API initialized successfully");
}
catch (Exception ex)
{
    app.Logger.LogError(ex, "Failed to initialize Hypercube services");
    throw;
}

app.Run();

// Make Program class public for testing
public partial class Program { }