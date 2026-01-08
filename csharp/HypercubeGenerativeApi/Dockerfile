# Multi-stage build for C# API with C++ native library

# Stage 1: Build C++ native library (assuming source is available)
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS cpp-builder
WORKDIR /cpp

# Copy C++ source and build scripts
# NOTE: This assumes C++ source is available in the build context
# In production, you would COPY the pre-built hypercube_generative.dll instead
# COPY ["path/to/cpp/source", "./"]
# RUN cmake . && make

# For now, assume the DLL is provided in the build context
# COPY ["hypercube_generative.dll", "/app/native/"]

# Stage 2: Build C# application
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dotnet-builder
WORKDIR /src

# Copy and restore C# project
COPY ["HypercubeGenerativeApi.csproj", "./"]
RUN dotnet restore "HypercubeGenerativeApi.csproj"

# Copy source and build
COPY . .
RUN dotnet build "HypercubeGenerativeApi.csproj" -c Release -o /app/build

# Publish the application
FROM dotnet-builder AS publish
RUN dotnet publish "HypercubeGenerativeApi.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Stage 3: Runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# Install runtime dependencies for PostgreSQL and native interop
RUN apt-get update && apt-get install -y \
    libc6-dev \
    libicu-dev \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy published .NET application
COPY --from=publish /app/publish .

# Copy native C++ library (must be provided in build context)
# TODO: Automate C++ build in multi-stage Dockerfile
COPY ["hypercube_generative.dll", "./"]

# Create non-root user for security
RUN groupadd -r hypercube && useradd -r -g hypercube hypercube
RUN chown -R hypercube:hypercube /app
USER hypercube

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Environment variables
ENV ASPNETCORE_URLS=http://*:5000
ENV ASPNETCORE_ENVIRONMENT=Production
ENV DOTNET_gcServer=1
ENV DOTNET_GCDynamicAdaptationMode=1

# Run the application
ENTRYPOINT ["dotnet", "HypercubeGenerativeApi.dll"]