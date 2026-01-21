#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/cpp/build/lib/Release
dotnet test csharp/HypercubeGenerativeApi/HypercubeGenerativeApiTests.csproj
