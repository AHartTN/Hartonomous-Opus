# OpenAI API Endpoints Overview

This document provides an overview of the OpenAI API endpoints that are relevant for local/self-hosted model implementations. Excluded are audio and image processing endpoints as they are not required for the current project.

## Core Endpoints

### Text Generation and Conversation
- **Chat Completions** (`POST /v1/chat/completions`): Generate conversational responses using chat-based models
- **Completions (Legacy)** (`POST /v1/completions`): Legacy text completion endpoint for older models

### Embeddings
- **Embeddings** (`POST /v1/embeddings`): Generate vector embeddings for text

### Content Moderation
- **Moderations** (`POST /v1/moderations`): Check content for policy violations and safety

## Advanced Features

### Assistants
- **Assistants** (`POST /v1/assistants`): Create and manage AI assistants with persistent state and tools

### Batch Processing
- **Batch** (`POST /v1/batch`): Process multiple API requests asynchronously

### Model Customization
- **Fine-tuning** (`POST /v1/fine-tuning`): Fine-tune models on custom datasets

### Real-time Interactions
- **Realtime** (`POST /v1/realtime`): Real-time streaming API interactions

### Responses API
- **Responses** (`POST /v1/responses`): Advanced response handling and formatting (may include conversational features)

## Model Management
- **Models** (`GET /v1/models`): List available models and their capabilities

## Notes
- All endpoints follow RESTful conventions
- Authentication is handled via API keys in headers
- Request/response formats are JSON-based
- Streaming responses are supported where applicable
- Error handling follows standard HTTP status codes

For detailed documentation on each endpoint, refer to the individual endpoint documentation files.