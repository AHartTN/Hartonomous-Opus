# Realtime API

The Realtime API endpoints are currently not implemented in the local self-hosted gateway. These endpoints would provide real-time, low-latency streaming capabilities for interactive AI applications.

## Endpoints

All Realtime API endpoints return `501 Not Implemented`:

- `POST /v1/realtime` - Establish realtime session

## Status: Not Implemented

**Note**: All Realtime API endpoints return a `501 Not Implemented` error. Real-time streaming is not supported in the local llama.cpp gateway implementation.

## Description (Reference Only)

The Realtime API enables low-latency, bidirectional communication for real-time AI interactions, supporting:

- **Audio Streaming**: Real-time speech input/output
- **Text Streaming**: Ultra-low latency text generation
- **Interactive Sessions**: Persistent connections for continuous conversation
- **Function Calling**: Real-time tool execution during conversations

**Local Implementation Challenges**: Real-time APIs require WebSocket support, low-latency infrastructure, and specialized audio processing capabilities not available in standard REST-based inference servers.

## Current Implementation Response

All Realtime API calls return:

```json
{
  "detail": "Realtime API is not implemented in the local llama.cpp gateway"
}
```

## Alternative Approaches for Real-time Applications

### WebSocket Integration

```python
import asyncio
import websockets
import json
from typing import Dict, Any

class LocalRealtimeClient:
    def __init__(self, gateway_url: str, api_key: str):
        self.gateway_url = gateway_url.replace("http", "ws")
        self.api_key = api_key

    async def connect(self):
        """Establish WebSocket connection"""
        uri = f"{self.gateway_url}/realtime"
        async with websockets.connect(uri) as websocket:
            # Send authentication
            await websocket.send(json.dumps({
                "type": "auth",
                "api_key": self.api_key
            }))

            await self.handle_messages(websocket)

    async def handle_messages(self, websocket):
        """Handle incoming messages"""
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "text_delta":
                print(f"Received: {data['delta']}", end="")
            elif data["type"] == "audio_chunk":
                # Process audio data
                self.process_audio(data["audio"])
            elif data["type"] == "function_call":
                result = await self.execute_function(data["function"])
                await websocket.send(json.dumps({
                    "type": "function_result",
                    "call_id": data["call_id"],
                    "result": result
                }))

    async def send_text(self, websocket, text: str):
        """Send text input"""
        await websocket.send(json.dumps({
            "type": "text_input",
            "content": text
        }))

    async def send_audio(self, websocket, audio_data: bytes):
        """Send audio input"""
        await websocket.send(json.dumps({
            "type": "audio_input",
            "audio": audio_data.hex()
        }))

    async def execute_function(self, function_data: Dict[str, Any]) -> Any:
        """Execute a function call"""
        # Implement function execution logic
        func_name = function_data["name"]
        args = function_data["arguments"]

        if func_name == "get_weather":
            return {"temperature": 72, "condition": "sunny"}
        # Add more function implementations

        return {"error": "Function not implemented"}
```

### HTTP Streaming Optimization

```python
import requests
import json
from sseclient import SSEClient  # Server-Sent Events client

class StreamingClient:
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream"
        }

    def stream_completion(self, prompt: str, **kwargs):
        """Stream completion with low latency"""
        payload = {
            "prompt": prompt,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs
        }

        response = requests.post(
            f"{self.endpoint}/v1/completions",
            json=payload,
            headers=self.headers,
            stream=True
        )

        client = SSEClient(response)
        for event in client.events():
            if event.data:
                data = json.loads(event.data)
                if data["choices"][0]["finish_reason"] is None:
                    yield data["choices"][0]["text"]
                else:
                    break

    def stream_chat(self, messages: list, **kwargs):
        """Stream chat completion"""
        payload = {
            "messages": messages,
            "stream": True,
            **kwargs
        }

        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            stream=True
        )

        client = SSEClient(response)
        for event in client.events():
            if event.data != "[DONE]":
                data = json.loads(event.data)
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
```

### Real-time Audio Processing

```python
import pyaudio
import numpy as np
from typing import Callable

class AudioStreamer:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self, callback: Callable[[bytes], None]):
        """Start recording audio and call callback with chunks"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        def audio_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (in_data, pyaudio.paContinue)

        self.stream.start_stream()

    def stop_recording(self):
        """Stop recording"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def play_audio(self, audio_data: bytes):
        """Play audio data"""
        output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True
        )
        output_stream.write(audio_data)
        output_stream.close()
```

## Example Real-time Application

```python
import asyncio

class RealtimeChatApp:
    def __init__(self, gateway_url: str, api_key: str):
        self.client = StreamingClient(gateway_url, api_key)
        self.audio_streamer = AudioStreamer()
        self.conversation_history = []

    async def interactive_chat(self):
        """Run interactive chat session"""
        print("Starting real-time chat. Type 'quit' to exit.")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break

            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})

            # Stream response
            print("AI: ", end="", flush=True)
            full_response = ""
            for chunk in self.client.stream_chat(self.conversation_history):
                print(chunk, end="", flush=True)
                full_response += chunk

            print()  # New line

            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": full_response})

    def voice_chat_mode(self):
        """Enable voice input/output (requires additional setup)"""
        print("Voice chat mode - recording...")

        audio_chunks = []

        def audio_callback(chunk):
            audio_chunks.append(chunk)
            # Process audio chunk (speech-to-text, etc.)

        self.audio_streamer.start_recording(audio_callback)

        # Process collected audio
        # This would integrate with speech recognition and synthesis
```

## Performance Considerations

### Latency Optimization

- **Model Quantization**: Use quantized models for faster inference
- **GPU Acceleration**: Ensure GPU utilization for parallel processing
- **Connection Pooling**: Maintain persistent connections
- **Caching**: Cache frequent computations

### Scalability

- **Load Balancing**: Distribute requests across multiple model instances
- **Queue Management**: Handle request bursts gracefully
- **Resource Monitoring**: Track memory and compute usage

The current gateway supports HTTP streaming for real-time text generation. Full real-time API capabilities would require WebSocket infrastructure and audio processing pipelines.