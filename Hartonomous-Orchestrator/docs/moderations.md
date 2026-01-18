# Moderations API

The Moderations API endpoint is currently not implemented in the local self-hosted gateway. This endpoint would check content for policy violations and safety concerns, but is not available when using local llama.cpp models.

## Endpoint

```
POST /v1/moderations
```

## Status: Not Implemented

**Note**: This endpoint returns a `501 Not Implemented` error. Content moderation is not supported in the local llama.cpp gateway implementation.

## Description (Reference Only)

The Moderations API is designed to check whether content violates OpenAI's usage policies. It analyzes text inputs and returns classification results indicating whether the content violates various safety categories.

**Local Implementation**: Content moderation requires access to specialized classification models that are not typically available in standard llama.cpp deployments. For local self-hosted solutions, users should implement their own moderation logic or use external moderation services.

## Request Parameters (Reference)

### Required Parameters

- **`input`** (string or array of strings): The content to check for violations

### Optional Parameters

- **`model`** (string): The moderation model to use (not applicable in local implementation)

## Response Format (Reference)

```json
{
  "id": "modr-123",
  "model": "text-moderation-latest",
  "results": [
    {
      "flagged": false,
      "categories": {
        "sexual": false,
        "hate": false,
        "harassment": false,
        "self-harm": false,
        "sexual/minors": false,
        "hate/threatening": false,
        "violence/graphic": false,
        "self-harm/intent": false,
        "self-harm/instructions": false,
        "harassment/threatening": false,
        "violence": false
      },
      "category_scores": {
        "sexual": 0.0001,
        "hate": 0.0001,
        "harassment": 0.0001,
        "self-harm": 0.0001,
        "sexual/minors": 0.0001,
        "hate/threatening": 0.0001,
        "violence/graphic": 0.0001,
        "self-harm/intent": 0.0001,
        "self-harm/instructions": 0.0001,
        "harassment/threatening": 0.0001,
        "violence": 0.0001
      }
    }
  ]
}
```

## Current Implementation Response

When called, this endpoint returns:

```json
{
  "detail": "Content moderation is not implemented in the local llama.cpp gateway"
}
```

With HTTP status code `501 Not Implemented`.

## Alternative Approaches for Local Deployments

### Custom Moderation Implementation

For local deployments requiring content moderation, consider:

1. **External Services**: Route moderation requests to cloud-based moderation APIs
2. **Local Models**: Deploy specialized moderation models (e.g., using transformers library)
3. **Rule-Based Filtering**: Implement keyword-based or pattern-based content filtering
4. **Hybrid Approach**: Combine local rule-based filtering with periodic cloud moderation checks

### Example Custom Moderation Integration

```python
import requests
from typing import List, Dict

class LocalModerationService:
    def __init__(self, gateway_url: str, api_key: str):
        self.gateway_url = gateway_url
        self.api_key = api_key

    def check_content(self, content: str) -> Dict:
        # This would be a custom implementation
        # For now, return safe response
        return {
            "flagged": False,
            "categories": {
                "inappropriate": False,
                "spam": False,
                "harmful": False
            }
        }

    async def moderate_texts(self, texts: List[str]) -> List[Dict]:
        results = []
        for text in texts:
            result = self.check_content(text)
            results.append(result)
        return results

# Usage
moderation_service = LocalModerationService("http://localhost:8000", "your-api-key")
results = await moderation_service.moderate_texts(["some text", "another text"])
```

## Future Implementation Considerations

If moderation support is added to the local gateway:

- **Model Requirements**: Would need access to fine-tuned moderation models
- **Performance Impact**: Moderation adds latency to content generation pipelines
- **Accuracy Trade-offs**: Local models may have different accuracy characteristics than cloud models
- **Privacy Benefits**: Local moderation keeps sensitive content on-premises

## Compliance and Legal Notes

- **No Warranty**: The absence of built-in moderation does not constitute a warranty of content safety
- **User Responsibility**: Deployers are responsible for implementing appropriate content policies
- **Regulatory Compliance**: Ensure local implementations meet relevant legal requirements for content moderation
- **Transparency**: Clearly document moderation capabilities and limitations to users