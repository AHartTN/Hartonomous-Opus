"""
Moderations API routes
"""
import uuid
import time
from typing import List, Union
import logging

from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from ..models import ModerationRequest, ModerationResponse, ModerationResult, ModerationCategories, ModerationCategoryScores

logger = logging.getLogger(__name__)

router = APIRouter()

# Keyword lists for each category
SEXUAL_KEYWORDS = [
    "sex", "porn", "nude", "naked", "erotic", "sexual", "fuck", "pornography",
    "adult", "xxx", "nsfw", "lust", "orgasm", "penis", "vagina", "breast"
]

HATE_KEYWORDS = [
    "hate", "racist", "nazi", "supremacist", "bigot", "intolerant", "discrimination",
    "prejudice", "xenophobia", "homophobia", "antisemitism", "islamophobia"
]

HARASSMENT_KEYWORDS = [
    "harass", "bully", "intimidate", "threaten", "abuse", "stalk", "cyberbully",
    "troll", "insult", "mock", "ridicule"
]

SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "self-harm", "cut myself", "overdose", "hang myself"
]

SEXUAL_MINORS_KEYWORDS = [
    "child porn", "pedophile", "underage", "minor sex", "child abuse", "loli", "shota"
]

HATE_THREATENING_KEYWORDS = [
    "kill all", "exterminate", "genocide", "ethnic cleansing", "hate crime"
]

VIOLENCE_GRAPHIC_KEYWORDS = [
    "murder", "rape", "torture", "blood", "gore", "mutilation", "dismember",
    "decapitation", "cannibalism"
]

SELF_HARM_INTENT_KEYWORDS = [
    "i want to die", "i'm going to kill myself", "planning suicide"
]

SELF_HARM_INSTRUCTIONS_KEYWORDS = [
    "how to commit suicide", "suicide methods", "ways to self-harm"
]

HARASSMENT_THREATENING_KEYWORDS = [
    "i'll kill you", "i'm coming for you", "death threat", "bomb threat"
]

VIOLENCE_KEYWORDS = [
    "kill", "murder", "attack", "assault", "violent", "fight", "weapon", "gun",
    "knife", "bomb", "explosion", "terrorism"
]

KEYWORD_CATEGORIES = {
    "sexual": SEXUAL_KEYWORDS,
    "hate": HATE_KEYWORDS,
    "harassment": HARASSMENT_KEYWORDS,
    "self_harm": SELF_HARM_KEYWORDS,
    "sexual_minors": SEXUAL_MINORS_KEYWORDS,
    "hate_threatening": HATE_THREATENING_KEYWORDS,
    "violence_graphic": VIOLENCE_GRAPHIC_KEYWORDS,
    "self_harm_intent": SELF_HARM_INTENT_KEYWORDS,
    "self_harm_instructions": SELF_HARM_INSTRUCTIONS_KEYWORDS,
    "harassment_threatening": HARASSMENT_THREATENING_KEYWORDS,
    "violence": VIOLENCE_KEYWORDS,
}

CATEGORY_ALIASES = {
    "sexual": "sexual",
    "hate": "hate",
    "harassment": "harassment",
    "self_harm": "self-harm",
    "sexual_minors": "sexual/minors",
    "hate_threatening": "hate/threatening",
    "violence_graphic": "violence/graphic",
    "self_harm_intent": "self-harm/intent",
    "self_harm_instructions": "self-harm/instructions",
    "harassment_threatening": "harassment/threatening",
    "violence": "violence",
}


def moderate_text(text: str) -> ModerationResult:
    """Moderate a single text input using keyword filtering."""
    text_lower = text.lower()
    categories = {}
    category_scores = {}
    flagged = False

    for category, keywords in KEYWORD_CATEGORIES.items():
        # Check if any keyword is in the text
        is_flagged = any(keyword in text_lower for keyword in keywords)
        alias = CATEGORY_ALIASES[category]
        categories[alias] = is_flagged
        category_scores[alias] = 1.0 if is_flagged else 0.0
        if is_flagged:
            flagged = True

    return ModerationResult(
        flagged=flagged,
        categories=ModerationCategories(**categories),
        category_scores=ModerationCategoryScores(**category_scores)
    )


@router.post("/v1/moderations")
async def moderations(
    request: ModerationRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible moderations API endpoint."""
    logger.info(f"Moderation request received: input_type={type(request.input)}, model={request.model}")

    # Handle input as string or list
    if isinstance(request.input, str):
        inputs = [request.input]
    elif isinstance(request.input, list):
        inputs = request.input
    else:
        raise HTTPException(status_code=400, detail="Input must be a string or list of strings")

    # Validate inputs
    if not inputs:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    for i, inp in enumerate(inputs):
        if not isinstance(inp, str):
            raise HTTPException(status_code=400, detail=f"Input[{i}] must be a string")

    # Perform moderation
    results = []
    for text in inputs:
        result = moderate_text(text)
        results.append(result)

    # Create response
    response = ModerationResponse(
        id=f"modr-{uuid.uuid4().hex[:20]}",
        model=request.model or "text-moderation-latest",
        results=results
    )

    logger.info(f"Moderation completed: flagged_any={any(r.flagged for r in results)}")
    return response