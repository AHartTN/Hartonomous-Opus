"""
Pydantic models for OpenAI Gateway API

This module defines the data models used for API request/response validation
and serialization, following OpenAI API specifications where applicable.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Union, Literal


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: The role of the message author (e.g., 'user', 'assistant', 'system')
        content: The content of the message (text or structured content blocks)
        name: Optional name for the message author
        tool_calls: Optional list of tool calls made by the assistant
        tool_call_id: Optional ID for tool call responses
    """
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    """Definition of a function that can be called by the model.

    Attributes:
        name: The name of the function
        description: Optional description of what the function does
        parameters: Optional JSON schema for the function parameters
        strict: Whether to enforce strict parameter validation
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = False


class Tool(BaseModel):
    """A tool that the model can use.

    Attributes:
        type: The type of tool (currently only "function" is supported)
        function: The function definition for this tool
    """
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ResponseFormat(BaseModel):
    """Format specification for model responses.

    Attributes:
        type: The response format type ("text" or "json_object")
    """
    type: Literal["text", "json_object"] = "text"


class ChatCompletionRequest(BaseModel):
    """Request model for chat completions API.

    This follows the OpenAI Chat Completions API specification with additional
    RAG-specific parameters.

    Attributes:
        model: The model to use for completion
        messages: List of chat messages
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        max_completion_tokens: Alternative to max_tokens
        stream: Whether to stream the response
        stop: Stop sequences for generation
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty for repetition
        presence_penalty: Presence penalty for repetition
        seed: Random seed for reproducible results
        logprobs: Whether to return log probabilities
        top_logprobs: Number of top log probabilities to return
        n: Number of completions to generate (not fully supported)
        user: User identifier for tracking
        tools: List of tools available to the model
        tool_choice: How to choose tools
        parallel_tool_calls: Whether to make parallel tool calls
        response_format: Format for the response
        rag_enabled: Whether to enable RAG (Retrieval-Augmented Generation)
        rag_top_k: Number of documents to retrieve for RAG
        rag_rerank_top_n: Number of documents to rerank for RAG
    """
    model: str = "qwen3-coder-30b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    n: Optional[int] = 1
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = True
    response_format: Optional[ResponseFormat] = None
    # RAG-specific
    rag_enabled: Optional[bool] = None
    rag_top_k: Optional[int] = None
    rag_rerank_top_n: Optional[int] = None


class EmbeddingRequest(BaseModel):
    """Request model for embeddings API.

    Attributes:
        model: The embedding model to use
        input: Text or list of texts to embed
        encoding_format: Format for the embeddings ("float" or "base64")
        dimensions: Optional dimensionality reduction
        user: User identifier for tracking
    """
    model: str = "qwen3-embedding-4b"
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class RerankRequest(BaseModel):
    """Request model for reranking API.

    Attributes:
        model: The reranking model to use
        query: The search query
        documents: List of documents to rerank
        top_n: Number of top results to return
    """
    model: str = "qwen3-reranker-4b"
    query: str
    documents: List[str]
    top_n: Optional[int] = None


class IngestRequest(BaseModel):
    """Request model for document ingestion API.

    Attributes:
        documents: List of documents to ingest
        metadata: Optional metadata for each document
        ids: Optional custom IDs for documents
        chunk: Whether to chunk documents before ingestion
    """
    documents: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    ids: Optional[List[str]] = None
    chunk: Optional[bool] = True


class SearchRequest(BaseModel):
    """Request model for vector search API.

    Attributes:
        query: The search query text
        top_k: Number of results to return
        filter: Optional filter conditions for search
    """
    query: str
    top_k: Optional[int] = 10
    filter: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    """Request model for text completions API.

    Attributes:
        model: The completion model to use
        prompt: Text prompt or list of prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        n: Number of completions per prompt
        stream: Whether to stream responses
        logprobs: Number of log probabilities to return
        echo: Whether to include the prompt in response
        stop: Stop sequences for generation
        presence_penalty: Presence penalty for repetition
        frequency_penalty: Frequency penalty for repetition
        best_of: Generate multiple completions and return best (not supported)
        logit_bias: Bias for specific tokens (not supported)
        user: User identifier for tracking
        suffix: Suffix to append (not supported)
    """
    model: str = "qwen3-coder-30b"
    prompt: Union[str, List[str]] = ""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    suffix: Optional[str] = None


class CompletionChoice(BaseModel):
    """A single completion choice in the response.

    Attributes:
        text: The generated text
        index: Index of this choice in the response
        logprobs: Log probabilities for the tokens (if requested)
        finish_reason: Reason why generation stopped
    """
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    """Token usage statistics for a completion request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens generated
        total_tokens: Total tokens used
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response model for text completions API.

    Attributes:
        id: Unique identifier for the response
        object: Object type ("text_completion")
        created: Unix timestamp of creation
        model: Model used for completion
        choices: List of completion choices
        usage: Token usage statistics
    """
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ModerationRequest(BaseModel):
    """Request model for moderations API.

    Attributes:
        input: Text or list of texts to moderate
        model: Optional moderation model (ignored in local implementation)
    """
    input: Union[str, List[str]]
    model: Optional[str] = "text-moderation-latest"


class ModerationCategories(BaseModel):
    """Categories for moderation results.

    Attributes:
        sexual: Sexual content
        hate: Hate speech
        harassment: Harassment
        self_harm: Self-harm content
        sexual_minors: Sexual content involving minors
        hate_threatening: Hate speech with threats
        violence_graphic: Graphic violence
        self_harm_intent: Intent to self-harm
        self_harm_instructions: Instructions for self-harm
        harassment_threatening: Threatening harassment
        violence: Violence
    """
    model_config = ConfigDict(populate_by_name=False)

    sexual: bool = Field(alias="sexual")
    hate: bool = Field(alias="hate")
    harassment: bool = Field(alias="harassment")
    self_harm: bool = Field(alias="self-harm")
    sexual_minors: bool = Field(alias="sexual/minors")
    hate_threatening: bool = Field(alias="hate/threatening")
    violence_graphic: bool = Field(alias="violence/graphic")
    self_harm_intent: bool = Field(alias="self-harm/intent")
    self_harm_instructions: bool = Field(alias="self-harm/instructions")
    harassment_threatening: bool = Field(alias="harassment/threatening")
    violence: bool = Field(alias="violence")


class ModerationCategoryScores(BaseModel):
    """Scores for moderation categories.

    Attributes:
        sexual: Score for sexual content
        hate: Score for hate speech
        harassment: Score for harassment
        self_harm: Score for self-harm content
        sexual_minors: Score for sexual content involving minors
        hate_threatening: Score for hate speech with threats
        violence_graphic: Score for graphic violence
        self_harm_intent: Score for intent to self-harm
        self_harm_instructions: Score for instructions for self-harm
        harassment_threatening: Score for threatening harassment
        violence: Score for violence
    """
    model_config = ConfigDict(populate_by_name=False)

    sexual: float = Field(alias="sexual")
    hate: float = Field(alias="hate")
    harassment: float = Field(alias="harassment")
    self_harm: float = Field(alias="self-harm")
    sexual_minors: float = Field(alias="sexual/minors")
    hate_threatening: float = Field(alias="hate/threatening")
    violence_graphic: float = Field(alias="violence/graphic")
    self_harm_intent: float = Field(alias="self-harm/intent")
    self_harm_instructions: float = Field(alias="self-harm/instructions")
    harassment_threatening: float = Field(alias="harassment/threatening")
    violence: float = Field(alias="violence")


class ModerationResult(BaseModel):
    """Individual moderation result.

    Attributes:
        flagged: Whether the content was flagged
        categories: Category flags
        category_scores: Category scores
    """
    flagged: bool
    categories: ModerationCategories
    category_scores: ModerationCategoryScores


class ModerationResponse(BaseModel):
    """Response model for moderations API.

    Attributes:
        id: Unique identifier for the response
        model: Model used for moderation
        results: List of moderation results
    """
    id: str
    model: str
    results: List[ModerationResult]


class ExpiresAfter(BaseModel):
    """Expiration policy for files.

    Attributes:
        anchor: Reference point for expiration
        seconds: Number of seconds until expiration
    """
    anchor: str = "created_at"
    seconds: int


class FileObject(BaseModel):
    """File object model following OpenAI API specification.

    Attributes:
        id: Unique identifier for the file
        object: Object type, always "file"
        bytes: Size of the file in bytes
        created_at: Unix timestamp of creation
        expires_at: Unix timestamp of expiration, if applicable
        filename: Name of the file
        purpose: Intended purpose of the file
    """
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    expires_at: Optional[int] = None
    filename: str
    purpose: str


class FileListResponse(BaseModel):
    """Response model for file listing.

    Attributes:
        object: Object type, always "list"
        data: List of file objects
        first_id: ID of the first file in the list
        last_id: ID of the last file in the list
        has_more: Whether there are more files to retrieve
    """
    object: str = "list"
    data: List[FileObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class FileDeleteResponse(BaseModel):
    """Response model for file deletion.

    Attributes:
        id: ID of the deleted file
        object: Object type, always "file"
        deleted: Whether the file was deleted successfully
    """
    id: str
    object: str = "file"
    deleted: bool


class FineTuningHyperparameters(BaseModel):
    """Hyperparameters for fine-tuning jobs.

    Attributes:
        n_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate_multiplier: Learning rate multiplier
    """
    n_epochs: int = Field(default=3, ge=1, le=50)
    batch_size: Optional[int] = Field(default=1, ge=1, le=256)
    learning_rate_multiplier: Optional[float] = Field(default=2.0, gt=0, le=10)


class FineTuningJobRequest(BaseModel):
    """Request model for creating a fine-tuning job.

    Attributes:
        model: The base model to fine-tune
        training_file: ID of the uploaded training file
        hyperparameters: Optional hyperparameters for fine-tuning
        suffix: Optional suffix for the fine-tuned model name
        validation_file: Optional ID of the validation file
    """
    model: str
    training_file: str
    hyperparameters: Optional[FineTuningHyperparameters] = None
    suffix: Optional[str] = Field(None, max_length=40)
    validation_file: Optional[str] = None


class FineTuningJob(BaseModel):
    """Fine-tuning job object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the job
        object: Object type, always "fine_tuning.job"
        model: Base model being fine-tuned
        created_at: Unix timestamp of creation
        finished_at: Unix timestamp when job finished (null if not finished)
        fine_tuned_model: Name of the resulting fine-tuned model (null if not completed)
        organization_id: Organization ID (not used in local implementation)
        result_files: List of result file IDs (not used in local implementation)
        status: Current job status
        validation_file: Validation file ID (optional)
        training_file: Training file ID
        hyperparameters: Hyperparameters used for training
        trained_tokens: Number of tokens trained (null if not completed)
        error: Error details if job failed (null if successful)
    """
    id: str
    object: str = "fine_tuning.job"
    model: str
    created_at: int
    finished_at: Optional[int] = None
    fine_tuned_model: Optional[str] = None
    organization_id: Optional[str] = None
    result_files: List[str] = []
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    validation_file: Optional[str] = None
    training_file: str
    hyperparameters: FineTuningHyperparameters
    trained_tokens: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


class FineTuningJobList(BaseModel):
    """Response model for listing fine-tuning jobs.

    Attributes:
        object: Object type, always "list"
        data: List of fine-tuning jobs
        has_more: Whether there are more jobs to retrieve
        first_id: ID of the first job in the list
        last_id: ID of the last job in the list
    """
    object: str = "list"
    data: List[FineTuningJob]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class FineTuningJobEvent(BaseModel):
    """Fine-tuning job event object.

    Attributes:
        id: Unique identifier for the event
        object: Object type, always "fine_tuning.job.event"
        created_at: Unix timestamp of event creation
        level: Severity level of the event
        message: Event message
    """
    id: str
    object: str = "fine_tuning.job.event"
    created_at: int
    level: Literal["info", "warn", "error"]
    message: str


class FineTuningJobEventsList(BaseModel):
    """Response model for listing fine-tuning job events.

    Attributes:
        object: Object type, always "list"
        data: List of job events
        has_more: Whether there are more events to retrieve
    """
    object: str = "list"
    data: List[FineTuningJobEvent]
    has_more: bool = False


class CodeInterpreterTool(BaseModel):
    """Code interpreter tool configuration.

    Attributes:
        type: Tool type, always "code_interpreter"
    """
    type: Literal["code_interpreter"] = "code_interpreter"


class FileSearchTool(BaseModel):
    """File search tool configuration.

    Attributes:
        type: Tool type, always "file_search"
        file_search: Optional file search configuration
    """
    type: Literal["file_search"] = "file_search"
    file_search: Optional[Dict[str, Any]] = None


class AssistantTool(BaseModel):
    """Tool configuration for assistants.

    Attributes:
        type: Tool type ("code_interpreter" or "file_search")
    """
    type: Literal["code_interpreter", "file_search"]


class CreateAssistantRequest(BaseModel):
    """Request model for creating an assistant.

    Attributes:
        model: The model to use for the assistant
        name: Optional name for the assistant
        description: Optional description of the assistant
        instructions: Optional system instructions for the assistant
        tools: Optional list of tools enabled on the assistant
        file_ids: Optional list of file IDs attached to the assistant
        metadata: Optional metadata for the assistant
    """
    model: str
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    file_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Assistant(BaseModel):
    """Assistant object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the assistant
        object: Object type, always "assistant"
        created_at: Unix timestamp of creation
        name: Optional name of the assistant
        description: Optional description of the assistant
        model: Model used by the assistant
        instructions: Optional system instructions
        tools: List of tools enabled on the assistant
        file_ids: List of file IDs attached to the assistant
        metadata: Optional metadata for the assistant
    """
    id: str
    object: str = "assistant"
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: List[AssistantTool] = []
    file_ids: List[str] = []
    metadata: Optional[Dict[str, Any]] = None


class ModifyAssistantRequest(BaseModel):
    """Request model for modifying an assistant.

    Attributes:
        model: Optional new model for the assistant
        name: Optional new name for the assistant
        description: Optional new description
        instructions: Optional new instructions
        tools: Optional new list of tools
        file_ids: Optional new list of file IDs
        metadata: Optional new metadata
    """
    model: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    file_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AssistantListResponse(BaseModel):
    """Response model for listing assistants.

    Attributes:
        object: Object type, always "list"
        data: List of assistant objects
        first_id: ID of the first assistant in the list
        last_id: ID of the last assistant in the list
        has_more: Whether there are more assistants to retrieve
    """
    object: str = "list"
    data: List[Assistant]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class AssistantFile(BaseModel):
    """Assistant file object.

    Attributes:
        id: Unique identifier for the assistant file
        object: Object type, always "assistant.file"
        created_at: Unix timestamp of creation
        assistant_id: ID of the assistant this file is attached to
    """
    id: str
    object: str = "assistant.file"
    created_at: int
    assistant_id: str


class AssistantFileListResponse(BaseModel):
    """Response model for listing assistant files.

    Attributes:
        object: Object type, always "list"
        data: List of assistant file objects
        first_id: ID of the first file in the list
        last_id: ID of the last file in the list
        has_more: Whether there are more files to retrieve
    """
    object: str = "list"
    data: List[AssistantFile]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class Thread(BaseModel):
    """Thread object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the thread
        object: Object type, always "thread"
        created_at: Unix timestamp of creation
        metadata: Optional metadata for the thread
    """
    id: str
    object: str = "thread"
    created_at: int
    metadata: Optional[Dict[str, Any]] = None


class CreateThreadRequest(BaseModel):
    """Request model for creating a thread.

    Attributes:
        messages: Optional list of messages to start the thread with
        metadata: Optional metadata for the thread
    """
    messages: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModifyThreadRequest(BaseModel):
    """Request model for modifying a thread.

    Attributes:
        metadata: Optional new metadata for the thread
    """
    metadata: Optional[Dict[str, Any]] = None


class ThreadMessage(BaseModel):
    """Thread message object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the message
        object: Object type, always "thread.message"
        created_at: Unix timestamp of creation
        thread_id: ID of the thread this message belongs to
        role: Role of the message author ("user" or "assistant")
        content: Content of the message
        file_ids: List of file IDs attached to the message
        assistant_id: ID of the assistant that authored the message (if applicable)
        run_id: ID of the run this message was generated in (if applicable)
        metadata: Optional metadata for the message
    """
    id: str
    object: str = "thread.message"
    created_at: int
    thread_id: str
    role: Literal["user", "assistant"]
    content: List[Dict[str, Any]]  # List of content blocks
    file_ids: List[str] = []
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CreateMessageRequest(BaseModel):
    """Request model for creating a message.

    Attributes:
        role: Role of the message author ("user" or "assistant")
        content: Content of the message
        file_ids: Optional list of file IDs to attach
        metadata: Optional metadata for the message
    """
    role: Literal["user", "assistant"]
    content: str
    file_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModifyMessageRequest(BaseModel):
    """Request model for modifying a message.

    Attributes:
        metadata: Optional new metadata for the message
    """
    metadata: Optional[Dict[str, Any]] = None


class MessageListResponse(BaseModel):
    """Response model for listing messages.

    Attributes:
        object: Object type, always "list"
        data: List of message objects
        first_id: ID of the first message in the list
        last_id: ID of the last message in the list
        has_more: Whether there are more messages to retrieve
    """
    object: str = "list"
    data: List[ThreadMessage]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class Run(BaseModel):
    """Run object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the run
        object: Object type, always "thread.run"
        created_at: Unix timestamp of creation
        thread_id: ID of the thread this run belongs to
        assistant_id: ID of the assistant used for this run
        status: Current status of the run
        started_at: Unix timestamp when the run started
        expires_at: Unix timestamp when the run expires
        cancelled_at: Unix timestamp when the run was cancelled
        failed_at: Unix timestamp when the run failed
        completed_at: Unix timestamp when the run completed
        last_error: Last error encountered by the run
        model: Model used for this run
        instructions: Instructions used for this run
        tools: Tools available for this run
        file_ids: File IDs attached to the assistant
        metadata: Optional metadata for the run
        usage: Token usage statistics
        temperature: Temperature setting used
        top_p: Top P setting used
        max_prompt_tokens: Maximum prompt tokens
        max_completion_tokens: Maximum completion tokens
        truncation_strategy: Truncation strategy used
        response_format: Response format used
        tool_choice: Tool choice setting
        parallel_tool_calls: Whether parallel tool calls are enabled
    """
    id: str
    object: str = "thread.run"
    created_at: int
    thread_id: str
    assistant_id: str
    status: Literal["queued", "in_progress", "requires_action", "cancelling", "cancelled", "failed", "completed", "expired"]
    started_at: Optional[int] = None
    expires_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    last_error: Optional[Dict[str, Any]] = None
    model: str
    instructions: Optional[str] = None
    tools: List[AssistantTool] = []
    file_ids: List[str] = []
    metadata: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[Dict[str, Any]] = None
    response_format: Optional[str] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True


class CreateRunRequest(BaseModel):
    """Request model for creating a run.

    Attributes:
        assistant_id: ID of the assistant to use
        model: Optional model override
        instructions: Optional instructions override
        tools: Optional tools override
        metadata: Optional metadata for the run
        temperature: Optional temperature setting
        top_p: Optional top_p setting
        max_prompt_tokens: Optional maximum prompt tokens
        max_completion_tokens: Optional maximum completion tokens
        truncation_strategy: Optional truncation strategy
        tool_choice: Optional tool choice setting
        parallel_tool_calls: Optional parallel tool calls setting
        response_format: Optional response format
    """
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    metadata: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[str] = None


class ModifyRunRequest(BaseModel):
    """Request model for modifying a run.

    Attributes:
        metadata: Optional new metadata for the run
    """
    metadata: Optional[Dict[str, Any]] = None


class RunListResponse(BaseModel):
    """Response model for listing runs.

    Attributes:
        object: Object type, always "list"
        data: List of run objects
        first_id: ID of the first run in the list
        last_id: ID of the last run in the list
        has_more: Whether there are more runs to retrieve
    """
    object: str = "list"
    data: List[Run]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class RunStep(BaseModel):
    """Run step object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the step
        object: Object type, always "thread.run.step"
        created_at: Unix timestamp of creation
        run_id: ID of the run this step belongs to
        thread_id: ID of the thread this step belongs to
        type: Type of step
        status: Status of the step
        step_details: Details of the step execution
        last_error: Last error encountered by the step
        expired_at: Unix timestamp when the step expired
        cancelled_at: Unix timestamp when the step was cancelled
        failed_at: Unix timestamp when the step failed
        completed_at: Unix timestamp when the step completed
        metadata: Optional metadata for the step
        usage: Token usage statistics for the step
    """
    id: str
    object: str = "thread.run.step"
    created_at: int
    run_id: str
    thread_id: str
    type: Literal["message_creation", "tool_calls"]
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"]
    step_details: Dict[str, Any]
    last_error: Optional[Dict[str, Any]] = None
    expired_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None


class RunStepListResponse(BaseModel):
    """Response model for listing run steps.

    Attributes:
        object: Object type, always "list"
        data: List of run step objects
        first_id: ID of the first step in the list
        last_id: ID of the last step in the list
        has_more: Whether there are more steps to retrieve
    """
    object: str = "list"
    data: List[RunStep]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class CreateThreadAndRunRequest(BaseModel):
    """Request model for creating a thread and run together.

    Attributes:
        assistant_id: ID of the assistant to use
        thread: Optional thread data to create
        model: Optional model override
        instructions: Optional instructions override
        tools: Optional tools override
        metadata: Optional metadata for the run
        temperature: Optional temperature setting
        top_p: Optional top_p setting
        max_prompt_tokens: Optional maximum prompt tokens
        max_completion_tokens: Optional maximum completion tokens
        truncation_strategy: Optional truncation strategy
        tool_choice: Optional tool choice setting
        parallel_tool_calls: Optional parallel tool calls setting
        response_format: Optional response format
    """
    assistant_id: str
    thread: Optional[CreateThreadRequest] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[AssistantTool]] = None
    metadata: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[str] = None


class ChunkingStrategy(BaseModel):
    """Chunking strategy for file processing.

    Attributes:
        type: The chunking strategy type ("auto" or "static")
        max_chunk_size_tokens: Maximum chunk size in tokens (for static)
        chunk_overlap_tokens: Overlap between chunks in tokens (for static)
    """
    type: Literal["auto", "static"] = "auto"
    max_chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None


class CreateVectorStoreRequest(BaseModel):
    """Request model for creating a vector store.

    Attributes:
        file_ids: Optional list of file IDs to attach
        name: Optional name for the vector store
        description: Optional description
        metadata: Optional metadata
        expires_after: Optional expiration policy
        chunking_strategy: Optional chunking strategy
    """
    file_ids: Optional[List[str]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_after: Optional[ExpiresAfter] = None
    chunking_strategy: Optional[ChunkingStrategy] = None


class VectorStore(BaseModel):
    """Vector store object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the vector store
        object: Object type, always "vector_store"
        created_at: Unix timestamp of creation
        name: Optional name of the vector store
        description: Optional description
        usage_bytes: Total bytes used by the vector store
        file_counts: Counts of files in different statuses
        status: Current status of the vector store
        expires_after: Optional expiration policy
        expires_at: Optional expiration timestamp
        last_active_at: Timestamp of last activity
        metadata: Optional metadata
    """
    id: str
    object: str = "vector_store"
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    usage_bytes: int = 0
    file_counts: Dict[str, int] = Field(default_factory=lambda: {
        "in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0
    })
    status: Literal["expired", "in_progress", "completed"] = "completed"
    expires_after: Optional[ExpiresAfter] = None
    expires_at: Optional[int] = None
    last_active_at: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ModifyVectorStoreRequest(BaseModel):
    """Request model for modifying a vector store.

    Attributes:
        name: Optional new name
        description: Optional new description
        metadata: Optional new metadata
        expires_after: Optional new expiration policy
    """
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_after: Optional[ExpiresAfter] = None


class VectorStoreListResponse(BaseModel):
    """Response model for listing vector stores.

    Attributes:
        object: Object type, always "list"
        data: List of vector store objects
        first_id: ID of the first vector store in the list
        last_id: ID of the last vector store in the list
        has_more: Whether there are more vector stores to retrieve
    """
    object: str = "list"
    data: List[VectorStore]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class CreateVectorStoreFileRequest(BaseModel):
    """Request model for creating a vector store file.

    Attributes:
        file_id: ID of the file to attach
        attributes: Optional attributes for the file
        chunking_strategy: Optional chunking strategy override
    """
    file_id: str
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None


class VectorStoreFile(BaseModel):
    """Vector store file object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the file (same as file_id)
        object: Object type, always "vector_store.file"
        created_at: Unix timestamp of creation
        vector_store_id: ID of the vector store this file belongs to
        status: Processing status
        usage_bytes: Bytes used by this file
        attributes: Optional attributes
        last_error: Optional last error information
    """
    id: str
    object: str = "vector_store.file"
    created_at: int
    vector_store_id: str
    status: Literal["in_progress", "completed", "failed", "cancelled"] = "completed"
    usage_bytes: int = 0
    attributes: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None


class VectorStoreFileListResponse(BaseModel):
    """Response model for listing vector store files.

    Attributes:
        object: Object type, always "list"
        data: List of vector store file objects
        first_id: ID of the first file in the list
        last_id: ID of the last file in the list
        has_more: Whether there are more files to retrieve
    """
    object: str = "list"
    data: List[VectorStoreFile]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class VectorStoreFileDeleteResponse(BaseModel):
    """Response model for deleting a vector store file.

    Attributes:
        id: ID of the deleted file
        object: Object type, always "vector_store.file.deleted"
        deleted: Whether the file was deleted successfully
    """
    id: str
    object: str = "vector_store.file.deleted"
    deleted: bool


class VectorStoreFileBatchFile(BaseModel):
    """File specification for batch operations.

    Attributes:
        file_id: ID of the file
        attributes: Optional attributes
        chunking_strategy: Optional chunking strategy override
    """
    file_id: str
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None


class CreateVectorStoreFileBatchRequest(BaseModel):
    """Request model for creating a vector store file batch.

    Attributes:
        file_ids: List of file IDs (mutually exclusive with files)
        files: List of file specifications (mutually exclusive with file_ids)
        attributes: Optional batch-level attributes
        chunking_strategy: Optional batch-level chunking strategy
    """
    file_ids: Optional[List[str]] = None
    files: Optional[List[VectorStoreFileBatchFile]] = None
    attributes: Optional[Dict[str, Any]] = None
    chunking_strategy: Optional[ChunkingStrategy] = None


class VectorStoreFileBatch(BaseModel):
    """Vector store file batch object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the batch
        object: Object type, always "vector_store.file_batch"
        created_at: Unix timestamp of creation
        vector_store_id: ID of the vector store
        status: Processing status
        file_counts: Counts of files in different statuses
        completes_at: Optional completion timestamp
        expires_at: Optional expiration timestamp
    """
    id: str
    object: str = "vector_store.file_batch"
    created_at: int
    vector_store_id: str
    status: Literal["in_progress", "completed", "failed", "cancelled"] = "completed"
    file_counts: Dict[str, int] = Field(default_factory=lambda: {
        "in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0
    })
    completes_at: Optional[int] = None
    expires_at: Optional[int] = None


class VectorStoreFileBatchListResponse(BaseModel):
    """Response model for listing vector store file batches.

    Attributes:
        object: Object type, always "list"
        data: List of file batch objects
        first_id: ID of the first batch in the list
        last_id: ID of the last batch in the list
        has_more: Whether there are more batches to retrieve
    """
    object: str = "list"
    data: List[VectorStoreFileBatch]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class CreateBatchRequest(BaseModel):
    """Request model for creating a batch job.

    Attributes:
        completion_window: The time frame within which the batch should be processed
        endpoint: The endpoint to be used for all requests in the batch
        input_file_id: The ID of an uploaded file containing requests
        metadata: Optional set of key-value pairs for storing additional information
        output_expires_after: Optional expiration policy for generated output and error files
    """
    completion_window: str = "24h"
    endpoint: str
    input_file_id: str
    metadata: Optional[Dict[str, Any]] = None
    output_expires_after: Optional[ExpiresAfter] = None


class BatchRequestInput(BaseModel):
    """Individual request input for batch processing.

    Attributes:
        custom_id: Developer-provided ID to match outputs to inputs
        method: HTTP method for the request
        url: OpenAI API relative URL for the request
        body: Request body containing API-specific parameters
    """
    custom_id: str
    method: str = "POST"
    url: str
    body: Dict[str, Any]


class BatchRequestCounts(BaseModel):
    """Request statistics for a batch.

    Attributes:
        total: Total number of requests
        completed: Number of completed requests
        failed: Number of failed requests
    """
    total: int = 0
    completed: int = 0
    failed: int = 0


class Batch(BaseModel):
    """Batch object following OpenAI API specification.

    Attributes:
        id: Unique identifier for the batch
        object: Object type, always "batch"
        endpoint: The endpoint used for all requests in the batch
        errors: Validation errors if any
        input_file_id: ID of the input file
        completion_window: Processing window (e.g., "24h")
        status: Current batch status
        output_file_id: ID of the file containing batch results
        error_file_id: ID of the file containing error details
        created_at: Unix timestamp of batch creation
        in_progress_at: Unix timestamp when batch processing started
        expires_at: Unix timestamp when batch expires
        finalizing_at: Unix timestamp when batch started finalizing
        completed_at: Unix timestamp when batch completed
        failed_at: Unix timestamp when batch failed
        expired_at: Unix timestamp when batch expired
        cancelling_at: Unix timestamp when batch is cancelling
        cancelled_at: Unix timestamp when batch was cancelled
        request_counts: Request statistics
        metadata: Custom metadata attached to the batch
    """
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[Dict[str, Any]] = None
    input_file_id: str
    completion_window: str = "24h"
    status: Literal["validating", "queued", "in_progress", "completed", "failed", "expired", "cancelled", "cancelling"] = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: BatchRequestCounts = BatchRequestCounts()
    metadata: Optional[Dict[str, Any]] = None


class BatchListResponse(BaseModel):
    """Response model for listing batches.

    Attributes:
        object: Object type, always "list"
        data: List of batch objects
        first_id: ID of the first batch in the list
        last_id: ID of the last batch in the list
        has_more: Whether there are more batches to retrieve
    """
    object: str = "list"
    data: List[Batch]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class BatchResult(BaseModel):
    """Individual batch result object.

    Attributes:
        custom_id: The custom ID from the request
        response: The API response for this request
        error: Error details if the request failed
    """
    custom_id: str
    response: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None