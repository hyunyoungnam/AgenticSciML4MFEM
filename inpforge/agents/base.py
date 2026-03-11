"""
Base classes for the AgenticSciML agent system.

Defines the fundamental abstractions for agents, messages, and contexts
that all specialized agents inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic
import uuid


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    EVALUATOR = "evaluator"
    PROPOSER = "proposer"
    CRITIC = "critic"
    ENGINEER = "engineer"
    DEBUGGER = "debugger"
    RESULT_ANALYST = "result_analyst"
    RETRIEVER = "retriever"
    SELECTOR = "selector"


class MessageType(Enum):
    """Types of messages exchanged between agents."""
    QUERY = "query"
    RESPONSE = "response"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    APPROVAL = "approval"
    REJECTION = "rejection"
    ERROR = "error"
    DEBUG_REQUEST = "debug_request"
    DEBUG_RESPONSE = "debug_response"


@dataclass
class AgentMessage:
    """
    Message exchanged between agents.

    Attributes:
        id: Unique message identifier
        sender: Role of the sending agent
        receiver: Role of the receiving agent (optional for broadcasts)
        type: Type of the message
        content: Message content (string or structured data)
        metadata: Additional metadata (timestamps, references, etc.)
        timestamp: When the message was created
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentRole = AgentRole.EVALUATOR
    receiver: Optional[AgentRole] = None
    type: MessageType = MessageType.QUERY
    content: Any = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "sender": self.sender.value,
            "receiver": self.receiver.value if self.receiver else None,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary representation."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            sender=AgentRole(data["sender"]),
            receiver=AgentRole(data["receiver"]) if data.get("receiver") else None,
            type=MessageType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
        )


@dataclass
class AgentContext:
    """
    Shared context passed between agents during a workflow.

    Contains all information needed for agents to make decisions,
    including the current solution state, conversation history,
    and retrieved knowledge.

    Attributes:
        session_id: Unique session identifier
        base_inp_path: Path to the base .inp file
        current_solution_id: ID of the current solution being worked on
        conversation_history: List of messages in the current conversation
        knowledge_context: Retrieved knowledge entries
        failure_history: Previous failure patterns
        model_info: Information about the loaded Abaqus model
        evaluation_criteria: Criteria from the Evaluator agent
        config: Runtime configuration
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    base_inp_path: Optional[str] = None
    current_solution_id: Optional[str] = None
    conversation_history: List[AgentMessage] = field(default_factory=list)
    knowledge_context: List[Dict[str, Any]] = field(default_factory=list)
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)

    def get_messages_by_sender(self, sender: AgentRole) -> List[AgentMessage]:
        """Get all messages from a specific sender."""
        return [m for m in self.conversation_history if m.sender == sender]

    def get_messages_by_type(self, msg_type: MessageType) -> List[AgentMessage]:
        """Get all messages of a specific type."""
        return [m for m in self.conversation_history if m.type == msg_type]

    def get_last_message(self) -> Optional[AgentMessage]:
        """Get the most recent message in the conversation."""
        return self.conversation_history[-1] if self.conversation_history else None

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary representation."""
        return {
            "session_id": self.session_id,
            "base_inp_path": self.base_inp_path,
            "current_solution_id": self.current_solution_id,
            "conversation_history": [m.to_dict() for m in self.conversation_history],
            "knowledge_context": self.knowledge_context,
            "failure_history": self.failure_history,
            "model_info": self.model_info,
            "evaluation_criteria": self.evaluation_criteria,
            "config": self.config,
        }


T = TypeVar("T")


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all agents in the system.

    Provides common functionality for LLM interaction, message handling,
    and context management. Specialized agents implement the abstract
    methods to define their specific behavior.

    Type parameter T represents the type of response the agent produces.
    """

    def __init__(
        self,
        role: AgentRole,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Initialize the base agent.

        Args:
            role: The role of this agent
            model: LLM model identifier
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens in LLM response
        """
        self.role = role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm_provider = None

    @property
    def name(self) -> str:
        """Human-readable name for the agent."""
        return self.role.value.replace("_", " ").title()

    def set_llm_provider(self, provider: Any) -> None:
        """
        Set the LLM provider for this agent.

        Args:
            provider: LLMProvider instance
        """
        self._llm_provider = provider

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def build_user_prompt(self, context: AgentContext, **kwargs) -> str:
        """
        Build the user prompt based on context and additional arguments.

        Args:
            context: Current agent context
            **kwargs: Additional arguments for prompt building

        Returns:
            User prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> T:
        """
        Parse the LLM response into a structured output.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed response of type T
        """
        pass

    async def execute(self, context: AgentContext, **kwargs) -> T:
        """
        Execute the agent's main task.

        Args:
            context: Current agent context
            **kwargs: Additional arguments

        Returns:
            Agent's response of type T
        """
        if self._llm_provider is None:
            raise RuntimeError(f"LLM provider not set for {self.name}")

        system_prompt = self.get_system_prompt()
        user_prompt = self.build_user_prompt(context, **kwargs)

        response = await self._llm_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return self.parse_response(response.content)

    def create_message(
        self,
        content: Any,
        msg_type: MessageType,
        receiver: Optional[AgentRole] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        Create a message from this agent.

        Args:
            content: Message content
            msg_type: Type of message
            receiver: Intended recipient (optional)
            metadata: Additional metadata

        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=self.role,
            receiver=receiver,
            type=msg_type,
            content=content,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.role.value}, model={self.model})"
