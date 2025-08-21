"""Shared logging configuration for VSR API and Worker."""

import logging
import sys
from typing import Any, Dict, List, Optional, Union

import structlog
from structlog.types import EventDict, Processor


def setup_logging(
    level: Union[str, int] = logging.INFO,
    json_format: bool = True,
    add_console_handler: bool = True,
    add_file_handler: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        json_format: Whether to use JSON format (default: True)
        add_console_handler: Whether to add a console handler (default: True)
        add_file_handler: Whether to add a file handler (default: False)
        log_file: Log file path (required if add_file_handler is True)
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure standard logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
        handlers=[],
    )
    
    # Add handlers
    if add_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        logging.getLogger().addHandler(console_handler)
    
    if add_file_handler:
        if not log_file:
            raise ValueError("log_file must be provided when add_file_handler is True")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        logging.getLogger().addHandler(file_handler)
    
    # Configure structlog processors
    processors: List[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_format:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger
    """
    return structlog.get_logger(name)


class RequestIdProcessor:
    """Processor that adds request_id to log events."""
    
    def __call__(self, logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        """
        Add request_id to the event dict if available in context vars.
        
        Args:
            logger: Logger instance
            method_name: Method name
            event_dict: Event dict
            
        Returns:
            Updated event dict
        """
        from structlog.contextvars import get_contextvars
        
        context = get_contextvars()
        request_id = context.get("request_id")
        
        if request_id:
            event_dict["request_id"] = request_id
        
        return event_dict


def bind_request_id(request_id: str) -> None:
    """
    Bind request_id to the current context.
    
    Args:
        request_id: Request ID
    """
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_id() -> None:
    """Clear request_id from the current context."""
    structlog.contextvars.unbind_contextvars("request_id")


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to the current context.
    
    Args:
        **kwargs: Context variables
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context(*keys: str) -> None:
    """
    Clear context variables from the current context.
    
    Args:
        *keys: Context variable keys
    """
    structlog.contextvars.unbind_contextvars(*keys)
