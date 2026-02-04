# Custom exceptions for the BI Platform
from fastapi import HTTPException, status
from typing import Optional, Any


class BIPlatformException(Exception):
    """Base exception for BI Platform."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatasetNotFoundException(BIPlatformException):
    """Raised when a dataset is not found."""
    pass


class InvalidDatasetException(BIPlatformException):
    """Raised when a dataset is invalid or corrupted."""
    pass


class MappingConfirmationRequired(BIPlatformException):
    """Raised when user confirmation is needed for mappings."""
    pass


class AnalysisInProgressException(BIPlatformException):
    """Raised when analysis is already in progress."""
    pass


class InsufficientDataException(BIPlatformException):
    """Raised when there's not enough data for analysis."""
    pass


class LLMException(BIPlatformException):
    """Raised when LLM inference fails."""
    pass


# HTTP Exception helpers
def not_found(detail: str = "Resource not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def bad_request(detail: str = "Bad request") -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def conflict(detail: str = "Conflict") -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def unprocessable(detail: str = "Unprocessable entity") -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)
