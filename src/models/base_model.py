from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.logger_config import setup_logger

logger = setup_logger(__name__)

class BaseEmbeddingModel(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def __init__(self):
        """Initialize the model."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts (List[str]): List of texts to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings, shape (len(texts), embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        pass