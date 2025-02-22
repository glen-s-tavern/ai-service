from enum import Enum
from typing import Optional
from .base_model import BaseEmbeddingModel
from .roberta_model import RobertaModel
from .fasttext_model import FastTextModel
from src.logger_config import setup_logger

logger = setup_logger(__name__)

class ModelType(str, Enum):
    """Supported model types."""
    roberta = "roberta"
    fasttext = "fasttext"

class ModelFactory:
    """Factory class for creating embedding models."""

    _instances = {}

    @classmethod
    def get_model(cls, model_type: ModelType) -> BaseEmbeddingModel:
        """
        Get or create an instance of the specified model type.
        Uses singleton pattern to avoid loading models multiple times.

        Args:
            model_type (ModelType): Type of model to create

        Returns:
            BaseEmbeddingModel: Instance of the requested model

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls._instances:
            logger.info(f"Creating new instance of {model_type} model")
            if model_type == ModelType.roberta:
                cls._instances[model_type] = RobertaModel()
            elif model_type == ModelType.fasttext:
                cls._instances[model_type] = FastTextModel()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        return cls._instances[model_type]