import torch
from transformers import AutoTokenizer, AutoModel
from .base_model import BaseEmbeddingModel
import numpy as np
from typing import List
from logger_config import setup_logger

logger = setup_logger(__name__)

class RobertaModel(BaseEmbeddingModel):
    """RuRoBERTa model for text embeddings."""

    def __init__(self):
        """Initialize the RuRoBERTa model."""
        super().__init__()
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoberta-large")
            self.model = AutoModel.from_pretrained("ai-forever/ruRoberta-large")
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("RuRoBERTa model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RuRoBERTa model: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts using RuRoBERTa.

        Args:
            texts (List[str]): List of texts to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings, shape (len(texts), embedding_dim)
        """
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                add_special_tokens=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use CLS token (first token) for the embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings

        except Exception as e:
            logger.error(f"Error generating RuRoBERTa embeddings: {str(e)}")
            return None

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of RuRoBERTa embeddings."""
        return 1024  # RuRoBERTa-large hidden size