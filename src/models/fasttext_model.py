import traceback
import gensim.downloader as api
import numpy as np
from typing import List
from .base_model import BaseEmbeddingModel
from src.logger_config import setup_logger
import fasttext
import os
import re
import nltk
from nltk.corpus import stopwords
import pymorphy2

logger = setup_logger(__name__)

class FastTextModel(BaseEmbeddingModel):
    """Russian FastText model for text embeddings."""

    def __init__(self):
        """Initialize the Russian FastText model."""
        super().__init__()
        try:
            logger.info("Loading Russian FastText model...")
            # Load pre-downloaded FastText model from resources
            model_path = os.path.join('resources', 'model.bin')
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    "FastText model not found. Please download it first using: "
                    "python -c \"from huggingface_hub import hf_hub_download; "
                    "hf_hub_download(repo_id='facebook/fasttext-ru-vectors', "
                    "filename='model.bin', local_dir='resources')\""
                )

            self.model = fasttext.load_model(model_path)

            # Download Russian stopwords if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)

            self.stop_words = set(stopwords.words('russian'))
            self.stop_words.add("поставка")
            #self.stop_words.add("")
            # Initialize morphological analyzer
            self.morph = pymorphy2.MorphAnalyzer()

            logger.info("FastText model, stopwords, and morphological analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FastText model: {str(e)}")
            logger.error(traceback.print_exc())
            raise

    def lemmatize(self, text: str) -> str:
        """
        Lemmatize Russian text using pymorphy2.

        Args:
            text (str): Input text

        Returns:
            str: Lemmatized text
        """
        words = text.split()
        lemmatized_words = []
        for word in words:
            # Get the normal form (lemma) of the word
            lemma = self.morph.parse(word)[0].normal_form
            lemmatized_words.append(lemma)
        return ' '.join(lemmatized_words)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for FastText model.
        - Removes stopwords
        - Replaces newlines and multiple spaces with single space
        - Removes leading/trailing whitespace
        - Lemmatizes words
        - Converts to lowercase

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        # Replace newlines and multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        # Join words back together
        text = ' '.join(words)
        # Lemmatize text
        text = self.lemmatize(text)

        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        text = ' '.join(words)

        return text

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts using FastText.
        FastText can handle out-of-vocabulary words better than Word2Vec
        due to its subword information.

        Args:
            texts (List[str]): List of texts to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings, shape (len(texts), embedding_dim)
        """
        try:
            embeddings = []
            for text in texts:
                # Preprocess text to handle newlines and other special characters
                processed_text = self.preprocess_text(text)
                # Get sentence vector from FastText
                embedding = self.model.get_sentence_vector(processed_text)
                embeddings.append(embedding)
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error generating FastText embeddings: {str(e)}")
            return None

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of FastText embeddings."""
        return 300  # FastText dimension