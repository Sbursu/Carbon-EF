import logging
from typing import Any, Dict, List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from src.utils import load_config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings for emission factor data
    """

    def __init__(
        self, model_name: Optional[str] = None, config_path: Optional[str] = None
    ):
        """
        Initialize the embedding generator

        Args:
            model_name: Name of the embedding model to use
            config_path: Path to the configuration file
        """
        self.config = load_config(config_path)
        embedding_config = self.config.get(
            "embedding", {}
        )  # Get embedding config section

        # Use provided model name or default from new embedding config section
        if model_name is None:
            # Default to using the model specified in the 'embedding' section
            self.model_name = embedding_config.get(
                "model_name", "all-MiniLM-L6-v2"
            )  # Default fallback
        else:
            self.model_name = model_name

        logger.info(f"Initializing embedding generator with model: {self.model_name}")

        # Check if we're using a sentence-transformer model or a HF model
        if "sentence-transformers" in self.model_name:
            self.model = SentenceTransformer(self.model_name)
            self.model_type = "sentence_transformer"
        else:
            # For Phi-2 or other models, use the HF model directly
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model_type = "hf_model"

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        if self.model_type == "hf_model":
            self.model = self.model.to(self.device)

    def format_emission_factor(self, ef_data: Dict[str, Any]) -> str:
        """
        Format emission factor data into a text representation for embedding

        Args:
            ef_data: Emission factor data dictionary

        Returns:
            Formatted text representation
        """
        # Create a standardized text representation
        formatted_text = (
            f"Emission Factor: {ef_data.get('entity_name', 'Unknown')} "
            f"Type: {ef_data.get('entity_type', 'Unknown')} "
            f"Region: {ef_data.get('region', 'Global')} "
            f"Value: {ef_data.get('ef_value', 0)} {ef_data.get('ef_unit', 'kg CO2e')} "
            f"Source: {ef_data.get('source_dataset', 'Unknown')}"
        )

        # Add any additional metadata if present
        if "metadata" in ef_data and ef_data["metadata"]:
            formatted_text += f" Metadata: {str(ef_data['metadata'])}"

        return formatted_text

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        if self.model_type == "sentence_transformer":
            # Use sentence-transformers model for embedding
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            # Use HF model for embedding
            embeddings = []
            batch_size = 8  # Process in small batches to avoid OOM

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Use mean pooling for sentence representation
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Calculate mean of token embeddings (ignoring padding tokens)
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

                embeddings.extend(batch_embeddings.tolist())

            return embeddings

    def embed_emission_factors(
        self, emission_factors: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of emission factor data points

        Args:
            emission_factors: List of emission factor data dictionaries

        Returns:
            List of embedding vectors
        """
        # Format each emission factor into text
        texts = [self.format_emission_factor(ef) for ef in emission_factors]

        # Generate embeddings
        return self.generate_embeddings(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query string

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([query])
        return embeddings[0]
