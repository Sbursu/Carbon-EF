import logging
from typing import Any, Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils import load_config

logger = logging.getLogger(__name__)

# Global model and tokenizer to implement singleton pattern
_MODEL_INSTANCE = None
_TOKENIZER_INSTANCE = None


class Phi2Model:
    """
    Phi-2 model implementation
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Phi-2 model

        Args:
            config_path: Path to configuration file
        """
        global _MODEL_INSTANCE, _TOKENIZER_INSTANCE

        self.config = load_config(config_path)
        self.model_config = self.config["agent"]["phi2_model"]
        self.model_name = self.model_config["model_name"]

        # Set device (CPU or GPU)
        self.device = "cpu"  # Default to CPU
        logger.info(f"Using device: {self.device}")

        # Load the model and tokenizer only if they haven't been loaded yet
        if _MODEL_INSTANCE is None or _TOKENIZER_INSTANCE is None:
            logger.info(f"Loading model: {self.model_name}")

            # Load the model with quantization if needed
            try:
                # Avoid device argument - use default behavior
                _TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(self.model_name)

                # Set load_in_8bit directly without using device parameter
                load_in_8bit = self.model_config.get("quantize_8bit", False)
                if load_in_8bit:
                    print("Device set to use cpu")
                    _MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_8bit=load_in_8bit,
                        trust_remote_code=True,
                    )
                else:
                    _MODEL_INSTANCE = AutoModelForCausalLM.from_pretrained(
                        self.model_name, trust_remote_code=True
                    ).to(self.device)

                logger.info(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise

        # Use the global instances
        self.model = _MODEL_INSTANCE
        self.tokenizer = _TOKENIZER_INSTANCE

        # Set generation parameters
        self.max_length = self.model_config.get("max_length", 512)
        self.temperature = self.model_config.get("temperature", 0.7)
        self.top_p = self.model_config.get("top_p", 0.9)

        # Create text generation pipeline
        self.text_generation = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
        )

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response for the given prompt

        Args:
            prompt: Input prompt

        Returns:
            Generated response
        """
        try:
            # Generate response
            outputs = self.text_generation(
                prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=1,
            )

            # Extract and clean response
            response = outputs[0]["generated_text"]

            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt) :].strip()

            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query to extract key information

        Args:
            query: User query

        Returns:
            Analysis results
        """
        prompt = f"""
        Analyze the following user query about emission factors:
        
        Query: "{query}"
        
        Extract the following information:
        1. Entity type (e.g., electricity, natural gas, transportation, etc.)
        2. Entity name (specific entity being referred to)
        3. Region (specific location or 'global' if not specified)
        4. Is this a comparison query? (true/false)
        5. If comparison, list regions being compared
        
        Format the response as a JSON object.
        """

        response = self.generate_response(prompt)

        # Parse JSON from response
        try:
            import json
            import re

            # Try to extract JSON using regex
            json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
            else:
                # Fallback to manual extraction
                analysis = {
                    "entity_type": "unknown",
                    "entity_name": "unknown",
                    "region": "global",
                    "comparison": False,
                    "regions": [],
                }

                # Try to extract key information
                if "electricity" in query.lower():
                    analysis["entity_type"] = "electricity"
                elif "natural gas" in query.lower():
                    analysis["entity_type"] = "natural gas"
                elif "transport" in query.lower():
                    analysis["entity_type"] = "transportation"

                # Check for regions
                regions = [
                    "us",
                    "usa",
                    "america",
                    "united states",
                    "europe",
                    "eu",
                    "china",
                    "india",
                    "canada",
                    "uk",
                    "australia",
                    "global",
                    "california",
                    "texas",
                ]

                for region in regions:
                    if region in query.lower():
                        if region in ["us", "usa", "america", "united states"]:
                            analysis["region"] = "USA"
                        elif region == "california":
                            analysis["region"] = "CA"
                        elif region == "texas":
                            analysis["region"] = "TX"
                        else:
                            analysis["region"] = region.upper()
                        break

                # Check if it's a comparison
                comparison_terms = [
                    "compare",
                    "comparison",
                    "versus",
                    "vs",
                    "difference",
                ]
                analysis["comparison"] = any(
                    term in query.lower() for term in comparison_terms
                )

            return analysis
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            return {
                "entity_type": "unknown",
                "entity_name": "unknown",
                "region": "global",
                "comparison": False,
                "error": str(e),
            }

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on provided context

        Args:
            question: User question
            context: Context information

        Returns:
            Generated answer
        """
        prompt = f"""
        Context: {context}
        
        Question: {question}
        
        Based on the context provided, answer the question in a comprehensive way.
        If the context doesn't contain enough information to answer the question,
        say so and explain what additional information would be needed.
        """

        return self.generate_response(prompt)

    def explain_reasoning(
        self, query: str, results: List[Dict[str, Any]], answer: str
    ) -> str:
        """
        Explain the reasoning behind an answer

        Args:
            query: Original user query
            results: Query results
            answer: Answer provided

        Returns:
            Explanation of reasoning
        """
        # Format results as text
        results_text = ""
        for i, result in enumerate(results):
            results_text += f"Result {i+1}:\n"
            for key, value in result.items():
                if key != "result":  # Skip nested result object for clarity
                    results_text += f"  {key}: {value}\n"
            results_text += "\n"

        prompt = f"""
        User Query: {query}
        
        Answer Provided: {answer}
        
        Data Used:
        {results_text}
        
        Explain the reasoning behind the answer in a few sentences. Focus on:
        1. How the data supports the answer
        2. Any assumptions or limitations
        3. Confidence in the answer based on the data quality
        """

        return self.generate_response(prompt)

    def synthesize_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Synthesize results into a coherent answer

        Args:
            query: User query
            results: Query results

        Returns:
            Synthesized answer
        """
        # Format results as text
        results_text = ""
        for i, result in enumerate(results):
            results_text += f"Result {i+1} ({result.get('query_type', 'unknown')}):\n"

            if "error" in result:
                results_text += f"  Error: {result['error']}\n"
                continue

            # Handle different types of results
            if "result" in result and isinstance(result["result"], dict):
                sub_result = result["result"]

                # Handle emission factors
                if "emission_factors" in sub_result:
                    results_text += "  Emission Factors:\n"
                    for j, ef in enumerate(
                        sub_result["emission_factors"][:3]
                    ):  # Limit to 3 for clarity
                        results_text += f"    {j+1}. "

                        # Format key properties
                        properties = []
                        if "entity_name" in ef:
                            properties.append(f"Entity: {ef['entity_name']}")
                        if "entity_type" in ef:
                            properties.append(f"Type: {ef['entity_type']}")
                        if "region" in ef:
                            properties.append(f"Region: {ef['region']}")
                        if "ef_value" in ef or "value" in ef:
                            value = ef.get("ef_value", ef.get("value", "N/A"))
                            unit = ef.get("ef_unit", ef.get("unit", ""))
                            properties.append(f"Value: {value} {unit}")
                        if "confidence" in ef:
                            properties.append(f"Confidence: {ef['confidence']}")
                        if "source" in ef:
                            properties.append(f"Source: {ef['source']}")

                        results_text += ", ".join(properties) + "\n"

                    if (
                        "emission_factors" in sub_result
                        and len(sub_result["emission_factors"]) > 3
                    ):
                        results_text += f"    ... and {len(sub_result['emission_factors']) - 3} more\n"

                # Handle statistics
                if "statistics" in sub_result:
                    stats = sub_result["statistics"]
                    results_text += "  Statistics:\n"
                    for key, value in stats.items():
                        results_text += f"    {key}: {value}\n"

                # Handle graph context
                if "graph_context" in sub_result and sub_result["graph_context"]:
                    results_text += (
                        "  Graph Context: [Available but omitted for brevity]\n"
                    )

            results_text += "\n"

        prompt = f"""
        User Query: {query}
        
        Available Information:
        {results_text}
        
        Based on the available information, provide a comprehensive answer to the user's query.
        Focus on being accurate, clear, and concise.
        If the information is incomplete or uncertain, acknowledge this in your answer.
        """

        return self.generate_response(prompt)

    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Decompose a complex query into subqueries

        Args:
            query: Complex user query

        Returns:
            List of subqueries with metadata
        """
        prompt = f"""
        Decompose the following complex query about emission factors into simpler subqueries:
        
        Query: "{query}"
        
        Break it down into 2-4 simpler subqueries that can be answered independently.
        For each subquery, provide:
        1. The subquery text
        2. The query type (basic_ef_lookup, regional_comparison, entity_subgraph)
        3. Parameters needed (entity_type, entity_name, region, etc.)
        
        Format the response as a JSON array of objects.
        """

        response = self.generate_response(prompt)

        # Parse JSON from response
        try:
            import json
            import re

            # Try to extract JSON using regex
            json_match = re.search(r"(\[.*\])", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                subqueries = json.loads(json_str)
            else:
                # Fallback to a simple decomposition
                analysis = self.analyze_query(query)
                subqueries = [
                    {
                        "subquery": query,
                        "query_type": "basic_ef_lookup",
                        "params": {
                            "entity_type": analysis.get("entity_type", "unknown"),
                            "entity_name": analysis.get("entity_name", "unknown"),
                            "region": analysis.get("region", "global"),
                        },
                    }
                ]

            return subqueries
        except Exception as e:
            logger.error(f"Error parsing decomposition response: {str(e)}")
            return [
                {
                    "subquery": query,
                    "query_type": "basic_ef_lookup",
                    "params": {
                        "entity_type": "unknown",
                        "entity_name": "unknown",
                        "region": "global",
                    },
                    "error": str(e),
                }
            ]
