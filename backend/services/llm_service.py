"""
LLM Service - Ollama integration for FlowML

Provides AI-powered features:
- Data cleaning suggestions (generates Polars code)
- Results explanation (interprets metrics and confusion matrices)
- Feature importance insights

Safety:
- Token-limited responses
- Deterministic templates
- Human-in-the-loop approval for code execution
"""
import httpx
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from config import settings


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    # Future: OPENAI = "openai"


@dataclass
class LLMResponse:
    """Structured LLM response"""
    success: bool
    content: str
    model: str
    tokens_used: int
    error: Optional[str] = None


@dataclass
class CleaningSuggestion:
    """Data cleaning suggestion with Polars code"""
    issue: str
    description: str
    polars_code: str
    preview_rows: int = 5
    confidence: float = 0.0


@dataclass
class ResultExplanation:
    """Model result explanation"""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    threshold_suggestion: Optional[float] = None


class OllamaClient:
    """
    Ollama API client with retry and timeout handling.
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: float = 120.0,
        max_tokens: int = 2048,
    ):
        self.base_url = base_url or settings.OLLAMA_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system: System prompt for context
            temperature: Creativity (0.0-1.0)
            max_tokens: Max response tokens
        
        Returns:
            LLMResponse with generated content
        """
        max_tokens = max_tokens or self.max_tokens
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                success=True,
                content=data.get("response", ""),
                model=self.model,
                tokens_used=data.get("eval_count", 0),
            )
            
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                tokens_used=0,
                error="Request timed out. The model may be loading or the prompt is too complex.",
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code}")
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                tokens_used=0,
                error=f"HTTP error: {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                tokens_used=0,
                error=str(e),
            )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Chat completion using Ollama.
        
        Args:
            messages: List of {"role": "user|assistant|system", "content": "..."}
            temperature: Creativity (0.0-1.0)
            max_tokens: Max response tokens
        
        Returns:
            LLMResponse with generated content
        """
        max_tokens = max_tokens or self.max_tokens
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                success=True,
                content=data.get("message", {}).get("content", ""),
                model=self.model,
                tokens_used=data.get("eval_count", 0),
            )
            
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return LLMResponse(
                success=False,
                content="",
                model=self.model,
                tokens_used=0,
                error=str(e),
            )
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama"""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()


class LLMService:
    """
    High-level LLM service for FlowML features.
    """
    
    # System prompts
    CLEANING_SYSTEM_PROMPT = """You are a data cleaning expert. You analyze datasets and suggest Polars DataFrame operations to clean and improve data quality.

Rules:
1. Only suggest Polars operations (not pandas)
2. Always use `pl.col()` syntax
3. Provide executable Python code
4. Explain why each cleaning step is needed
5. Be conservative - don't remove data unnecessarily

Response format: JSON with fields:
- issue: Brief description of the data issue
- description: Detailed explanation
- polars_code: Valid Polars code to fix the issue
- confidence: 0.0-1.0 confidence score"""

    EXPLAIN_SYSTEM_PROMPT = """You are a machine learning expert. You analyze model results and provide actionable insights.

Rules:
1. Be concise and practical
2. Focus on business implications
3. Suggest specific improvements
4. Use simple language

Response format: JSON with fields:
- summary: One paragraph overview
- strengths: List of model strengths
- weaknesses: List of areas for improvement
- recommendations: List of actionable next steps
- threshold_suggestion: Optional probability threshold (0.0-1.0) if classification"""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
    
    async def suggest_cleaning(
        self,
        columns: List[str],
        dtypes: Dict[str, str],
        sample_data: List[Dict[str, Any]],
        null_counts: Dict[str, int],
        unique_counts: Dict[str, int],
        num_rows: int,
    ) -> List[CleaningSuggestion]:
        """
        Analyze dataset and suggest cleaning operations.
        
        Args:
            columns: List of column names
            dtypes: Column name -> dtype mapping
            sample_data: Sample rows from the dataset
            null_counts: Column name -> null count
            unique_counts: Column name -> unique value count
            num_rows: Total number of rows
        
        Returns:
            List of CleaningSuggestion with Polars code
        """
        # Build context for LLM
        context = f"""Dataset Analysis:
- Total rows: {num_rows}
- Columns: {len(columns)}

Column Details:
"""
        for col in columns:
            null_pct = (null_counts.get(col, 0) / num_rows * 100) if num_rows > 0 else 0
            unique = unique_counts.get(col, 0)
            context += f"- {col}: {dtypes.get(col, 'unknown')}, {null_pct:.1f}% nulls, {unique} unique values\n"
        
        context += f"\nSample data (first 3 rows):\n{json.dumps(sample_data[:3], indent=2, default=str)}"
        
        prompt = f"""{context}

Analyze this dataset and suggest up to 5 data cleaning operations. For each issue found, provide:
1. What the issue is
2. Why it matters
3. Polars code to fix it

Return a JSON array of suggestions."""

        response = await self.client.generate(
            prompt=prompt,
            system=self.CLEANING_SYSTEM_PROMPT,
            temperature=0.3,  # More deterministic for code
            max_tokens=2048,
        )
        
        if not response.success:
            logger.error(f"LLM cleaning suggestion failed: {response.error}")
            return []
        
        # Parse response
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            suggestions_data = json.loads(content)
            
            if isinstance(suggestions_data, dict):
                suggestions_data = [suggestions_data]
            
            suggestions = []
            for item in suggestions_data[:5]:  # Limit to 5
                suggestions.append(CleaningSuggestion(
                    issue=item.get("issue", "Unknown issue"),
                    description=item.get("description", ""),
                    polars_code=item.get("polars_code", ""),
                    confidence=float(item.get("confidence", 0.5)),
                ))
            
            return suggestions
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Return raw response as single suggestion
            return [CleaningSuggestion(
                issue="Analysis Complete",
                description=response.content,
                polars_code="# See description for suggestions",
                confidence=0.3,
            )]
    
    async def explain_results(
        self,
        problem_type: str,
        metrics: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        confusion_matrix: Optional[List[List[int]]] = None,
        class_labels: Optional[List[str]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> ResultExplanation:
        """
        Explain model results in plain language.
        
        Args:
            problem_type: "classification" or "regression"
            metrics: Model metrics (accuracy, f1, rmse, etc.)
            feature_importance: Feature name -> importance score
            confusion_matrix: For classification
            class_labels: Class names for confusion matrix
            dataset_info: Optional dataset metadata
        
        Returns:
            ResultExplanation with insights
        """
        # Build context
        context = f"""Model Type: {problem_type}

Performance Metrics:
"""
        for metric, value in metrics.items():
            if value is not None:
                context += f"- {metric}: {value:.4f}\n"
        
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            context += f"\nTop 10 Important Features:\n"
            for feat, imp in top_features:
                context += f"- {feat}: {imp:.4f}\n"
        
        if confusion_matrix and class_labels:
            context += f"\nConfusion Matrix (rows=actual, cols=predicted):\n"
            context += f"Labels: {class_labels}\n"
            context += f"{confusion_matrix}\n"
        
        if dataset_info:
            context += f"\nDataset: {dataset_info.get('num_rows', '?')} rows, {dataset_info.get('num_columns', '?')} columns\n"
        
        prompt = f"""{context}

Analyze these model results and provide:
1. A brief summary of model performance
2. Key strengths of the model
3. Areas that need improvement
4. Specific recommendations for the user
5. If classification: suggest an optimal probability threshold

Return as JSON."""

        response = await self.client.generate(
            prompt=prompt,
            system=self.EXPLAIN_SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=1500,
        )
        
        if not response.success:
            logger.error(f"LLM explain failed: {response.error}")
            return ResultExplanation(
                summary="Unable to generate explanation. Please check if Ollama is running.",
                strengths=[],
                weaknesses=[],
                recommendations=["Ensure Ollama is running and try again"],
            )
        
        # Parse response
        try:
            content = response.content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            return ResultExplanation(
                summary=data.get("summary", "No summary available"),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                recommendations=data.get("recommendations", []),
                threshold_suggestion=data.get("threshold_suggestion"),
            )
            
        except json.JSONDecodeError:
            # Return raw response
            return ResultExplanation(
                summary=response.content,
                strengths=[],
                weaknesses=[],
                recommendations=[],
            )
    
    async def generate_feature_engineering(
        self,
        columns: List[str],
        dtypes: Dict[str, str],
        target_column: str,
        problem_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest feature engineering operations.
        
        Returns list of suggested feature transformations with Polars code.
        """
        context = f"""Dataset columns and types:
"""
        for col in columns:
            if col != target_column:
                context += f"- {col}: {dtypes.get(col, 'unknown')}\n"
        
        context += f"\nTarget column: {target_column}"
        context += f"\nProblem type: {problem_type}"
        
        prompt = f"""{context}

Suggest up to 5 feature engineering operations that could improve model performance.
For each suggestion, provide:
1. Name of the new feature
2. Description of what it captures
3. Polars code to create it

Return as JSON array with fields: name, description, polars_code"""

        response = await self.client.generate(
            prompt=prompt,
            system="You are a feature engineering expert. Suggest practical feature transformations using Polars syntax.",
            temperature=0.4,
            max_tokens=1500,
        )
        
        if not response.success:
            return []
        
        try:
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)[:5]
        except:
            return []
    
    async def close(self):
        """Close the LLM client"""
        await self.client.close()


# Global service instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama service status and available models"""
    client = OllamaClient()
    try:
        available = await client.is_available()
        models = await client.list_models() if available else []
        return {
            "available": available,
            "url": client.base_url,
            "default_model": client.model,
            "models": models,
        }
    finally:
        await client.close()
