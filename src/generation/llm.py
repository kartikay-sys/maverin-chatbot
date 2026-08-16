import os
import logging
from dotenv import load_dotenv
from src.generation.prompts import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()

def apply_guardrails(response_text: str) -> str:
    """Applies a post-generation deterministic guardrail against knowledge leaks."""
    fallback_string = "The provided 3GPP evidence is insufficient to answer this question."
    lower_text = response_text.lower()
    
    leak_phrases = [
        "general knowledge",
        "general 3gpp knowledge",
        "supplementary information",
        "not sourced from the provided evidence",
        "based on standard 3gpp specifications"
    ]
    
    # 1. Leak phrase detection
    if any(phrase in lower_text for phrase in leak_phrases):
        logger.warning("Guardrail triggered: Leak phrase detected. Truncating response.")
        return fallback_string
        
    # 2. Uncited prose after fallback string
    fallback_lower = fallback_string.lower()
    if fallback_lower in lower_text:
        parts = lower_text.split(fallback_lower, 1)
        if len(parts) > 1:
            after_fallback = parts[1].strip()
            # If there is prose after the fallback, and it lacks [Evidence N] tags, it's a leak
            if after_fallback and "[evidence " not in after_fallback:
                logger.warning("Guardrail triggered: Uncited prose after refusal. Truncating response.")
                return fallback_string

    return response_text

def format_evidence(retrieved_chunks: list[dict]) -> str:
    """Formats the top reranked chunks into a single structured text block for the prompt."""
    evidence_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        evidence_text += f"--- Evidence {i+1} ---\n"
        evidence_text += f"Text: {chunk['text']}\n\n"
    return evidence_text

class NvidiaGenerator:
    """
    Handles answer generation using NVIDIA API models, strictly grounded in the retrieved 3GPP evidence.
    """
    def __init__(self, model_name: str = 'nvidia/nemotron-3-ultra-550b-a55b'):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            logger.error("Valid NVIDIA_API_KEY not found in .env file.")
            raise ValueError("Missing NVIDIA_API_KEY")
            
        import openai
        self.client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        self.model_name = model_name
        logger.info(f"Initialized Generative AI model via NVIDIA API: {model_name}")

    def generate_answer(self, question: str, retrieved_chunks: list[dict]) -> str:
        """Generates a grounded answer based on the user question and the reranked evidence."""
        if not retrieved_chunks:
            return "No relevant 3GPP evidence was found to answer this question."
            
        evidence_block = format_evidence(retrieved_chunks)
        
        system_instructions = RAG_SYSTEM_PROMPT.format(evidence=evidence_block)
        
        logger.info(f"Generating answer using {len(retrieved_chunks)} pieces of evidence...")
        
        # Using stream=False to match the interface of other generators,
        # but including the extra_body parameters for thinking features.
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":1024}
        )
        
        # With enable_thinking=True, the model might put reasoning in the message content 
        # or a custom field. We return the main content here.
        return apply_guardrails(response.choices[0].message.content)
