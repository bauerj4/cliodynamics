from typing import List, Dict, Optional, Any
from smolagents import tool
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    pipeline,
)
import torch


def get_smollm_chat_pipeline() -> TextGenerationPipeline:
    """
    Load the default HuggingFace SmolLM-360M-Instruct model with chat formatting.
    """
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def format_chat_prompt(report_text: str) -> str:
    """
    Formats a user/system prompt for SmolLM-Instruct models.
    """
    return (
        "<|im_start|>system\n"
        "You are a geopolitical analyst assistant that summarizes crisis reports.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Summarize the following CrisisWatch reports in a concise, markdown-formatted paragraph:\n\n{report_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@tool
def summarize_reports(
    reports: List[Dict[str, str]],
    db_path: str = "crisiswatch.db",  # unused
    model: Optional[TextGenerationPipeline] = None,
    do_sample: bool = True,
) -> str:
    """
    description: Generates a high-level summary of CrisisWatch reports using a SmolLM instruction-tuned model.

    Args:
        reports: A list of dictionaries with 'title' and 'summary' fields.
        db_path: Unused (placeholder for compatibility).
        model: Optional HuggingFace text-generation pipeline. Defaults to SmolLM-360M-Instruct.
        do_sample: Optional Whether or not to do beam search.

    Returns:
        A markdown-formatted string summarizing the content.
    """
    if not reports:
        return "No reports to summarize."

    if model is None:
        model = get_smollm_chat_pipeline()

    # Concatenate report titles + summaries
    report_text = "\n".join(
        f"- {r.get('title', 'Untitled')}: {r.get('summary', '')}" for r in reports
    )

    # Format for chat-style prompting
    prompt = format_chat_prompt(report_text)

    # Generate completion
    output = model(
        prompt, max_new_tokens=300, do_sample=do_sample, return_full_text=False
    )
    return output[0]["generated_text"].strip()
