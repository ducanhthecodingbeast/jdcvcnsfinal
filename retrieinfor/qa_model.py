"""
Shared QA model loader for information retrieval from CV and JD texts.
Uses ahotrod/electra_large_discriminator_squad2_512 for question answering.

All questions are in Vietnamese because the project data (CVs and JDs) is in Vietnamese.
The model performs extractive QA, so answers will be spans extracted directly from
the Vietnamese context, ensuring Vietnamese output.
"""

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model/tokenizer instances
_model = None
_tokenizer = None
_device = None

def get_model_and_tokenizer():
    """
    Load and return the QA model and tokenizer. Uses singleton pattern.
    """
    global _model, _tokenizer, _device
    if _model is None or _tokenizer is None:
        logger.info("Loading QA model: ahotrod/electra_large_discriminator_squad2_512")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModelForQuestionAnswering.from_pretrained(
            "ahotrod/electra_large_discriminator_squad2_512"
        )
        _tokenizer = AutoTokenizer.from_pretrained(
            "ahotrod/electra_large_discriminator_squad2_512"
        )
        _model = _model.to(_device)
        _model.eval()
        logger.info(f"QA model loaded on device: {_device}")
    return _model, _tokenizer, _device

def extract_answer(context: str, question: str, max_length: int = 512) -> str:
    """
    Extract answer from context using QA model.

    Args:
        context: The text to extract from (CV or JD content) - expected to be Vietnamese
        question: The question to ask about the context (in Vietnamese)
        max_length: Max context length (model limit)

    Returns:
        Extracted answer (Vietnamese span from context) or empty string if no good answer
    """
    if not context or not isinstance(context, str):
        return ""

    context = context.strip()
    if len(context) > max_length:
        context = context[:max_length]

    try:
        model, tokenizer, device = get_model_and_tokenizer()

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        if answer_end <= answer_start:
            return ""

        answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        if answer and len(answer.strip()) > 0:
            return answer.strip()
        else:
            return ""

    except Exception as e:
        logger.warning(f"QA extraction failed for question '{question[:50]}...': {e}")
        return ""

def extract_multiple_answers(
    context: str,
    questions: List[str],
    combine: bool = True
) -> str:
    """
    Ask multiple questions (in Vietnamese) and combine answers.

    Args:
        context: Text to extract from (Vietnamese)
        questions: List of questions in Vietnamese
        combine: If True, join non-empty answers with "; "

    Returns:
        Combined or first valid answer (Vietnamese text)
    """
    answers = []
    for q in questions:
        ans = extract_answer(context, q)
        if ans:
            answers.append(ans)

    if not answers:
        return ""

    if combine:
        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for a in answers:
            a_lower = a.lower()
            if a_lower not in seen:
                seen.add(a_lower)
                unique_answers.append(a)
        return "; ".join(unique_answers)
    else:
        return answers[0] if answers else ""

def batch_extract(
    texts: List[str],
    question: str,
    batch_size: int = 32
) -> List[str]:
    """
    Extract answers for a batch of texts with the same Vietnamese question.
    """
    results = []
    pipeline = get_qa_pipeline()

    for i, text in enumerate(texts):
        if i % 100 == 0:
            logger.info(f"Processing {i}/{len(texts)}")
        ans = extract_answer(text, question)
        results.append(ans)

    return results