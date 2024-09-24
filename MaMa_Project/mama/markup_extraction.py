def extract_markup(query: str) -> Dict[str, float]:
    """
    Extract markup prompts from the input query.
    
    Args:
        query (str): The input sentence or query.
    
    Returns:
        Dict[str, float]: Extracted markup prompts with weights.
    """
    markup = {
        "positive": 0.8 if "good" in query or "happy" in query else 0.2,
        "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
        "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1
    }
    return markup
