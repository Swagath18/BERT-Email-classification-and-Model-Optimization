import re

def preprocess_email(text: str) -> str:
    # Lowercase everything
    text = text.lower()

    # Remove common greetings
    greetings = [
        "hi team", "hello team", "dear team",
        "hope you are doing well", "good morning", "good evening",
        "greetings", "to whom it may concern"
    ]
    for greet in greetings:
        text = text.replace(greet, "")

    # Remove common sign-offs
    sign_offs = [
        "thanks", "regards", "best regards", "sincerely",
        "warm regards", "thank you", "cheers", "kind regards"
    ]
    for sign in sign_offs:
        text = text.replace(sign, "")

    # Remove email footers or signatures (very basic)
    text = re.sub(r"--[\s\S]*", "", text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
