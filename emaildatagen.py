import pandas as pd
import random

# Templates for each category
urgent_templates = [
    "Hi Team, \n\nWe are experiencing {issue} right now and it's impacting all customers. Need urgent attention! Please prioritize this.\n\nThanks, {name}",
    "Hello Support, \n\nThe {issue} has completely taken down our service. Can you escalate this issue immediately?\n\nRegards, {name}",
    "Dear Support,\n\nCritical alert on {issue} - major outage reported. Requesting urgent resolution.\n\nCheers, {name}"
]

non_urgent_templates = [
    "Hi, \n\nI would like to inquire about {topic}. Could you please share the detailed information?\n\nThanks, {name}",
    "Hello, \n\nI'm considering {topic}. When would be a good time to discuss further?\n\nRegards, {name}",
    "Dear Team,\n\nNeed clarification on {topic}. No immediate rush, but would appreciate a response soon.\n\nThanks, {name}"
]

needs_human_templates = [
    "Hi there, \n\nI tried the automated system but it's not solving my problem related to {issue}. Please connect me to a live agent.\n\nThanks, {name}",
    "Hello Support,\n\nHaving trouble with {issue}. The chatbot couldn't assist. Need a real person to help.\n\nRegards, {name}",
    "Dear Customer Service,\n\nI'm stuck trying to resolve {issue} via the FAQ. Please escalate this to human support.\n\nThanks a lot, {name}"
]

# Noise (typos, casual phrases)
casual_phrases = ["Pls", "ASAP", "Need asap", "Thnx", "Kindly", "Appreciate it", "Need urgent hlp", "thanx", "urgent pls"]
names = ["John", "Emily", "Alex", "Pat", "Chris", "Taylor", "Jordan"]

urgent_issues = ["server outage", "payment gateway failure", "database crash"]
non_urgent_topics = ["subscription renewal", "feature request", "upgrading to premium plan"]
needs_human_issues = ["password reset", "billing discrepancy", "account verification"]

# Helper to randomly insert casual phrases
def maybe_add_noise(text):
    if random.random() < 0.3:  # 30% chance to add noise
        phrase = random.choice(casual_phrases)
        return text.replace("Thanks", phrase).replace("Regards", phrase)
    return text

# Create dataset
def create_realistic_dataset(num_samples_per_class=200):
    data = []

    for _ in range(num_samples_per_class):
        urgent_email = random.choice(urgent_templates).format(issue=random.choice(urgent_issues), name=random.choice(names))
        urgent_email = maybe_add_noise(urgent_email)

        non_urgent_email = random.choice(non_urgent_templates).format(topic=random.choice(non_urgent_topics), name=random.choice(names))
        non_urgent_email = maybe_add_noise(non_urgent_email)

        needs_human_email = random.choice(needs_human_templates).format(issue=random.choice(needs_human_issues), name=random.choice(names))
        needs_human_email = maybe_add_noise(needs_human_email)

        data.append({"Email Text": urgent_email, "Label": "urgent"})
        data.append({"Email Text": non_urgent_email, "Label": "non-urgent"})
        data.append({"Email Text": needs_human_email, "Label": "needs human"})

    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Generate and save
df = create_realistic_dataset()
df.to_csv("./data/train.csv", index=False)
print(df.head())
