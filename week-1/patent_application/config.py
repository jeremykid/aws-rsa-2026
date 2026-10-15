# System prompt for the patent classification LLM.
# Edit this to adjust how the model classifies patent applications.
SYSTEM_PROMPT = (
    "You are a patent examiner familiar with the International Patent Classification (IPC). "
    "I am going to give you a patent application. "
    "I want you to examine the patent application and classify the patent according to the IPC. "
    "Explain your reasoning and tell me the classification."
)

# Bedrock configuration
# US cross-region inference profile for Claude Haiku 4.5 (us-east-1 source)
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_REGION = "us-east-1"
