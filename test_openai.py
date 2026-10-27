
from openai import OpenAI

client = OpenAI()

try:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Reply with exactly: ChurnShield AI connected!"
    )
    print(response.output_text)

except Exception as e:
    print(f"API test failed: {type(e).__name__}: {e}")
