import os
from dotenv import load_dotenv

try:
    from groq import Groq
except Exception:
    Groq = None

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")


def test_groq():
    if Groq is None:
        print("Groq package not installed. Run: pip install groq")
        return

    if not api_key:
        print("GROQ_API_KEY not set in .env file")
        return

    client = Groq(api_key=api_key)

    models_to_try = [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.1-8b-instant"
    ]

    for model in models_to_try:
        print(f"\n--- Attempting connection with: {model} ---")
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Reply with 'Groq Active'"}
                ],
                model=model,
            )

            print(f"✅ Success! {model} says:")
            print(chat_completion.choices[0].message.content)
            return

        except Exception as e:
            print(f"❌ {model} failed: {e}")

    print("\nAll models failed.")


if __name__ == "__main__":
    test_groq()