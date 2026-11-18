from google import genai
from google.genai import types

# ... di dalam fungsi analisis AI:
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[...], # isi prompt dan gambar milikmu
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        max_output_tokens=300,
    )
)