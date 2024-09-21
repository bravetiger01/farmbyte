"""
Install the Google AI Python SDK

$ pip install google-generativeai
"""

import os
import google.generativeai as genai

# Set API key as environment variable (replace with your actual path)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Project CS\AGribyte\client_secret_979856931806-kqceo61kebo7gr82k4f32g73q5rg3avv.apps.googleusercontent.com.json"

# Configure the SDK
genai.configure(api_key="AIzaSyCwAeaEg5YM3SKRK8yj2075lbTjaCI-hoE")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("")

print(response.text)