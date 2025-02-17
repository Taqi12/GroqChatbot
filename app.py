import openai
import os
import gradio as gr
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY") # replace with your own api

# Ensure API key is loaded
if API_KEY is None:
    raise ValueError("API Key not found. Please check your .env file in Hugging Face settings.")

# Set up the Groq API client
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")

def chat_with_groq(user_input, chat_history=[]):
    """Function to send messages to the Groq API and receive responses."""
    try:
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        
        # Add previous chat history
        for user, bot in chat_history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": bot})
        
        # Add latest user message
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Use "llama3-70b-8192" for better responses
            messages=messages,
            temperature=0.7
        )

        bot_response = response.choices[0].message.content
        chat_history.append((user_input, bot_response))
        return bot_response, chat_history

    except Exception as e:
        return f"Error: {e}", chat_history

# Gradio Chatbot Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>🤖 Groq-Powered Chatbot</h1>")
    
    chatbot = gr.Chatbot(label="Groq Chatbot")
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear Chat")

    def respond(user_input, chat_history=[]):
        bot_response, chat_history = chat_with_groq(user_input, chat_history)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), [], [msg, chatbot])

# Launch the Gradio app
demo.launch()
