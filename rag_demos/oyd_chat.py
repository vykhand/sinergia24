import os

# load dotenv
from dotenv import load_dotenv
from openai import AzureOpenAI
from rag_demos import zakon_index as ZI
from rag_demos.openai_helpers import get_openai_response
import json

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview"
)

DEPLOYMENT_LIST = ["gpt-4o", "gpt4-turbo", "gpt-4v"]
# read prompt from prompt.txt
with open("chat_system_prompt.txt") as f:
    prompt = f.read()
# with open("advanced_prompt.txt") as f:
#     prompt = f.read()

def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


def respond(msg, chat_history, model, temperature, top_p, top_k, query_type):
    body = ZI.extra_body.copy()
    body["data_sources"][0]["parameters"]["query_type"] = query_type
    body["data_sources"][0]["parameters"]["top_n_documents"] = top_k
    if query_type == "vector_semantic_hybrid":
        body["data_sources"][0]["parameters"]["semantic_configuration"] = "my-semantic-config"
    bot_response, full_js = get_openai_response(messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": msg}
    ], body=ZI.extra_body, model=model, temperature=temperature, top_p=top_p)
    chat_history.append({"role": "user", "content": msg})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history, json.dumps(full_js)


if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Accordion("Settings", open=False):
                model_name_ddn = gr.Dropdown(choices=DEPLOYMENT_LIST, value="gpt-4o")
                temperature_sldr = gr.Slider(label="Temp", minimum=0.0, maximum=0.9, value=1.0, step=0.1)
                top_p_sldr = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Top P")
                top_k_sldr = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K")
                query_type_ddn = gr.Dropdown(
                    choices=["vector_semantic_hybrid", "vector_simple_hybrid", "vector", "simple"],
                    value="vector_semantic_hybrid", label="Query Type")

        with gr.Tab("QA"):
            with gr.Row():
                bot_input_tb = gr.Textbox(label="Input", value="what is the punishment  for kicking hedgehog?")
                bot_btn = gr.Button("Submit", scale=0)
            with gr.Row():
                bot = gr.Chatbot(type="messages", height=700)
                full_js = gr.Json(label="Full JSON")
            with gr.Row():
                bot_clear_btn = gr.Button("Clear")

        # (model, is_stream, temperature, history)

        bot_btn.click(respond, [bot_input_tb, bot, model_name_ddn, temperature_sldr, top_p_sldr, top_k_sldr, query_type_ddn ], [bot_input_tb, bot, full_js],
                      queue=False)

    demo.launch(share=True)
