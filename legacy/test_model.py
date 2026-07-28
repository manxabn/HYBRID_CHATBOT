from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="C:/Users/manxa/OneDrive/Desktop/thesis/database_approach/models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.7
)

response = llm("What is the capital of France?")
print(response)
