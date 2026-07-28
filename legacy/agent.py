# agent.py

from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, AgentType
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from knowledgebase import KnowledgeBase

# def create_search_tool(kb: KnowledgeBase) -> Tool:
#     """
#     Tool that the agent can call to query the knowledge base.
#     """
    #def _search_tool(query: str) -> str:
    #     results = kb.query(query_text=query, n_results=3)
    #     docs = results.get("documents", [[]])[0]
    #     metadatas = results.get("metadatas", [[]])[0]

    #     # Combine them as needed
    #     combined = ""
    #     for doc, meta in zip(docs, metadatas):
    #         combined += f"Document:\n{doc}\nMetadata: {meta}\n\n"
    #     return combined

    # return Tool(
    #     name="KnowledgeBaseSearch",
    #     func=_search_tool,
    #     description="Use this tool to search for relevant context in the knowledge base."
    # )
def create_search_tool(kb: KnowledgeBase) -> Tool:
    """
    Tool that the agent can call to query the knowledge base.
    """
    def _search_tool(query: str) -> str:
        results = kb.query(query_text=query, n_results=3)
        if isinstance(results, str):
            return results  # Handle string responses
        if not results or not results["documents"][0]:
            return "No relevant data found in the knowledge base."

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Format the retrieved results
        response = "\n".join([f"{doc}\nMetadata: {meta}" for doc, meta in zip(docs, metadatas)])
        return response

    return Tool(
        name="KnowledgeBaseSearch",
        func=_search_tool,
        description="Search the knowledge base for relevant information."
    )


def create_llm_agent(kb: KnowledgeBase):
    """
    Create a LangChain agent with local LLaMA or GPT4All model.
    """
#     template = """You are a helpful AI assistant for a university. 
# When you are unsure or need more context, you MUST use the tool: {tool_names}.
# Answer as accurately as possible.

# You have access to the following tool:
# ---
# {tool_names}
# ---

# Use the following format:
# Question: the user's question
# Thought: your internal reasoning
# Action: the action to take (must be one of [KnowledgeBaseSearch])
# Action Input: the input to the action
# Observation: the result of the action
# Thought: your internal reasoning
# Final Answer: the final answer to the user's question

# Begin!

# Question: {input}
# {agent_scratchpad}"""
    template = """
You are a helpful AI assistant for a university. You have EXACTLY ONE tool:
- KnowledgeBaseSearch(query: str)

When you need more context, use the tool in the EXACT format:

Question: {input}
Thought: (reasoning)
Action: KnowledgeBaseSearch
Action Input: (the EXACT string to search)
Observation: (result from the tool)
Thought: (further reasoning)
Final Answer: (the best possible answer)

Begin now!

Question: {input}
{agent_scratchpad}
    """.strip()

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad"]
    )

    # Initialize the local LLaMA model (or GPT4All, etc.)
    llm = LlamaCpp(
        model_path="C:/Users/manxa/OneDrive/Desktop/thesis/database_approach/models/llama-2-7b-chat.Q4_K_M.gguf",  # e.g. "/models/llama-7b.ggmlv3.q4_0.bin"
        n_ctx=1024,  # Reduce context size
        n_batch=16,   # Process smaller chunks
        n_threads=6, # Use more CPU threads
        temperature=0.1
    )

    search_tool = create_search_tool(kb)

    from langchain.agents import initialize_agent
    agent = initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        prompt=prompt,
        handle_parsing_errors=True  # Allow error handling to avoid crashes
    )
    return agent
