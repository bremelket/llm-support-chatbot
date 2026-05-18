import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from sentence_transformers import SentenceTransformer
from typing import TypedDict
from retriever import load_vectorstore, retrieve

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
index, metadata = load_vectorstore()
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- State ---
class AgentState(TypedDict):
    query: str
    context: str
    answer: str


# --- Nodes ---
def retrieve_node(state: AgentState) -> AgentState:
    results = retrieve(state["query"], model, index, metadata, top_k=3)
    context = "\n\n".join([r["text"] for r in results])
    return {**state, "context": context}


def answer_node(state: AgentState) -> AgentState:
    prompt = f"""You are a helpful support assistant for Nexus, a project management SaaS.
Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{state["context"]}

Question: {state["query"]}

Answer:"""

    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


# --- Graph ---
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    questions = [
        "How do I reset my password?",
        "What is included in the Pro plan?",
        "How do I connect Slack to Nexus?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = app.invoke({"query": question, "context": "", "answer": ""})
        print(f"A: {result['answer']}")