import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from agent import build_graph

st.set_page_config(page_title="Nexus Support Chat", page_icon="💬")
st.title("💬 Nexus Support Chatbot")
st.caption("Ask me anything about Nexus — billing, integrations, account, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "app" not in st.session_state:
    st.session_state.app = build_graph()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.app.invoke({
                "query": prompt,
                "context": "",
                "answer": ""
            })
            answer = result["answer"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})