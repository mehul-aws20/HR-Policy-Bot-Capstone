import streamlit as st
import os
import uuid
from agent import build_app

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="centered"
)

@st.cache_resource
def load_agent():
    """Load all expensive resources once and cache them."""
    return build_app()

# Load cached app
app = load_agent()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.title("💼 HR Policy Assistant")
    st.markdown("**Your 24/7 HR handbook companion**")
    st.markdown("---")
    st.markdown("**Topics I can help with:**")
    topics = [
        "📅 Annual & Sick Leave",
        "🏠 Work From Home Policy",
        "💰 Payroll & Salary",
        "🏥 Health Insurance",
        "⭐ Performance Appraisal",
        "🚪 Resignation & Exit",
        "🧾 Expense Reimbursement",
        "📋 Code of Conduct",
        "🎓 Training & Learning",
        "📆 Current Date & Deadlines",
    ]
    for t in topics:
        st.markdown(f"- {t}")
    st.markdown("---")
    st.caption(f"Session ID: {st.session_state.thread_id[:8]}...")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Main header
st.title("HR Policy Assistant")
st.caption("Ask me anything about company HR policies. I only answer from the official handbook.")

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask an HR policy question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Checking the HR handbook..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = {
                "question": prompt,
                "messages": st.session_state.messages[:-1],
                "route": "",
                "retrieved": "",
                "sources": [],
                "tool_result": "",
                "answer": "",
                "faithfulness": 0.0,
                "eval_retries": 0,
                "user_name": ""
            }
            result = app.invoke(initial_state, config=config)
            answer = result["answer"]

        st.markdown(answer)

        # Show sources if available
        if result.get("sources"):
            with st.expander("📚 Sources from HR Handbook"):
                for src in result["sources"]:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({"role": "assistant", "content": answer})