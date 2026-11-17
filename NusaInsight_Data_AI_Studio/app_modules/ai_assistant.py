import os
import streamlit as st
from groq import Groq


def _groq_setting(name, default=""):
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value or os.getenv(name, default) or "").strip()


def render_ai_assistant():
    """Render the NusaInsight AI Assistant page."""

    st.title("🤖 NusaInsight AI Assistant")
    st.caption("An AI assistant for learning data science and understanding machine learning.")

    # Suggested questions — displayed in one horizontal row
    st.subheader("💡 Suggested Questions")
    st.caption("Select a question or type your own.")

    suggestions = [
        "What is the difference between Regression and Classification?",
        "How does a Gaussian Mixture Model (GMM) work?",
        "How can I interpret model evaluation results?",
    ]

    cols = st.columns(3)
    for i, question in enumerate(suggestions):
        with cols[i]:
            if st.button(
                f"💬 {question}",
                key=f"ai_suggestion_{i}",
                use_container_width=True,
            ):
                st.session_state.ai_suggested_prompt = question

    # Clear conversation
    if st.button("🗑️ Clear Conversation", key="clear_ai_chat"):
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I am the NusaInsight AI Assistant. What would you like to learn today?",
            }
        ]
        st.session_state.pop("ai_suggested_prompt", None)
        st.rerun()

    st.divider()
    st.subheader("💬 Chat")

    # Initialize chat history
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I am the NusaInsight AI Assistant. What would you like to learn today?",
            }
        ]

    # Display previous messages
    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # API key and client
    api_key = _groq_setting("GROQ_API_KEY")
    model_name = _groq_setting("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    if not api_key:
        st.error(
            "GROQ_API_KEY was not found. Save the key in "
            ".streamlit/secrets.toml or a server-side environment variable."
        )
        return

    try:
        client = Groq(api_key=api_key, timeout=30.0, max_retries=1)
    except Exception:
        st.error("The Groq client could not be created. Check that the groq library is installed.")
        return

    system_prompt = (
        "You are NusaInsight AI Assistant, a friendly and patient Data Science tutor. "
        "Explain concepts step by step and clearly, especially for beginners. "
        "You can explain general Data Science topics and help interpret NusaInsight "
        "datasets, charts, and machine-learning results when context is provided. "
        "Respond in simple, professional English while retaining standard technical terms "
        "such as Regression, Classification, Clustering, Random Forest, and F1-score. "
        "Use examples when helpful. Do not claim to see live app data unless it is "
        "included in the conversation."
    )

    # A clicked suggestion becomes the chat input prompt
    suggested_prompt = st.session_state.pop("ai_suggested_prompt", None)
    prompt = st.chat_input("Ask a question about data science...")

    if suggested_prompt:
        prompt = suggested_prompt

    if prompt:
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("The AI is thinking..."):
                try:
                    completion_args = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            *st.session_state.ai_messages,
                        ],
                        "temperature": 0.7,
                        "max_completion_tokens": 3000,
                    }
                    if "gpt-oss" in model_name.lower():
                        completion_args["reasoning_effort"] = "low"
                    response = client.chat.completions.create(**completion_args)
                    response_message = response.choices[0].message
                    answer = response_message.content
                    if not answer or not str(answer).strip():
                        if response_message.tool_calls:
                            answer = "The model requested a tool call, but no tool is enabled in this chat. Please rephrase your question."
                        else:
                            answer = "The model returned an empty response. Please try again or check the selected Groq model and token limit."
                    st.markdown(answer)
                    st.session_state.ai_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    st.error(f"Could not get an AI response: {e}")
