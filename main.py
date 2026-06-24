import streamlit as st
import uuid
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="Verna Chat App", page_icon=":key:", layout="wide")

# ----- Helpers --------
def create_session():
    return {"id": str(uuid.uuid4()), "name": "New Chat", "messages": [], "memory": None}

def get_current():
    return next((s for s in st.session_state.sessions if s["id"] == st.session_state.active_id), None)

# --- Bootstrap session state -----

if "sessions" not in st.session_state:
    first = create_session()
    st.session_state.sessions = [first]
    st.session_state.active_id = first["id"]

# -- Sidebar ---
with st.sidebar:

    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key here")
    model_name = st.selectbox("Model",options=["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0,help="Select the OpenAI model to use for chat."    )
    st.markdown("---")

    #New Chat Button
    if st.button("New Chat"):
        new = create_session()
        st.session_state.sessions.insert(0,new)
        st.session_state.active_id = new["id"]
        st.rerun()
    
    st.markdown("## Chat History ")

    for session in st.session_state.sessions:
        is_active = session["id"] == st.session_state.active_id
        label = (" " if is_active else "") + session["name"]
        #Highlight active chat;disbale its button to avoid re-clicking
        if st.button(label, key=session["id"],use_container_width=True, disabled=is_active):
            st.session_state.active_id = session["id"]
            st.rerun()

st.title("Verna Chat App")

current = get_current()

if current is None:
    st.error("No active chat session found.")
    st.stop()

#Display Chat history for current session
for message in current["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Chat Input 

user_input = st.chat_input("Type your message here...")

if user_input:
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    #Auto name the session from its first user message
    if not current["messages"]:
        current["name"] = user_input[:40] + "..." if len(user_input) > 40 else ""

    if current["memory"] is None:
        current["memory"] = ConversationBufferMemory(return_messages=True)

    llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0)

    chain = ConversationChain(llm=llm, memory=current["memory"],verbose=False)

    current["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Verna is thinking..."):
            try:
                response = chain.predict(input=user_input)
                st.markdown(response)
                current["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = str(e)
                if "AuthenticationError" in error_message:
                    st.error("Authentication Error: Please check your OpenAI API key.")
                else:
                    st.error("An unexpected error occurred. Please try again. : {error_message}")
                
          
