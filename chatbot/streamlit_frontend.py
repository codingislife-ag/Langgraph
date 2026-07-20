import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict -> this will only reset when we manually refresh the page
# otherwise the messages will accumulate 
# session state -> itself a dictionary
CONFIG = {"configurable": {'thread_id': 'thread_1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# message_history = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role': 'user', 'content': 'Hi'}
# {'role': 'assisstant', 'content': 'Hello'}

user_input = st.chat_input('Type Here')

if user_input:

    # first add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response['messages'][-1].content
    st.session_state['message_history'].append({'role': 'assisstant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)