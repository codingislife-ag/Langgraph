import streamlit as st
from langgraph_backend_streaming import chatbot
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
    
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable': {'thread_id': 'thread-1'}},
                stream_mode='messages'
            )
        )
    st.session_state['message_history'].append({'role': 'assisstant', 'content': ai_message})