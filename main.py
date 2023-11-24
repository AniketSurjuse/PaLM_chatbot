import langchain_helper
import streamlit as st

st.header("Dumbledore: The PDF Wizard")

# query = st.text_input("Enter your Question here")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

query = st.chat_input("Whats up?")
if query:
    with st.chat_message('user'):
        st.markdown(query)

    st.session_state.messages.append({'role': 'user', 'content': query})

    chain = langchain_helper.get_qa_chain()
    ans = chain(query)
    response = ans['result']
    with st.chat_message('assistant'):
        st.markdown(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})
    



