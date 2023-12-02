from dotenv import load_dotenv
import os
load_dotenv()
import pinecone
from llama_index import VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore #In order to create the vector store object, we need PineconeVectoreStore wrapper
                                        #which wraps around Pinecone SDK
from llama_index.callbacks import LlamaDebugHandler, CallbackManager
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext    
)
from llama_index.llms import OpenAI
import streamlit as st

# if __name__ == '__main__':
#     print("RAG...")
    
#     #We need to initiate vector store index (= 1 vector space = "class" in Weaviate = collection in nonSQL)
#     #We're using pinecone, so we need to create a pinecone vectorstore object.
#     pinecone_index = pinecone.Index(index_name="llamaindex-documentation-helper")
#     vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
#     #Add callbacks to the ServiceContext
#     llama_debug = LlamaDebugHandler(print_trace_on_end=True)
#     callback_manager= CallbackManager(handlers=[llama_debug])
#     service_context = ServiceContext.from_defaults(callback_manager=callback_manager)  
        
#     index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context) 
#         #an instance from VectorStoreIndex class; In order to create the instance, we used from_vector_store method
#         #under the hood, simply being connected to Pinecone SDK
        
#     query = "What is a LlamaIndex query engine?"
#     query_engine = index.as_query_engine() #This query engine will do all the RAG operation
#     response = query_engine.query(query)
#     print(response)


#To run in Terminal, run in the terminal %streamlit run main.py to check

print("***Streamlit LlamaIndex Documentation Helper")
# The function will return us to the VectorStoreIndex of LlamaIndex
@st.cache_resource(show_spinner=False) # Also make sure to move the pinecone client into the function bc we want to run only when the function is called.
# We're wrapping the function below
def get_index()-> VectorStoreIndex:  
    
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENV"])  

    pinecone_index = pinecone.Index(index_name="llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    #Add callbacks to the ServiceContext
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager= CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)  
        
    return  VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

index = get_index() #remember the output of get_index() is VectorStoreIndex
        #By default, streamlit is a while loop; Keep running main.py over and over again.
        #To pervent this, we're getting cacheing of the result of get_index (i.e., VectorStoreIndex here)
        

# Stream lit session states helps the varibles to persist, instead of being overwritten every time a loop is running.
# Every state in the session ("session state") has a data type of dictionary.
# That's why we're using .keys() method below.     
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="react", verbose = True)
                    #We created a session state called "chat_engine" and it's bascially VectoreStoreIndex.as_chat_engine

st.set_page_config(page_title="Chat with LlamaIndex docs powered by llamaIndex ",
                       page_icon="🦙",
                       layout = "centered",
                       initial_sidebar_state="auto",
                       menu_items=None
                       )
st.title("Chat with LlamaIndex docs 🦙")
    

#If session state named "messages" doesn't exist, we want to initialize it.
if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library." 
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
  
# Basically priniting out the session state variable 
for message in st.session_state.messages:
    with st.chat_message(message["role"]): #Steramlit's context manager chat_message will return an avatar.
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant": #is it's a user message
    with st.chat_message("assistant"): #We want to put assistant avatar in the answer
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt) #access the chat engine
            st.write(response.response) #get the response
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node {i+1}: score = {node.score}")
                    st.write(node.text)
            message = {        #The following lines are appending the results to our "messages" session state.
                "role": "assistant",
                "content":response.response
            }
            st.session_state.messages.append(message)

