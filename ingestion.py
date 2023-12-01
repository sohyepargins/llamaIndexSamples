from dotenv import load_dotenv # This function will take the env variables that I wrote in .env
import os 
from pathlib import Path
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext    
)

from llama_index.vector_stores import PineconeVectorStore
import pinecone


# print("Current Working Directory:", os.getcwd())

dotenv_path = '/Users/sohyepark/Documents/llamaindex-helloworld/documentation-helper-llamaindex/.env'
load_dotenv(dotenv_path)
print(f"Pinecone API Key: {os.getenv('PINECONE_API_KEY')}")
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"], 
    environment=os.environ["PINECONE_ENV"]
    )


os.chdir('/Users/sohyepark/Documents/llamaindex-helloworld/documentation-helper-llamaindex')

if __name__ == '__main__':
    print("Going to ingest Pinecone documentation")
    
    ######### Pipe 1: Load the document #######################################################################     
    UnstructuredReader = download_loader('UnstructuredReader')
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs", file_extractor={".html": UnstructuredReader(),}
    )
    documents = dir_reader.load_data()
    #Now let's split the documents into nodes
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
                    # we didn't pass text splitter so it will be split by default which is \n
    # nodes = node_parser.get_nodes_from_documents(documents=documents)
    
    
    ######### Pipe 2: Up to Service context (Embedding) #######################################################################   
    # Step 1: Call the foundation model you like 
    llm = OpenAI(model="gpt-3.5-turbo", temperature = 0)  #temperature = 0 (no funcky or no creative models)
    #Step 2: Create embedding (Here we use Open AI embeddings)
    embed_model = OpenAIEmbedding(model = "text-embedding-ada-002", embed_batch_size=100)
    #Step 3: Create ServiceContext that holds information about my data ingestion pipeline
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    
    
    ######### Pipe 3: Up to Storage Context (Vector Store) #######################################################################
    # Creating a vector store setup
    index_name = "llamaindex-documentation-helper"
    pinecone_index = pinecone.Index(index_name=index_name)      #This object is Pinecone SDK class. The reason we're not throwing API key is it is already set in the .env
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index) # Instance of Pinecone vectorstore class (Wrapper on the pinecone SDK)
    #Last step is to create a storage context as we did service context above
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    ######### Now Pipeline: Connecting the pipes above #######################################################################
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,         
    )
    
    print("finished ingesting...")