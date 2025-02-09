## 1. Ingest PDF files
# 2. Extract text from PDF files and split into small chunks
# 3. Send the text chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. Retrieve the similar documents and present them to the user

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import glob
import ollama


MODEL_NAME = "DR.TRUTH"  # Change this to your desired model name
MODEL_TONE = "Humorous, satire and sarcasm but end with honesty"  # Change this to your desired tone


doc_paths = glob.glob("./data/*.pdf")
model = "llama3.2"
vector_db = None

# Local PDF file uploads
for doc_path in doc_paths:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading...")

    # Split and chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)
    print("done splitting...")

    if vector_db is None:
        ollama.pull("nomic-embed-text")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="simple-rag",
        )
    else:
        vector_db.add_documents(documents=chunks)
    print(f"Done adding {doc_path} to vector database...")



# Set up out model to use
llm = ChatOllama(model=model, streaming=True)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=f"""You are {MODEL_NAME}, an AI language model assistant. Your tone is {MODEL_TONE}. 
    You are asked a question by a user. You are expected to provide an answer to the question.
    QUESTION: {{question}}
    """
)

num_chunks = vector_db._collection.count()  

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(search_kwargs={"k": num_chunks}),  # Setting k dynamically
    llm,
    prompt=QUERY_PROMPT
)

template = f"""You are {MODEL_NAME}, an AI language model assistant. Your tone is {MODEL_TONE}. 
Answer the question based ONLY on the following context:
{{context}}
QUESTION: {{question}}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streaming Function
def stream_response(response):
    for chunk in response:
        print(chunk, end="", flush=True)  # Print chunked response without newlines

while True:
    user_input = input("\nEnter your question (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Query the vector database with the user's input and stream response
    response = chain.stream(input=(user_input,))
    stream_response(response)  # Stream the LLM output in real-time
    print("\n")  # Add newline after streaming completion
