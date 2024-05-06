"""
This code loads environment variables using the `dotenv` library and sets the necessary environment variables for Azure services.
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch

from operator import itemgetter

from langchain.schema.runnable import RunnableMap

aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="ada-002",
    openai_api_version="2024-02-15-preview",  # e.g., "2023-12-01-preview"
)

vector_store_address: str = os.getenv("AZURE_SEARCH_ENDPOINT")
vector_store_password: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")

index_name: str = "newindex1"
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=aoai_embeddings.embed_query,
)

createAISearchIndex = False

if (createAISearchIndex):
    # Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
    loader = AzureAIDocumentIntelligenceLoader(file_path="./machinery_manual_1.pdf", api_key = doc_intelligence_key, api_endpoint = doc_intelligence_endpoint, api_model="prebuilt-layout")
    docs = loader.load()

    #Set to True to create the Azure Search index
    createAISearchIndex = True

    # Split the document into chunks base on markdown headers.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    docs_string = docs[0].page_content
    #print(docs_string)
    splits = text_splitter.split_text(docs_string)

    print("Length of splits: " + str(len(splits)))
    #print(splits)

    # Embed the splitted documents and insert into Azure Search vector store


    vector_store.add_documents(documents=splits)

# Retrieve relevant chunks based on the question

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

retrieved_docs = retriever.get_relevant_documents(
    "What are the Objectives of Application of 5S ?"
)

#print(retrieved_docs[0].page_content)

# Use a prompt for RAG that is checked into the LangChain prompt hub (https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=989ad331-949f-4bac-9694-660074a208a7)
prompt = hub.pull("rlm/rag-prompt")
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",  # e.g., "2023-12-01-preview"
    azure_deployment="gpt-4",
    temperature=0,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

output = rag_chain.invoke("What are the Objectives of Application of 5S ?")
print(output)

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain_with_source = RunnableMap(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

output_docs = rag_chain_with_source.invoke("What are the Objectives of Application of 5S ?")

print(output_docs)


