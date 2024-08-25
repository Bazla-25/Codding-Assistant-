from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool 
from langchain.prompts import PromptTemplate
from scrap_article_content import fetch_news, fetch_url
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def connect_to_vstore():
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    desired_namespace = os.getenv("ASTRA_DB_KEYSPACE")

    if desired_namespace:
         ASTRA_DB_KEYSPACE = desired_namespace
    else: 
         ASTRA_DB_KEYSPACE = None

    vstore = AstraDBVectorStore(
         embedding = embeddings,
         collection_name = "github_Rag_agent",
         api_endpoint = ASTRA_DB_API_ENDPOINT,
         token = ASTRA_DB_APPLICATION_TOKEN,
         namespace = ASTRA_DB_KEYSPACE,
    )
    return vstore

vstore = connect_to_vstore()
# ask the user to update the issues


Google_API_KEY = os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model='gemini-pro')

# Fetch and process the article
print("This is a BBC Supported new article! ")
topic = input("please enter topic \n")
print("your topic is: ", topic)
title = input( "please enter title for the article!")
print("\n your title is:", title)
url = fetch_url(topic,title)
document = fetch_news(url)

summarization_prompt_template = PromptTemplate(
    input_variables=["article_content"],
    template="Summarize the following article: {article_content}"
)

summarization_chain = summarization_prompt_template | llm | StrOutputParser()

# Question-Answering Prompt Template
qa_prompt_template = PromptTemplate(
    input_variables=["article_content", "question"],
    template="Based on the article content: {article_content}, answer the following question: {question}"
)
qa_chain =  prompt=qa_prompt_template | llm | StrOutputParser()

#=================================================
add_to_vectorstore = input("Do you want to add this article to your collection?  ").lower() in ["yes", "y"]
if add_to_vectorstore:
    
    vstore = connect_to_vstore()  # Connect to vector store

    # Add document to vector store without deleting previous entries
    vstore.add_documents(document)

    # Step 2: Perform a similarity search
    query = "climate change"
    results = vstore.similarity_search(query, k=3)
#     for res in results:
#         print(f"* {res.page_content} {res.metadata}")

    # Step 3: Create a retriever tool
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        "Article_summarizer",  # Name for tool
        "Search for information about articles. For any questions about articles, you must use this tool!",  # Tool description
    )

if retriever_tool:
    print("Retriever tool created successfully!")


tools = [
    Tool(
        name="SummarizeArticle",
        func=lambda input_data: summarization_chain.invoke({
            "article_content": input_data["document"]  # Assuming input_data["document"] has the content
        }),
        description="Summarizes an article given its content."
    ),
    Tool(
        name="AnswerQuestionsAboutArticle",
        func=lambda input_data: qa_chain.invoke({
            "article_content": input_data["document"],  # Replacing the URL fetch with direct document content
            "question": input_data["question"]
        }),
        description="Answers questions about an article given its content and a question."
    ),
    Tool(
         name="RetrieveRelevantContent",
         func=lambda input_data: retriever_tool.func({
            "document": input_data["document"],  # Pass the document to the retriever tool
            "query": input_data["query"]  # Optional: you can pass a query to search for specific content
        }),
        description="Retrieves relevant content from the article based on a query."        
     )
]




summary = tools[0].func({"document": document})
print("Summary:", summary)

question = "What are the key points of the article?"
answer = tools[1].func({"document": document, "question": question})
print("Answer:", answer)


retrieval_output = tools[2].func({"document": document, "query": query})


