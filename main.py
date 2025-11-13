import os
from langchain_community.document_loaders import TextLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
#The langchain imports in the document are no longer functional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


#Configurable Parameters
SPEECH_PATH = "speech.txt"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:7b"
#Loading text from the document
def load_text(path: str) -> str:
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    return docs[0].page_content.strip()

#Checking if database exists,else creating it
def build_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.isdir(CHROMA_DIR):
        print("Loading existing Chroma database")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    print("Building Chroma database for the first time")

    text = load_text(SPEECH_PATH)

    #ONE SINGLE CHUNK BECAUSE THE PROVIDED DOCUMENT IS TOO SMALL FOR CONTEXT
    #HENCE NO SMART CHUNKING PREPROCESSING
    docs = [Document(page_content=text, metadata={"chunk": 0})]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    vectordb.persist()
    print("Chroma database created and persisted.")
    return vectordb

#RAG Pipeline
def create_rag_pipeline(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})
    llm = Ollama(model=OLLAMA_MODEL)

    template = """
Answer the question solely using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided text."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


def main():
    
    print("Type your question below.")
    print('To exit, type quit.\n')

    vectordb = build_vector_db()
    rag = create_rag_pipeline(vectordb)#passing the vector embeddings as context for the RAG Pipeline

    while True:
        question = input("Question> ").strip()

        # EXIT ONLY ON "quit"
        if question.lower() == "quit":
            print("\nGoodbye!")
            break
        answer = rag.invoke(question)

        print("\n--- Answer ---")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
