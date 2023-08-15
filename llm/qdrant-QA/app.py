import os
import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

client = qdrant_client.QdrantClient(os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY"))


# create a collection
def create_collection(name):
    collection_config = qdrant_client.http.models.VectorParams(
        size=1536, distance=qdrant_client.http.models.Distance.COSINE
    )
    client.recreate_collection(collection_name=name, vectors_config=collection_config)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunk(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(collection_name):
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(
        client=client, collection_name=collection_name, embeddings=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(emory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Flourish QA on resource", page_icon="📄")
    st.header("QA with resources 📄")
    asking_collection_name = st.text_input("collection name:", max_chars=20)
    asking_text = st.text_input("Ask away:")
    if st.button("Generate respone"):
        with st.spinner("generation response"):
            if asking_collection_name:
                # create vector store
                vector_store = get_vectorstore(asking_collection_name)

                # create chain
                qa = RetrievalQA.from_chain_type(
                    llm=OpenAI(),
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(),
                )

                if asking_text:
                    st.write(f"Question: {asking_text}")
                    answer = qa.run(asking_text)
                    st.write(f"Answer: {answer}")

    with st.sidebar:
        with st.expander("Show collection list on Qdrant DB"):
            if st.button("show"):
                st.code(
                    client.get_collections().json(),
                )
        # To create new collection
        with st.expander("Create new collection on Qdrant DB"):
            collection_name = st.text_input("Name of the collection", key="cl_ti")
            if st.button("Add", key="cl"):
                with st.spinner("Adding"):
                    create_collection(collection_name)
                    success = f"Created new collection named {collection_name}"
                    st.success(success)

        # To upload pdf in a collection
        with st.expander("Upload new pdf on Qdrant DB"):
            key = "pdf_uploader"
            collection_name = st.text_input(
                placeholder="flourish", label="Collection name", key=key + "ti"
            )
            pdf_docs = st.file_uploader(
                "Upload your PDFs and press process", accept_multiple_files=True
            )
            if st.button("Process", key=key + "bt"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunk = get_text_chunk(raw_text)

                    vectore_store = get_vectorstore(collection_name)
                    # add to vector store
                    vectore_store.add_texts(text_chunk)

        # To upload text in a collection
        with st.expander("Upload text on Qdrant DB"):
            key = "text_uploader"
            collection_name = st.text_input(
                placeholder="flourish", label="Collection name", key=key + "ti"
            )
            text_input = st.text_area(label="Text to upload")
            if st.button("Process", key=key + "bt"):
                with st.spinner("Processing"):
                    text_chunk = get_text_chunk(text_input)

                    vectore_store = get_vectorstore(collection_name)
                    # add to vector store
                    vectore_store.add_texts(text_chunk)


if __name__ == "__main__":
    main()
