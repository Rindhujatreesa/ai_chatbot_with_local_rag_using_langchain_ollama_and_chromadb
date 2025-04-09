import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma.vectorstores import Chroma

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chromadb(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader("data")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)

# This user-defined function will define custom indices for chunks in the format: "{source}:{page_label}:{chunk_index}"
def define_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page_label")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index = 0
        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

CHROMA_PATH = "chromadb" #chroma db file path. The database will be created at this location.

def add_to_chromadb(chunks: list[Document]):
    
    # use the below code in the terminal to run chroma db
    # chroma run --host localhost --port 8000 --path ./chromadb
    db = Chroma(
        collection_name = "chunks", persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    # define custom indices for the chunks using the `define_chunk_ids()` function
    chunks_with_ids = define_chunk_ids(chunks)

    # This part of the script helps update the database with new documents, by checking if the chunks are present.
    existing_chunks = db.get(include =[])
    existing_ids = set(existing_chunks["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
        
    if len(new_chunks):
        print(f"New {len(new_chunks)} documents added to the DB")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        # adds the new chunk along with the custom index to the database
        db.add_documents(new_chunks, ids = new_chunk_ids)
        print("chunk embedded!")
    else:
        print("No documents to add!")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()