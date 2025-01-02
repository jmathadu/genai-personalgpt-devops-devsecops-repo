#!/usr/bin/env python3
import os
import argparse
import time
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama

# Constants
model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS


def get_embeddings_model(model_name):
    """
    Tries to load the Hugging Face embeddings model. Falls back to a local model if necessary.
    """
    try:
        print(f"Loading embeddings model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"Failed to load remote model due to: {e}")
        print("Attempting to load the local model...")

        # Fallback to local model path
        local_model_path = os.path.join("models", model_name)
        if os.path.exists(local_model_path):
            try:
                print(f"Loading local model from {local_model_path}")
                return HuggingFaceEmbeddings(model_name=local_model_path)
            except Exception as local_e:
                print(f"Failed to load local model: {local_e}")
                raise RuntimeError("Could not load embeddings model.")
        else:
            raise RuntimeError(
                "Local model not found. Ensure it is downloaded to the 'models' directory."
            )


def main():
    """
    Main entry point for the application.
    """
    # Parse the command line arguments
    args = parse_arguments()

    # Load embeddings model
    embeddings = get_embeddings_model(embeddings_model_name)

    # Set up vector store
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)

    # Set up the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
    )

    # Interactive Q&A loop
    while True:
        query = input("\nEnter a query: ")
        if query.lower() == "exit":
            break
        if not query.strip():
            continue

        # Get the answer
        start = time.time()
        res = qa(query)
        answer = res['result']
        docs = [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the question and answer
        print("\n\n> Question:")
        print(query)
        print(answer)

        # Print the relevant sources
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="privateGPT: Ask questions to your documents without an internet connection, "
                    "using the power of LLMs."
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of input documents used for answers.",
    )
    parser.add_argument(
        "--mute-stream",
        "-M",
        action="store_true",
        help="Use this flag to disable the streaming StdOut callback for LLMs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
