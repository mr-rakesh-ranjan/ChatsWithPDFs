from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All, LlamaCpp
from sentence_transformers import SentenceTransformer
import os
import argparse
import time

load_dotenv()

model_path = os.environ.get('MODEL_PATH')
model_n_ctxos = os.environ.get('MODEL N CTX')
model_n_batch = int (os.environ.get("MODEL_N_BATCH", 8)) 
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def main():
#Parse the command line arguments
    args = parse_arguments()
    # embeddings Hugging FaceEmbeddings (model_name=embeddings_model_name) 
    FAISS_INDEX_PATH = "./faiss-index-250"
    # embeddings =  HuggingFaceEmbeddings(model_name = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
    # embeddings = HuggingFaceEmbeddings(model_name = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')) 
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # embeddings = HuggingFaceEmbeddings(model) 
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # embeddings = model.encode("My name is rakesh")
    faiss_index = FAISS.load_local (FAISS_INDEX_PATH, embeddings)
    retriever = faiss_index.as_retriever (search_kwargs={"k": target_source_chunks})

    #activate/deactivate the streaming StdOut callback for LLMS
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    #Prepare the LLM
    llm = GPT4All (model=model_path, max_tokens=1000, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    qa = RetrievalQA.from_chain_type (llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    # questions and answers in loop
    while True:
        query=input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        #Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs =  res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        #Print the result
        print("\n\n> Queation:")
        print (query)
        print (f"\n> Answer (took {round (end - start, 2)} s.):") 
        print (answer)



def parse_arguments():
    parser = argparse.ArgumentParser (description='privateGPT: Ask questions on the document privately,' 'using the power of LLMs and gpt4all.')
    parser.add_argument ("--hide-source", "-S", action='store_true', 
                        help = 'Use this flag to disable printing of source documents used for answers.') 
    parser.add_argument ("--mute-stream", "-M", action='store_true', 
                        help = 'Use this flag to disable the streaming StdOut callback for LLMS.')
    return parser.parse_args()

if __name__ == "__main__":
    main()