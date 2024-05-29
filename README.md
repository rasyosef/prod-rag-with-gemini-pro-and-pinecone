# Retrieval Augmented Generation with Gemini Pro, Pinecone and LlamaIndex: Question Answering demo

### This demo uses the Gemini Pro LLM and Pinecone Vector Search for fast and performant Retrieval Augmented Generation (RAG).

The context is the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the Gemini Pro model is not aware of it.

Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt. The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.

## RAG Components
- **LLM** : `Gemini Pro`
- **Text Embedding Model** : `Gemini Embeddings (embedding-001)`
- **Vector Database** : `Pinecone`
- **Framework** : `LlamaIndex`

## Demo
The demo (built with `gradio`) has been depolyed to the following HuggingFace space.

https://huggingface.co/spaces/rasyosef/RAG-with-Gemini-Pinecone-LlamaIndex

## Notebooks