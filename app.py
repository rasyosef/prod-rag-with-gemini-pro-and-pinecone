import os
import gradio as gr
import google
from pinecone import Pinecone
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import VectorStoreIndex
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from llama_index.llms.gemini import Gemini

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Pinecone Vector Store
INDEX_NAME = "rag"
pinecone = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)


# Check if the provided Gemini api key has valid length
def is_valid_gemini_api_key(api_key):
    if len(api_key.strip()) == 39:
        return True
    return False


# Create query engine using the user-provided api key
def prepare_query_engine(api_key):
    # Gemini Embeddings
    embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        title="wikipedia page of the 'Oppenheimer' movie",
        embed_batch_size=16,
        api_key=api_key,
    )

    # Load Index
    index_loaded = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    # Gemini Safety Settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Gemini Pro
    llm = Gemini(
        model_name="models/gemini-pro",
        temperature=0,
        max_tokens=512,
        safety_settings=safety_settings,
        api_key=api_key,
    )

    # Query Engine
    query_engine = index_loaded.as_query_engine(
        llm=llm, streaming=True, similarity_top_k=3, response_mode="tree_summarize"
    )

    return query_engine


# Generates response using the query engine
def generate(query, api_key):
    if api_key.strip() == "" or not is_valid_gemini_api_key(api_key):
        yield "Please enter a valid Gemini api key"
    else:
        query_engine = prepare_query_engine(api_key)
        response = ""
        try:
            streaming_response = query_engine.query(query)
            for chunk in streaming_response.response_gen:
                response += chunk
                yield response
        except google.api_core.exceptions.BadRequest as br:
            yield "API key not valid. Please enter a valid API key"
        except Exception as e:
            yield str(e)


with gr.Blocks() as demo:
    gr.Markdown(
        """
  # Retrieval Augmented Generation with Gemini Pro, Pinecone and LlamaIndex: Question Answering demo
  ### This demo uses the Gemini Pro LLM and Pinecone Vector Search for fast and performant Retrieval Augmented Generation (RAG).
  ### The context is the new Oppenheimer movie's entire wikipedia page. The movie came out very recently in July, 2023, so the Gemini Pro model is not aware of it.
  Retrieval Augmented Generation (RAG) enables us to retrieve just the few small chunks of the document that are relevant to the our query and inject it into our prompt.
  The model is then able to answer questions by incorporating knowledge from the newly provided document. RAG can be used with thousands of documents, but this demo is limited to just one txt file.
  """
    )
    GEMINI_API_KEY = gr.Textbox(
        label="GEMINI_API_KEY",
        placeholder="Enter your GEMINI API KEY",
        lines=1,
        type="password",
    )
    gr.Markdown("## Enter your question")
    with gr.Row():
        with gr.Column():
            ques = gr.Textbox(label="Question", placeholder="Enter text here", lines=2)
        with gr.Column():
            ans = gr.Textbox(label="Answer", lines=4, interactive=False)
    with gr.Row():
        with gr.Column():
            btn = gr.Button("Submit")
        with gr.Column():
            clear = gr.ClearButton([ques, ans])

    btn.click(fn=generate, inputs=[ques, GEMINI_API_KEY], outputs=[ans])
    examples = gr.Examples(
        examples=[
            "Who portrayed J. Robert Oppenheimer in the new Oppenheimer movie?",
            "In the plot of the movie, why did Lewis Strauss resent Robert Oppenheimer?",
            "What happened while Oppenheimer was a student at the University of Cambridge?",
            "How much money did the Oppenheimer movie make at the US and global box office?",
            "What score did the Oppenheimer movie get on Rotten Tomatoes and Metacritic?",
        ],
        inputs=[ques],
    )

demo.queue().launch(debug=True)
