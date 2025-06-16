from fastapi import FastAPI
from loguru import logger
import uvicorn

from src.rag.retriever import RecipeRetriever
from src.llm.model_client import ModelClient


app = FastAPI()
logger.info("Initializing FastAPI application")


retriever = RecipeRetriever("config/config_rag.yaml")
logger.info("Loading or building FAISS index")
retriever.load_or_build_index()

llm = ModelClient("config/config_model_client.yaml")
logger.info("Model client client initialized.")


@app.get("/ask")
def ask(prompt: str):
    logger.info(f"Received user prompt: {prompt}")
    try:
        context = retriever.retrieve(prompt, top_k=3)
        if not context:
            logger.warning("No relevant recipes found for the query.")
            return {"answer": "Sorry, I could not find a relevant recipe."}

        answer = llm.generate(prompt, context)
        logger.info("Successfully generated LLM response.")
        return {"answer": answer}
    except Exception as error:
        logger.error(f"Error during processing: {error}")
        return {"error": str(error)}


if __name__ == "__main__":
    logger.info("Starting FastAPI server.")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
