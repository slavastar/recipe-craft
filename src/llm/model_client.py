import requests
from loguru import logger

from src.utils.utils_file import load_yaml


class ModelClient:
    
    def __init__(self, config_path):
        self.config = load_yaml(config_path)
        self.url = f"{self.config['host']}/api/generate"
        self.model = self.config.get('model', 'phi3')
        self.context_tokens = self.config.get('context_tokens', 2048)
        logger.info(f"Initialized model {self.model}.")

    def generate(self, user_prompt, recipes):
        context = self._format_context(recipes)
        prompt = f"""
            {self.config['system_prompt']} \n
            Context: {context} \n
            User Request: {user_prompt}\n
            Answer:\n
        """
        logger.debug("Sending a prompt to the model via Ollama.")
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "options": {
                "num_ctx": self.context_tokens
            },
            "stream": False
        })

        response.raise_for_status()
        logger.debug("Received a response from the model.")
        return response.json()["response"]

    def _format_context(self, recipes):
        
        def get_recipe(recipe):
            return f"""
                Title: {recipe['title']}
                Ingredients: {', '.join(recipe['ingredients'])}
                Instructions: {recipe['instructions']}
            """
        
        return "\n".join([get_recipe(recipe) for recipe in recipes])
