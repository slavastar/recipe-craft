import os
import sys
import glob
import json
import torch
import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.utils.utils_file import load_json, load_yaml


class RecipeRetriever:

    def __init__(self, config_path: str):
        logger.info(f"Loading configuration from: {config_path}")
        self.config = load_yaml(config_path)

        self.data_dir = self.config['data_path']
        self.recipe_storage_path = self.config['recipe_storage']
        self.faiss_storage_path = self.config['faiss_storage']
        self.index_type = self.config['index_type'].upper()
        self.model = SentenceTransformer(self.config['embedding_model'])

        self.index = None
        self.recipes = []
        self.embeddings = None

        logger.info(f"RecipeRetriever initialized with index type: {self.index_type}")

    def load_or_build_index(self):
        if os.path.exists(self.faiss_storage_path) and os.path.exists(self.recipe_storage_path):
            logger.info("Existing FAISS index and metadata found.")
            self._load_index()
        else:
            logger.warning("No FAISS index found. Building a new one.")
            self._build_index()
            self._save_index()

    def _load_index(self):
        try:
            self.index = faiss.read_index(self.faiss_storage_path)
            self.recipes = load_json(self.recipe_storage_path)
            logger.success("FAISS index and recipe data successfully loaded.")
        except Exception as e:
            logger.exception(f"Failed to load FAISS index or recipe metadata: {e}")
            raise

    def _build_index(self):
        raw_texts = []
        all_recipes = []

        try:
            logger.info("Reading recipe files.")
            for file in glob.glob(os.path.join(self.data_dir, "*.json")):
                data = load_json(file)
                for recipe in tqdm(data.values()):
                    if 'title' not in recipe or 'ingredients' not in recipe or 'instructions' not in recipe:
                        logger.warning(f"Skipping incomplete recipe in {file}: {recipe}")
                        continue
                    text = f"{recipe['title']} {' '.join(recipe['ingredients'])} {recipe['instructions']}"
                    raw_texts.append(text)
                    all_recipes.append(recipe)

            logger.info(f"Encoding {len(raw_texts)} recipes using {self.config['embedding_model']}.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            self.embeddings = self.model.encode(
                raw_texts,
                batch_size=64,
                show_progress_bar=True,
                device=device,
                tqdm_class=lambda *args, **kwargs: tqdm(*args, file=sys.stdout, **kwargs)
            )
            self.recipes = all_recipes
            vectors = np.array(self.embeddings).astype('float32')
            dim = vectors.shape[1]

            logger.info(f"Constructing FAISS index of type: {self.index_type}")
            if self.index_type == "HNSW":
                self.index = faiss.IndexHNSWFlat(dim, self.config['index']['hnsw_m'])
            elif self.index_type == "IVFPQ":
                quantizer = faiss.IndexHNSWFlat(dim, self.config['index']['hnsw_m'])
                self.index = faiss.IndexIVFPQ(
                    quantizer,
                    dim,
                    self.config['index']['nlist'],
                    self.config['index']['nsubquantizers'],
                    self.config['index']['nbits']
                )
                logger.info("Training IVFPQ index.")
                self.index.train(vectors)
            else:
                self.index = faiss.IndexFlatL2(dim)

            self.index.add(vectors)
            logger.success("FAISS index successfully built and trained.")
        except Exception as e:
            logger.exception(f"Error during index construction: {e}")
            raise

    def _save_index(self):
        try:
            os.makedirs(os.path.dirname(self.faiss_storage_path), exist_ok=True)
            faiss.write_index(self.index, self.faiss_storage_path)
            with open(self.recipe_storage_path, 'w', encoding='utf-8') as file:
                json.dump(self.recipes, file, ensure_ascii=False, indent=4)
            logger.success("FAISS index and recipe metadata saved to disk.")
        except Exception as e:
            logger.exception(f"Failed to save FAISS index or metadata: {e}")
            raise

    def retrieve(self, query, top_k=3):
        try:
            logger.info(f"Retrieving top {top_k} recipes for query: '{query}'")
            query_embedding = self.model.encode([query]).astype('float32')
            distances, indices = self.index.search(query_embedding, top_k)
            results = [self.recipes[i] for i in indices[0]]
            logger.info(f"Retrieved {len(results)} recipes.")
            return results
        except Exception as error:
            logger.exception(f"Error during retrieval: {error}")
            return []
