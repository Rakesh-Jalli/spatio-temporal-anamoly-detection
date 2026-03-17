"""
Semantic Understanding using CLIP embeddings.
"""

import clip
import torch
from PIL import Image
import numpy as np

class CLIPEmbeddings:
    def __init__(self, model_name='ViT-B/32'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_embedding(self, image):
        """
        Get CLIP embedding for an image or cropped object.
        Args:
            image: PIL Image or numpy array
        Returns:
            embedding: numpy array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        return embedding.cpu().numpy().flatten()