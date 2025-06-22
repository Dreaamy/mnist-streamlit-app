import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Generator (mesma arquitetura usada no treino)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    model.eval()
    return model

def generate_images(model, digit, n=5):
    noise = torch.randn(n, 100)
    labels = torch.tensor([digit]*n)
    with torch.no_grad():
        images = model(noise, labels).cpu().numpy()
    images = (images + 1) / 2  # [-1, 1] → [0, 1]
    return images

st.title("MNIST Digit Generator (0–9)")

digit = st.selectbox("Select a digit:", list(range(10)))

if st.button("Generate 5 images"):
    model = load_generator()
    imgs = generate_images(model, digit)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        img = (imgs[i][0] * 255).astype(np.uint8)
        col.image(Image.fromarray(img), caption=f"Digit {digit}", use_column_width=True)
