import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from model import VGG
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

my_file = Path("SVGG.pt")
if my_file.is_file():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VGG()
        model.load_state_dict(torch.load('SVGG.pt'))
        model.eval()
        model.cuda()
        
        col1, col2 = st.columns(2)
        input1 = None
        input2 = None
        transform = transforms.Compose(
                [transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
            )
        with col1:
            image1_file = st.file_uploader("Choose image 1", type=["jpg", "jpeg", "png"])
            if image1_file is not None:
                image1 = Image.open(image1_file).convert("RGB")
        
                st.image(np.squeeze(image1), caption="Uploaded Image", use_column_width=True)
                st.write("Image dimensions:", image1.size)
                input1 = image1
        with col2:
            image2_file = st.file_uploader("Choose image 2", type=["jpg", "jpeg", "png"])
            if image2_file is not None:
                image2= Image.open(image2_file).convert("RGB")
        
                st.image(np.squeeze(image2), caption="Uploaded Image", use_column_width=True)
                st.write("Image dimensions:", image2.size)
                input2 = image2
        
        if st.button('Predict'):
                        
            input1= transform(input1)
            input2= transform(input2)
            st.write("Shape1: {}, Shape2: {}".format(input1.shape, input2.shape))
            output1, output2 = model(input1.unsqueeze(0).cuda(), input2.unsqueeze(0).cuda())
            euclidean_distance = F.pairwise_distance(output1, output2)
            st.write("Euclidean Distance: {}".format(euclidean_distance.item()))
            st.write("If ED is less than 0.0096, It's a non-forged signature")
else:
        st.write("Please upload the SVGG model to continue. Refer to README.md")
                



def imshow(img, text=None, should_save = False, loss=None, actual=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text + "\nActual: {} \nPrediction: {}".format(actual, "Original" if loss < 0.0096 else "Forged"),
            style="italic",
            fontweight="bold",
            bbox = {"facecolor":"white", "alpha": 0.8, "pad":10}
        )
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
