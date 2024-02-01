# SigVer by Munkh-Orgil Batchuluun (erik)
This is a POC of signature verification model I created following these papers
- [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/abs/1707.02131)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

Dataset is from ICDAR 2013, which is collection of non-forged and forged signatures. Apparently, the signatures, as stated in the csv file, one non-forged signature picture is matched with multiple forged and non-forged pictures which makes the total dataset size of 23000+ even though the amount of images is 2200+.
- [ICDAR 2013](https://paperswithcode.com/dataset/icdar-2013)

Just download the required libraries as mentioned below:
```
pip install -r requirements.txt
```
**IMPORTANT** torch may be installed as cpu version. **YOU NEED CUDA VERSION**

## Model
Model is uploaded on Huggingface.co since I don't where else should I upload my model. If I uploaded it to any other file hosting services, i could lose my account on that file hosting service due to my carelessness or file hosting service straight up delete it.

- [SVGG16 model](https://huggingface.co/erik1126/Enet)

## Running the application

1. Upload the model to the folder
2. Install the libs mentioned above
3. Then `streamlit run main.py` or `python -m streamlit run main.py`. Both works.
