# SigVer by Munkh-Orgil Batchuluun (erik)
This is a POC of signature verification model I created following these papers
- [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/abs/1707.02131)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

Dataset is from ICDAR 2013, which is collection of non-forged and forged signatures. Apparently, the signatures, as stated in the csv file, one non-forged signature picture is matched with multiple forged and non-forged pictures which makes the total dataset size of 23000+ even though the amount of images is 2200+.
-[ICDAR 2013](https://paperswithcode.com/dataset/icdar-2013)

Just download the required libraries as mentioned below:
```
pip install -r requirements.txt
```
** IMPORTANT ** torch may be installed as cpu version. ** YOU NEED CUDA VERSION **
