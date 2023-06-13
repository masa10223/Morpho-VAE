## A deep learning approach for morphological feature extraction based on variational auto-encoder : an application to mandible shape
This repository contains the codes used for the following paper:

>**A deep learning approach for morphological feature extraction based on variational auto-encoder : an application to mandible shape**
>Masato Tsutsumi, Nen Saito, Daisuke Koyabu and Chikara Furusawa
>accepted in *npj systems biology and applications*
>**Abstract**: Shape measurements are crucial for evolutionary and developmental biology; however, they present difficulties in the objective and automatic quantification of arbitrary shapes. Conventional approaches are based on anatomically prominent landmarks, which require manual annotations by experts. Here, we develop a machine-learning approach by presenting morphological regulated variational AutoEncoder (Morpho-VAE), an image-based deep learning framework, to conduct landmark-free shape analysis. The proposed architecture combines the unsupervised and supervised learning models to reduce dimensionality by focusing on morphological features that distinguish data with different labels. We applied the method to primate mandible image data. The extracted morphological features reflected the characteristics of the families to which the organisms belonged, despite the absence of correlation between the extracted morphological features and phylogenetic distance.Furthermore, we demonstrated the reconstruction of missing segments from incomplete images. The proposed method provides a flexible and promising tool for analyzing a wide variety of image data of biological shapes even those with missing segments.

## Dependencies
This code only works on GPU environments.
```
pip install -r requirements.txt
```
This might take a few minutes.

## Usage
Show_results_notebook.ipynb is a jupyter notebook that can run the results of the paper on a calculator.
You can view the results by pressing the cell written on it, but keep in mind that you will need a GPU environment. 