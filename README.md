# Scraping and Classifying Images

This task is composed of two parts: scraping images from Google, specified by a keyword, and classifying these images. 

[BeautifulSoup](https://pypi.org/project/beautifulsoup4/) was used to scrape imagery from Google Images.

A [keras implementation](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/davit) of [DaViT](https://arxiv.org/abs/2204.03645) is used for classification. DaViT was chosen because it is effective yet efficient and ViTs have [recently become popular](https://viso.ai/deep-learning/vision-transformer-vit/) for classification tasks. One advantage over arcitectures like CNNs is that ViTs can handle input images of variable sizes making ViTs more adaptable to different image resolutions without the need for resizing or cropping. This is especially useful when dealing with images from various sources, such as this project which involves scraping the web. 

Because ViTs operate on the "patch-level", they can require fewer parameters than other architectures to achieve similar or better performance. The DaViT-T model used here only has 28.3M parameters, comparable to ResNet-50 with 25M, but achieves much better performance on the [ImageNet Benchmark](https://paperswithcode.com/sota/image-classification-on-imagenet). 


### Running the Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kittyschulz/scraped_image_classification/blob/main/scrape_and_classify.ipynb)

An Jupyter Notebook for this project is hosted on Google Colab, where you can play around with both scraping imagery from Google Images and running inference on these images using the tiny DaVit model.

Alternatively, the code can be run locally from the command line. To scrape imagery, call `scrape.py` and enter your key words, comma separated (for example, `horse,dog,stop sign,car`) and the number of images that should be downloaded for each class.

To run inference on images, call `classify.py`. When prompted to enter labels to run inference on, you can enter a subset of the image classes you downloaded comma separated (for example, `stop sign,car`). Alternatively, you can provide no input and `classify.py` will run inference on all images within the current directory. `classify.py` will save an *.xlsx file once it is finished.
