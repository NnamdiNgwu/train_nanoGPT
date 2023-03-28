### 1. Intro
Generative pretrained Transformer, is a decoder only Transformer model. Our trained nanoGPT is a transformer model that is based on attention mechanism.

### 2. Data Preparation
* To prepare a dataset to build a GPT model, the following steps can be followed:
* Data collection: You need to collect a large amount of text data, such as books, articles, and websites, to use it as the training data for your GPT model.
* Data cleaning: You should remove any irrelevant information, such as HTML tags or irrelevant headers, and standardize the text format.
* Tokenize the data: Divide the text into smaller units subwords, to enable the model to learn the structure and language patterns of shakespare
* Data pre-processing: Perform any necessary pre-processing tasks on the data, such as stemming, removing stop words, or converting the text to lowercase.
* Split the data: We divide the cleaned and pre-processed data into different sets, training, validation, and test sets to evaluate the model’s performance during training.
* Batch creation: Create batches of the training data to feed into the model during training. Our model required that we create batches sequentially.
* Convert the data to tensor:  Data is converted into tensor since TensorFlow and PyTorch are some basic data structures used in deep learning frameworks. 

It is essential to ensure that the data is of high quality, diverse, and in sufficient quantity to train the GPT model effectively and avoid overfitting
