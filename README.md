# DS-Project-7-Generating_Monet-esque_Images_CycleGAN
Designed a model that implements the CycleGAN architecture to convert pictures to a Monet style. Basically, the model adds the Monet style to any picture. 

* Built an image generator model that produces Monet'esque images. The model uses the CycleGAN algorithm to generate any image with a touch of Monet's skills.
* Dataset - Part of a Kaggle Competition. The dataset contains 300 Monet paintings and 7028 test photos on which we will apply the Monet effect.
* Model - The major aim in this project is to convert the test photos to have the Monet effect.  
## Code and Resources Used ##
**Python Version:** 3.10.5 <br />
**Packages:**  numpy, pandas, matplotlib, PIL, tensorflow, keras, re, os, kaggle_datasets <br />
**For Web Framework Requirements:** _pip install -r requirements.txt_ <br />
**Data Resources:** <https://www.kaggle.com/competitions/gan-getting-started> <br />

## About the Dataset ##
Computer vision has advanced tremendously in recent years and GANs(Generative Adversarial Networks)  are now capable of mimicking objects in a very convincing way.<br>
The **challenge** is to create museum worthy art from deep learning algorithms. <br>

The dataset contains 300 Monet paintings and 7028 test photos in the jpeg and TFRecord format.<br>
TFRecord is a simple format for storing a sequence of binary records.This is more efficient and saves a lot of space.
![](Monet_Paintings.png "Monet Paintings")
![](Test_Photos.png "Test Photos")


## Prepocessing the images for the model ## 
1. These images are not in same shape and since our pre=trained model requires a specific shape. We manipulate and resize all images in the required size which is (224,224,3).
2. Separating the cats and dogs in different files so that we can pick equal amounts of pictures to train our model.
3. Since we are using a pre-trained model with pre-trained weights, we dont have to use all the images in our dataset to train our model.I just use 2000 pictures of both cats and dogs for my model.
4. Creating labels for resized images of dogs and cats where cats are represented by 1 and dogs are represented by 0.
5. Converting all resized images to NumPy arrays as our model only takes numpy array as input.
6. Splitting the dataset into training and testing sets using the sklearn library.
7. Feature Scaling- Dividing all values by 255 because the that is my maximum intensity of colors

## Model Building ##
1. Uploading the mobilenet model from the url given above from the tensorflow-hub library 
2. Defining a simple keras Sequential model with the mobilenet model weights and an output layer.
3. Compiling the model accordingly:
    - optimizer : 'adam' - optimization technique for gradient descent
    - loss : 'SparseCategoricalCrossentropy' - as we are dealing with binary classification
    - metrics : 'accuracy' - simple as its just a binary classification

## Model Performance ##
As its a pre=trained model, we dont require a lot of epochs to train our model. After just 5 epochs we get an accuracy score of *0.99* and a loss score of *0.036* on the training data.
After Evaluating it on the test data, we receive an accuracy score of *0.97* and a loss score of *0.066*. That means we predict 97 pictures out of 100 accurately.
