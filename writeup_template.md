# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./test_images/1.png "Traffic Sign 1"
[image2]: ./test_images/2.png "Traffic Sign 2"
[image3]: ./test_images/3.png "Traffic Sign 3"
[image4]: ./test_images/4.png "Traffic Sign 4"
[image5]: ./test_images/5.png "Traffic Sign 5"
[image6]: ./test_images/6.png "Traffic Sign 6"
[image7]: ./test_images/7.png "Traffic Sign 7"
[image8]: ./test_images/8.png "Traffic Sign 8"

[orig_gen]: ./images/orig_gen.png "Original vs. Generated"
[y_chan]: ./images/y_channel.png "Original vs. Y channel"
[hist_train]: ./images/histogram_train.png "Histogram Training"
[hist_valid]: ./images/histogram_valid.png "Histogram Validation"
[hist_test]: ./images/histogram_test.png "Histogram Test"
[graph]: ./images/graph.png "Graph"
[accuracy_entropy]: ./images/accuracy_entropy.png "accuracy cross/entropy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the collections library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing a histogram of the training, validation and testing data sets.

![alt text][hist_train]
![alt text][hist_valid]
![alt text][hist_test]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV and to take the Y channel. This method was used by Sermanet and LeCun, 2011, and proved to yield good results.

Here is an example of a traffic sign image before and after taking the Y channel:

![alt text][y_chan]

Then I normalized the image and performed logarithmic corrections which helped to yield better training results. After this step I reapplied the normalization to ensure that the data that was acquired is normalized.

I decided to generate additional data because in two manners:

* Correct for the uneven distribution by generating images in every class to match the number of images in the highest represented class.
* Generate from each class two more samples to aid training.

For each time an image generation was required, I choose to do each of the following:

* randomly shift the image in the range of [-2,2] pixels
* Rotate the image in the ranenge of [-15,15] degrees
* scale the image in the range of [0.9,1.1]
* For 20% of the images add motion blur

Here is an example of an original image and an augmented image:

![alt text][orig_gen]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 (Y layer normalized) image   							| 
| Convolution (1) 5x5     	| 1x1 stride, same padding, outputs 28x28x12 	|
| RELU					| Activation layer
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 |
| Dropout	    (1)  	| keep_prob = 0.9 |
| Convolution (2) 5x5     	| 1x1 stride, same padding, outputs 10x10x24 |
| RELU					| Activation layer
| Max pooling	      	| 2x2 stride,  outputs 5x5x12 |
| Dropout	   (2)   	| keep_prob = 0.9 |
| Fully connected (1)	| Concatenation of convolution (1) and dropout (2)  (size is 14x14x12 + 5x5x24 = 2952 , outputs= 600  |
| Dropout	   (3)   	| keep_prob = 0.7 |
| Fully connected (2)	| input = 600, outpus = num_classes = 43 |
| Softmax				| |      
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

* The optimizer used is AdamOptimizer and minimizing the cross-entropy
* Learning rate = 0.001
* Number of epochs = 10
* Batch size = 128

I also experienced with different initializations of the weights, some layers were initialized with a truncated normal distribution weight of mu = 0, sigma = 0.1 and some with sigma = 0.2.
Also, the weights where initilized to a constant of 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

* I started with a simple LeNet network and tried to achieve a validation accuary of 0.93
* In order to get this accuracy I had to generate 5 times the number of training samples that were provided.
* Afterwards I added dropout layers and to the network and tried to change the number of filters in the convultions to get a better accuarcy.
* The accuarcy achieved on the validation set was around 0.93
* I tried to tune the hyperparameters, namely the dropout rates and also play around with generating samples with noisy samples.
* Since this is a CNN network, a convoultion is our first choice. And drop-out layers help fight the overfitting problem.

Afterwards I decided to take ideas from the work of Sermanet and LeCun, 2011. Their network took ideas from the LeNet model and extended it with techniques that better worked with the German Sign dataset.

My final model results were:

* training set accuracy of 0.995
* validation set accuracy of 0.933 
* test set accuracy of 0.956

Since the the accuracy of the training is ~100% it means that the training was successful. Also, the validation accuracy is higher than 93%, that means  that the model did learn. The test accuracy presents also a result which is close to the validation accuracy that means that the model didn't overfit/underfit.
 
Moreover to confirm that the model did learn, I used TensorBoard and analyzed the different layers of the model. TensorBoard enables looking at the netowkr graph, thus helping to see if this matches the graph that was designed in the first place. Also it gives the oppurtunity to explore the evolution of weights and biases for every layer. To work with TensorBoard, I had to tag the different layers of the network with proper labels that can be seen in the code.

The following is the network graph:

![alt text][graph]

An additional proof to the learning of the network is the development of the accuracy and the cross entropy throughout the training process:

![alt text][accuracy_entropy]

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are four German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4]

And the following are ones that I photographed myself:

![alt text][image5] ![alt text][image6] ![alt text][image7]  ![alt text][image8]
 
The first image should be relatively easy to classify since it is in the center and similar tthe ones in the database.
The second image might be difficult to recognize, since it is skewed.
For the 3rd and 4th image, they should be recognizable easily.
The 5th image is a bit skewed and has a hint of a different sign.
The 6th image is cluttered and is a bit tricky to recognized.
The 7th image is skewed and has bad lightning conditions but is still well visible.
The 8th image is not part of the 43 classes so it is impossible to recognize.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of no passing      		| End of no passing   									| 
| Road work     			| Road work 										|
| No passing for vehicles over 3.5 metric tons					| No passing for vehicles over 3.5 metric tons|
| Stop Sign      		| Stop sign   | 
| go straight or right	      		| go straight or right					 				|
| Turn right ahead			| Turn right ahead      							|
| General caution | General caution |
| Speed limit (10km/h) | |


The model was able to correctly guess 7 of the 8 traffic signs, but actually the 8th sign is not part of the database, so it could detect all 7 signs which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


