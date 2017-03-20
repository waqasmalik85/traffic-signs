#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/hist.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/image_1.png "Traffic Sign 1"
[image5]: ./examples/image_2.png "Traffic Sign 2"
[image6]: ./examples/image_3.png "Traffic Sign 3"
[image7]: ./examples/image_4.png "Traffic Sign 4"
[image8]: ./examples/image_5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.



###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the cell 18 of the IPython notebook.  

I used the Numpy, pandas and seaborn library to calculate and display summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the cell 19,20 and 21 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Cell 22 contains the conversion of data from RGB to YUV and cell 23 displays the images. Then histogram equilization is performed in cell 24 only on Luminance part. This technique published by Yann LeCun in "Traffic Sign Recognition with Multi-Scale Convolutional Networks". In my case I only kept the Lumince part as most of the information is in shape and not in the color in case of traffic signs. Images are displayed in cell 25.

Finally the images are normalized between -1 and 1. It maes the problem well conditioned and easier for optimzer to converge.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In cell 29 Ratio of each class is calculated to put avoid the network to get biased towards classes with high number of training examples. Idea is to penalize with higher loss in case the network predicts the wrong class for minority classes.

In cell 32, 20 Epochs are set along with batch size of 128. Training was performed on PC so the number higher then 128 had taken too long, moreover there weren't significant improvements after 20 Epochs.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.




####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the  cell 35 of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| flatten	| input 5x5x32 , output 800       									|
| Fully connected		| input 800, output 128       									|
| RELU					|												|
| dropout				|					probability 0.5							|
| Fully connected		| input 128, output 64       									|
| RELU					|												|
| dropout				|					probability 0.5							|
| Fully connected		| input 64, output 43      									|
| Softmax				|        			probabilities						|
|	Onehot	|										output		|




####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Cell 39 contains the training process. Adam optimizer is used with the learning rate of 0.001. Batch size of 128 on the PC and nework was trained for 20 Epochs.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The model is based on Lenet architecture however Lenet is only deep enough to recognize digits where as traffic signs contains numbers and digits. Model's depth was increased to learn the new shape features and it performed well on both training and validation data.

Since no augmented data was generated so it is important to make the network not too deep. It would cause overfitting and bad performance on validation set.

My final model results were:
* training set accuracy of 99.7
* validation set accuracy of 97.2
* test set accuracy of 94.8

If an iterative approach was chosen:
* First model was also similar to Lenet but inputs were 3 channel images. So the netork was approximately 3 times deeper than the current network.
* However the maximum validation accuracy was 89% as the training data was too small and model would overfit.
* Very high dropouts and L2 regularizations were tried. However in that case the network would underfit and both training and validation accuracy got worse.

If a well known architecture was chosen:
* Lenet architecture as it is  well known for classifying numbers which is very useful in classifying speed limit shields.



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

Second image "Roundabout mandatory" is one of the rare examples so it might be difficult for the network to recognize it. Third image "Speed limit 30km/h" is very similar to 80km/h speed limit so it would be a good test to see the network performance. The images are well lit and in focus.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 90th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry      		| No entry  									|
| Roundabout mandatory     			|Priority road									|
| Speed limit (30 km/h)				| Speed limit (30 km/h)									|
| Priority road      		| Priority road				 				|
| Yield		| Yield     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The model guessed the second image and is very certain about it.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 90th cell of the Ipython notebook. The network is very certain about image 2 which is false classified. Image 3 which is similar to 80km/h shows slightly less certainity. Bar graph of probabilities is shown in cell 90.

 The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| No entry  									|
| .98    				| Priority road									|
| .81				| Speed limit (30 km/h)											|
| 1.0	      			| Priority road					 				|
| 1.0				    | Yield     							|
