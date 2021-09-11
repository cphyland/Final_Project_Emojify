# Final_Project_Emojify
cph presentation notes:

Slide 9:
First part of the code: takes the image of the face and coverts it to the format that the model trained on. Converting the image from color to grey scale.

Second part of code:
Runs the face through the deep neural network layers and produces an outcome.

Training vs Accuracy:
The green bars show the final validation accuracies and the blue ones show the corresponding testing accuracies.  As you can see the Reduced Learning Rate on Plateau performs the best with a validation accuracy of 73.59%.  The RLRP performs better since it monitors the current performance before deciding when to drop the learning rate as opposed to systematically reducing the learning rate. 

Confusion Matrix:
The model shows the best classification on the “happiness” and “surprise” emotions.  However, it performs poorly when classifying between “disgust” and “anger”.  Performing only slightly better differentiating between “disgust” and “fear”.  Which can be attributed to the fact that they have a lower number of samples in the original training set.  
