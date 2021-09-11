# Final_Project_Emojify
Hello everyone!

For our project we started off by googling machine learning and came across the idea of emotional facial recognition. We were able to find an existing repo that contained the starter code and data, we then added the component for the camera to be able to detect your emotion and create the bridge between the trained model and the camera. We utilized openCV and webRTC, and of course we used machine learning to train the model. We believe this is important because facial recognition can contribute data to several fields in today's age. The code relies on two machine learning algorithms, the HAAR cascade is used to isolate the users face, which is facilitated by opencv. The emotion prediction is made using a deep neural network, which is facilitated by tensorflow. Python flask ran the server, and heroku hosted our app. 

cph presentation notes:

Slide 9:
First part of the code: takes the image of the face and coverts it to the format that the model trained on. Converting the image from color to grey scale.

Second part of code:
Runs the face through the deep neural network layers and produces an outcome.

Training vs Accuracy:
The green bars show the final validation accuracies and the blue ones show the corresponding testing accuracies.  As you can see the Reduced Learning Rate on Plateau performs the best with a validation accuracy of 73.59%.  The RLRP performs better since it monitors the current performance before deciding when to drop the learning rate as opposed to systematically reducing the learning rate. 

Confusion Matrix:
The model shows the best classification on the “happiness” and “surprise” emotions.  However, it performs poorly when classifying between “disgust” and “anger”.  Performing only slightly better differentiating between “disgust” and “fear”.  Which can be attributed to the fact that they have a lower number of samples in the original training set.  
