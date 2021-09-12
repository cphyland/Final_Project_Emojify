# What Does Your Face Say About You?
Facial recognition is an increasingly important field in today's age. Use cases range from law enforcement and digital securtiy, to advertising and location trackings. This project demonstrates that it is possible to take off the shelf example code, and implement an effective application for face detection and emotion recognition. We utilized Python, Javascript, Tensorflow, openCV, webRTC, aiohttp, aiortc, and other libraries to accomplish our goal.

The code relies on two machine different learning algorithms. The Haar cascade is used to isolate the user's face, which is facilitated by opencv. The emotion prediction is made using an implementation of the VGGNet convolutional neural network (CNN), which is facilitated by tensorflow. Python aiohttp is used to run the backend server, supported by aiortc and asyncio to handle the webRTC stream, and the final app is hosted on heroku. 

[<img src="">]()

## Communication Process

The first thing that happens when app launches, is the client is told to request their public IP from a STUN server. This is done to  enable communication through NATs, giving the web server a valid return address for the media stream.  Once that is accomplished, the front end Javascript initiates a stream using webRTC, and sends this to the web server, along with the public IP address as the return point. The back end server processes the stream, makes a prediction, and then returns that prediction to the client on top of the original video stream.

[<img src="">]()

## Back-End Processing
### Face Detection Using Haar Cascade
The first step in processing the image stream is to convert the frame to grayscale, and use the [haar cascade]() to detect faces. This is readily accomplished using built in methods from the openCV library.

[<img src="">]()

 Developed by Viola and Jones in 2001, haar cascades work by applying a sliding window of filters over the image. These filters are used to indentify lines, shapes, and features within the image. All in all 6000 of these filters are used by the algorithm.

[<img src="">]()

### Emotion Prediction Using CNN
After face detection has occured, the portion of the image containing the face isolated, and coverted it to the format that the CNN was trained on. Reducing the grayscale image to  a 48 x 48 pixel. The face is then run through the CNN, using Tensorflow Lite, and predicts a facial expression.

[<img src="">]()

### Returning the Prediction

[<img src="">]()

## Prediction Accuracy
The CNN used was taken from [Facial Emotion Recognition: State of the Art Performance on FER2013](). The authors use extensive hyperparamter tuning and learning rate scheduling to achieve a test accuracy of 73.28% on the [FER2013 dataset]().

The green bars show the final validation accuracies achieved during training, and the blue ones show the corresponding testing accuracies for those models. As you can see, the Reduced Learning Rate on Plateau (RLRP) performs the best with a validation accuracy of 73.59%.  The RLRP performs better since it monitors the current performance before deciding when to drop the learning rate as opposed to systematically reducing the learning rate. 

[<img src="">]()

The model shows the best classification on the “happiness” and “surprise” emotions.  However, it performs poorly when classifying between “disgust” and “anger”.  Performing only slightly better differentiating between “disgust” and “fear”.  Which can be attributed to the fact that they have a lower number of samples in the original training set. 

[<img src="">]()


## Sources
### Articles
- Behera, Girjia Shankar. "Face Detection with Haar Cascade." Medium, Towards Data Science, 29 Dec 2020, towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08.

- "Elements Marketplace: Heroku-Buildpack-Apt." Heroku/Heroku-Buildpack-Apt - Buildpacks - Heroku Elements, elements.heroku.com/buidpacks/heroku/heroku-buildpack-apt.

- Khaireddin, Yousif, and Shoufa Chen. "Facial Emotion Recognition: State of the Art Performance on FER2013." ArXiv.org, 8 May 2021, arxiv.org/abs/2105.03588v1.

- Mellouk, Wafa, and Wahida Handouzi. "Facial Emotion Recognition Using Deep Learning: Review and Insights." Procedia Computer Science, Elsevier, 6 Aug. 2020, www.sciencedirect.com/science/article/pii/S1877050920318019

- "Rapid Object Detection Using a Boosted cascade of Simple Features." IEEE Xplore, ieeexplore.org/document/990517

- "TensorFlow Lite." tensorFlow, www.tensorflow.org/lite/guide

### Code
- Aiortc. "Aiortc/Aiortc: WebRTC and Ortc Implementation for Python Using Asyncio." Github, github.com/aiortc/aiortc.

- "Emojify - Create Your Own Emoji with Deep Learning." Dataflair, 14 Mar. 2021, data-flair.training/blogs/create-emoji-with-deep-learning/.

- Nagm Hemanth. "Camera App with Flask and OpenCV." Medium, Towards Data Science, 30 May 2021, towardsdatascience.com/camera-app-with-flask-and-opencv-bd147f6c0eec

### Models
- "Opencv." Github, github.com/opencv

- Usef-Kh. "Usef-kh/Fer: Code for the Paper "Facial Emotion Recognition: State of the Art Performance on FER2013." Github, github.com/usef-kh/fers

