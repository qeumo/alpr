# alpr
## About
This repository presents the result of the work of 12 teams on a task within the framework of the subject of deep machine learning in practice

## Team
* Dima - number on plate to text
* Rita - plate detection on image
* Pasha - car classification on image

## Pipeline
We consider images from the input stream with the most convenient frame rate for analysis. We process the received frames as follows: Initially, a car and its class and its license plate are detected on the frame. This license plate, by means of affine transformations, is brought to a working form. After that, the working section is binarized and fed to the "image-to-text" input of a convolutional recurrent neural network, which extracts text from the image and outputs it as a response.

## Change of ideas during working
Initially, after extracting the license plate and its binarization, it was planned to cluster the license plates on the board and their subsequent character-by-character classification. These plans had to be abandoned due to the need to process other objects that were clustered as a result of the work, and the need to add additional filters to extract only numbers from images

## Stack
* YOLО - for car and plate detection
* Convolutional-recurrent network for plate reading
