# OpenImaj-Swing-WebView
### FaceRecognition approximation

Information that have been used to perform most of this is here: http://openimaj.org/tutorial-pdf.pdf and http://openimaj.org/apidocs

This Java program is my diploma project and implements some computer vision algorithms using OpenImaj library.
First of all, i would like to say that it is my first big project in Java and it contains some design issues, because i don't have enough practical experience to avoid them all.
My program makes images from JavaFx pane with WebEngine inside, it means we can put in any facial set via browser (instead of WebCam realizations) to train realized algorithms.
I have realized HaarCascadeDetector to detect faces on the frame and several recognizers (EigenFaces and AnnotatorFaceRecogniser).
