# Plant Disease Detection

NAME: VED VISHAL MIRJANKAR <br/>
PROJECT: BUILDING A PLANT DISEASE DETECTION USING IMAGE PROCESSING AND GAN  <br/>

ABSTRACT: <br/>
Capstone Project: Developed an automated plant disease detection system using image processing and Generative Adversarial Networks (GANs). Researched and implemented image processing algorithms, designed and trained GAN models for accurate disease classification, and integrated system components for optimal performance. </br>

HIGH LEVEL SYSTEM DESIGN: <br/>
![image](https://github.com/user-attachments/assets/90edb5c3-dca2-4a0c-8097-859ee482d954)

PROPOSED METHODOLOGY: <br/>
GAN Model:
GAN or Generative Adversarial Networks is used to increase the dataset size and thereby increase the accuracy of the model. In this project, a hybrid of both DCGAN and Conditional GAN is used to generate realistic images that closely resemble the original dataset. 
Segmentation:
Since the input image will have a lot of background noise, it makes it harder for the model to accurately classify the image as the model is trained on lab images. Hence, for the model to work on images of all sorts, a segmentation UNET model is used to accurately highlight only the diseased leaf part making it easier for the CNN model to classify accurately.
CNN Model:
After the UNET model accurately segments the inputted image, the CNN model classifies the image as healthy or unhealthy, and if unhealthy it also mentions the diseased class name.
![image](https://github.com/user-attachments/assets/a05a52be-2a51-4491-ae90-0fd49451b598)

RESULTS: <br/>
![image](https://github.com/user-attachments/assets/550785bd-e6b9-4ac4-89bd-7a4319b11e57)



