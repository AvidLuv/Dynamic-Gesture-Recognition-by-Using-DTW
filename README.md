# Dynamic-Gesture-Recognition-by-Using-DTW

Overview

This project is developing a dynamic gesture recognition system for embedded sytem. In this project, using the Mediapipe hand keypoints model to extract the keypoints from the live stream video. By calculating and stacking the angle features between the keypoints vectors to form the feature time sires of the gesture. Then using the Fast Dynamic Time Warping (FastDTW) to calculate the similaritis of the time series of the features between the live stream gesture and the dataset. 

Installation

The python version used in this project is 3.8. Before running the mycode.py file, please make sure to install the packages of Mediapipe, OpenCV and FastDTW:
pip install mediapipe 0.10.9
pip install opencv    4.6.0
pip install fastdtw   0.3.4

The action dataset can be accessed by downlong the action_train, then placing the dataset folder in the same address of the mycode.py file 

How it works

In order to extract the features of the hand, the MediaPipe’s hand keypoints model processes  hand images and converting them into keypoints data to support the feature extraction algorithm. The vectors will be contructed between special keypoints. Then the angles between vectors will be calculated using the following the equation:

cosθ = （X1*X2 + Y1*Y2)/√(X1*X1 + Y1*Y1) + √(X2*X2 + Y2*Y2)

By staking the angle features of continues frames, a feature of time series of dynamic gestures is formed. Use the FastDTW algorithm to calculate the feature time series similarity between the dataset and real-time dynamic gestures, and obtain recognition results. The FastDTW algorithm returns an increasing distance sequence that describes the similarity order between real-time gesture features and features in the dataset. In the recognition algorithm, the algorithm establishes two key decision thresholds: frequency occurrence threshold 𝑓𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 and decision range threshold 𝑅𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑. Within the range of the decision range threshold, the frequency of the gesture possessing the minimal distance is ascertained utilizing the following equation:

𝑓𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 = T/𝑅𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑
𝑤ℎ𝑒𝑟𝑒:
𝑇 = 𝑠𝑚𝑎𝑙𝑙𝑒𝑠𝑡 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 𝑔𝑒𝑠𝑡𝑢𝑟𝑒 𝑜𝑐𝑐𝑢𝑟 𝑡𝑖𝑚𝑒𝑠 𝑤𝑖𝑡ℎ𝑖𝑛 𝑡ℎ𝑒 𝑑𝑒𝑐𝑖𝑠𝑖𝑜𝑛
𝑅𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 = 𝑑𝑒𝑐𝑖𝑠𝑖𝑜𝑛 𝑟𝑎𝑛𝑔𝑒 𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑

In this project, 𝑓𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 is set to 0.5, and 𝑅𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑 is set to 6. This configuration means that if the gesture with the highest similarity among the first six recognized potential gestures (recognized as the first gesture) is observed more than three times (𝑓𝑜𝑐𝑐𝑢𝑟𝑒𝑛𝑐𝑦>𝑓𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑), then it is classified as the first gesture. On the contrary, frequencies below this threshold will be judged as unknown gestures. (𝑓𝑜𝑐𝑐𝑢𝑟𝑒𝑛𝑐𝑦<𝑓𝑡ℎ𝑟𝑒𝑠ℎ𝑜𝑙𝑑)

Running an example

After install the packages and place the action_train in the same address of the mycode.py file. Excecute the following code to run the program:

For Lunux system:
python3 mycode.py

For Windows system:
Using the Visual Studio Code Compilor to run the code

Before running the code, please replace the file address of the following code with yourself action_train address:

saveFileName = 'C:/Users/Hank Yue/Desktop/data/trainingData.npz'
filename = 'C:/Users/Hank Yue/Desktop/data/action_train/*/*.mp4'

Then you can adjust the imput image resolution by change the followtion code, the default input image resolution is 320x240:

self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
Once the code is running, the program will first extract the features in the action_train folder then generate a TrainingData.npz file which contain the features of the action_train. Then the live stream video will display in your screen.  In the uppper left corner, the dot will be red, which means that the recognition state is off. In order to start the recognition state, you can press 'r' key on your keyboard. Then the dot will be yellow, which means it is the ready state now. This state is used to let user to get ready to perform the gesture, placing there hand in the live video. The default ready time is 2 seconds, you can adjust the ready time by change the code:

if time.time() - triger_time > 2:

In defaut condition, the recognition state will start after 2 seconds and the dot will be green. In this state, user can perform their gestures. When the program collect the features of 35 frames, the system will show the recognized gesture. You can adjust the following code to fit your gesture perform time if 35 frames is too short:

if frame_count < 35:

You can close the program by pressing the 'q' key on your keyboard.
