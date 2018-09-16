# Social_lstm_pedestrian_prediction
The aim of the project is to predict the trajectories of pedestrians using lstm neural networks. The project starts from the paper "Social LSTM: Human Trajectory Prediction in Crowded Spaces - Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, Silvio Savarese - Stanford University - The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 961-971", and its official implementation (https://github.com/vvanirudh/social-lstm-tf) and makes some modifications. 

This is the Multimedia communication course project made during my final year of the bachelor degree under the supervision of professor Nicola Conci and his phd student Niccolò Bisagno.  

The modifications introduced are two: 
- To every simulated pedestrian add the input goal; the goal is the final position (in x and y coordinates) of that pedestrian when it disappears from the video. This modification should improve the predicted trajectory of that pedestrian because of the introduction of this new information 
- The grid created for every pedestrian in the original project to identify nearby pedestrians is replaced with an array containing the position(in x and y coordinates) of the others pedestrians in distance order, from the closest to the farther. This modification should improve the model results beacuse it presents relevant informations in order to the neural network.    

Then these two modifications were combined in to a single model.  
Every model has been evaluated in the test videos with different parameters and in conclusion the model with the two modifications (goal and array) combined performed better than any other model. Also the two modify models performed better than the original model.   

These results can be seen in the report at page 12 and 13. Unfortunately I haven't the time to translate the report in english, because now is in italian, but the result table at page 12 an 13 should be pretty clear.   

Technical details: 
 - Programming language: Python 2.7  
 - Neural networks library used: Tensorflow 1.5 
 - External libraries: CUDA 8.0, CUDNN 6.0 
 - OS: Linux, Ubuntu 16.04 distrubution     
 
 License: GPL v3
