'''
Utils script for the social LSTM implementation
Handles processing the input and target data in batches and sequences

Author : Anirudh Vemula
Date : 17th October 2016
'''

import os
import pickle
import numpy as np
import ipdb
import random

# The data loader class that loads data from the datasets considering
# each frame as a datapoint and a sequence of consecutive frames as the
# sequence.
class SocialDataLoader():

    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=70, datasets=[0, 1, 2, 3, 4], forcePreProcess=False, infer=False):
        '''
        Initialiser function for the SocialDataLoader class
        params:
        batch_size : Size of the mini-batch
        grid_size : Size of the social grid constructed
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        #original: self.data_dirs = ['../data/eth/univ', '../data/eth/hotel','../data/ucy/zara/zara01', '../data/ucy/zara/zara02','../data/ucy/univ']
        self.data_dirs = ['../data/ucy/zara/zara01', '../data/ucy/zara/zara02',
                          '../data/eth/univ', '../data/eth/hotel', '../data/ucy/univ']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Data directory where the pre-processed pickle file resides
        self.data_dir = '../data'

        # Maximum number of peds in a single frame (Number obtained by checking the datasets)
        self.maxNumPeds = maxNumPeds

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validation arguments
        self.val_fraction = 0.2
        self.takeOneEveryNFrames = 6

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # pedId, x, y , in each frame is stored
        all_frame_data = []
        # Validation frame data
        valid_frame_data = []
        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:

            # Define path of the csv file of the current dataset
            # file_path = os.path.join(directory, 'pixel_pos.csv')
            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            dataset_validation_index = []
            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            # Number of frames
            numFrames = int(len(frameList) / self.takeOneEveryNFrames) * self.takeOneEveryNFrames

            if self.infer:
                valid_numFrames = 0
            else:
                valid_numFrames = int((numFrames * self.val_fraction) / self.takeOneEveryNFrames) * self.takeOneEveryNFrames

            dataset_validation_index.append(valid_numFrames)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)
            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            # Initialize the numpy array for the current dataset
            all_frame_data.append(np.zeros((int((numFrames - valid_numFrames) / self.takeOneEveryNFrames), self.maxNumPeds, 3)))
            # Initialize the numpy array for the current dataset
            valid_frame_data.append(np.zeros((int(valid_numFrames / self.takeOneEveryNFrames), self.maxNumPeds, 3)))

            # index to maintain the current frame
            curr_frame = 0
            ind = 0
            while ind < numFrames:
                frame = frameList[ind]
                # Extract all pedestrians in current frame
                pedsInFrame = data[:, data[0, :] == frame]

                # Extract peds list
                pedsList = pedsInFrame[1, :].tolist()

                # Helper print statement to figure out the maximum number of peds in any frame in any dataset
                # if len(pedsList) > 1:
                # print len(pedsList)
                # DEBUG
                #    continue

                # Add number of peds in the current frame to the stored data
                numPeds_data[dataset_index].append(len(pedsList))

                # Initialize the row of the numpy array
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array
                    pedsWithPos.append([ped, current_x, current_y])

                if (ind >= valid_numFrames) or (self.infer):
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index][int((ind - valid_numFrames)/self.takeOneEveryNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)
                else:
                    valid_frame_data[dataset_index][int(ind/self.takeOneEveryNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)
                # Increment the frame index
                curr_frame += 1
                ind += self.takeOneEveryNFrames
            # Increment the dataset index
            dataset_index += 1

        dir = 0
        while dir < len(all_frame_data):
            frame = 0
            while frame < len(all_frame_data[dir]):
                #print("dir " + str(dir) + " frame " + str(frame) + " : " + str(all_frame_data[dir][frame]))
                frame +=1
            dir+=1

        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print 'Training data from dataset', dataset, ':', len(all_frame_data)
            print 'Validation data from dataset', dataset, ':', len(valid_frame_data)
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length+2))
            valid_counter += int(len(valid_frame_data) / (self.seq_length+2))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]

                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 3))

                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    if numUniquePeds > self.maxNumPeds:
                        print ("Max num peds surpassed: " + str(numUniquePeds) + " out of " + str(self.maxNumPeds))
                        numUniquePeds = self.maxNumPeds

                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]

                        if pedID == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])
                            if sped.size != 0:
                                sourceData[seq, ped, :] = sped
                            if tped.size != 0:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                numUniquePeds = pedID_list.shape[0]

                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 3))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 3))

                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    if numUniquePeds > self.maxNumPeds:
                        print ("Max num peds surpassed: " + str(numUniquePeds) + " out of " + str(self.maxNumPeds))
                        numUniquePeds = self.maxNumPeds

                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped]

                        if pedID == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])
                            if sped.size != 0:
                                sourceData[seq, ped, :] = sped
                            if tped.size != 0:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                # Advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0    

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0
