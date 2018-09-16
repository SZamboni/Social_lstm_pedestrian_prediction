'''
Interface to test the trained model on custom scenarios

Author : Anirudh Vemula
Date : 17th November 2016
'''

import numpy as np
import tensorflow as tf

import os
import pickle
import argparse

from social_model import SocialModel
from grid import getSequenceGridMask


def get_mean_error(predicted_traj, true_traj, observed_length, maxNumPeds):
    '''
    Function that computes the mean euclidean distance error between the
    predicted and the true trajectory
    params:
    predicted_traj : numpy matrix with the points of the predicted trajectory
    true_traj : numpy matrix with the points of the true trajectory
    observed_length : The length of trajectory observed
    '''
    # The data structure to store all errors
    error = np.zeros(len(true_traj) - observed_length)
    # For each point in the predicted part of the trajectory
    for i in range(observed_length, len(true_traj)):
        # The predicted position. This will be a maxNumPeds x 3 matrix
        pred_pos = predicted_traj[i, :]
        # The true position. This will be a maxNumPeds x 3 matrix
        true_pos = true_traj[i, :]
        timestep_error = 0
        counter = 0
        for j in range(maxNumPeds):
            if true_pos[j, 0] == 0:
                # Non-existent ped
                continue
            elif pred_pos[j, 0] == 0:
                # Ped comes in the prediction time. Not seen in observed part
                continue
            else:
                if true_pos[j, 1] > 1 or true_pos[j, 1] < 0:
                    continue
                elif true_pos[j, 2] > 1 or true_pos[j, 2] < 0:
                    continue

                timestep_error += np.linalg.norm(true_pos[j, [1, 2]] - pred_pos[j, [1, 2]])
                counter += 1

        if counter != 0:
            error[i - observed_length] = timestep_error / counter

        # The euclidean distance is the error
        # error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)

    # Return the mean error
    return np.mean(error)


def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=4,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=4,
                        help='Predicted length of the trajectory')
    # Custom scenario to be tested on
    parser.add_argument('--scenario', type=str, default='collision',
                        help='Custom scenario to be tested on')

    sample_args = parser.parse_args()

    # Define the path for the config file for saved args
    with open(os.path.join('save', 'social_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Create a SocialModel object with the saved_args and infer set to true
    model = SocialModel(saved_args, True)
    # Initialize a TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize a saver
    saver = tf.train.Saver()

    # Get the checkpoint state for the model
    ckpt = tf.train.get_checkpoint_state('save')
    print ('loading model: ', ckpt.model_checkpoint_path)

    # Restore the model at the checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)

    results = []

    # Load the data
    file_path = os.path.join('matlab', 'csv', sample_args.scenario+'.csv')
    data = np.genfromtxt(file_path, delimiter=',')
    # Reshape data
    x_batch = np.reshape(data, [sample_args.obs_length+sample_args.pred_length, saved_args.maxNumPeds, 3])

    dimensions = [720, 576]
    grid_batch = getSequenceGridMask(x_batch, [720, 576], saved_args.neighborhood_size, saved_args.grid_size)

    obs_traj = x_batch[:sample_args.obs_length]
    obs_grid = grid_batch[:sample_args.obs_length]


    complete_traj = model.sample(sess, obs_traj, obs_grid, dimensions, x_batch, sample_args.pred_length)

    total_error = get_mean_error(complete_traj, x_batch, sample_args.obs_length, saved_args.maxNumPeds)

    print "Mean error of the model on this scenario is", total_error

if __name__ == '__main__':
    main()
