'''
Handles processing the input and target data in batches and sequences

Modified by : Simone Zamboni
Date : 2018-01-10
'''

import os
import pickle
import numpy as np
import ipdb
import random

class SocialDataLoader():

    # Questo costruttore non e' stato modificato rispetto a quella in Social_LSTM
    def __init__(self, batch_size=50, seq_length=5, maxNumPeds=70, datasets=[0, 1, 2, 3, 4], forcePreProcess=False, infer=False):

        # List of data directories where raw data resides (rispetto all'originale e' stato cambiato)
        self.data_dirs = ['../data/ucy/zara/zara01', '../data/ucy/zara/zara02',
                          '../data/eth/univ', '../data/eth/hotel','../data/ucy/univ']

        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer

        self.numDatasets = len(self.data_dirs)

        self.data_dir = '../data'

        self.maxNumPeds = maxNumPeds

        self.batch_size = batch_size
        self.seq_length = seq_length

        self.val_fraction = 0.2
        self.takeOneInNFrames = 6
        data_file = os.path.join(self.data_dir, "social-trajectories.cpkl")

        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(self.used_data_dirs, data_file)

        self.load_preprocessed(data_file)

        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    #funzione principale modificata
    def frame_preprocess(self, data_dirs, data_file):

        all_frame_data = []

        valid_frame_data = []

        frameList_data = []

        numPeds_data = []

        dataset_index = 0

        frames = []  # list where alla the frames are stored in the format of all_frame_data
        all_peds = []  # array with the dimension of (numDirectory,b) with b the sum of each time all the pedestian appera
        dataset_validation_index = []

        # For each dataset
        for directory in data_dirs:

            file_path = os.path.join(directory, 'pixel_pos_interpolate.csv')

            data = np.genfromtxt(file_path, delimiter=',')

            frameList = np.unique(data[0, :]).tolist()

            # Number of frames
            numFrames = int(len(frameList)/self.takeOneInNFrames)*self.takeOneInNFrames

            if self.infer:
                valid_numFrames = 0
            else:
                valid_numFrames = int((numFrames * self.val_fraction)/self.takeOneInNFrames)*self.takeOneInNFrames

            dataset_validation_index.append(valid_numFrames)

            frameList_data.append(frameList)

            numPeds_data.append([])
            all_peds.append([])

            all_frame_data.append(np.zeros( (int((numFrames - valid_numFrames)/self.takeOneInNFrames), self.maxNumPeds, 3) ) )

            valid_frame_data.append(np.zeros(  (int(valid_numFrames/self.takeOneInNFrames), self.maxNumPeds, 3) ) )

            frames.append(np.zeros((numFrames, self.maxNumPeds, 3)))

            ind = 0
            while ind < numFrames:
                frame = frameList[ind]
                pedsInFrame = data[:, data[0, :] == frame]

                pedsList = pedsInFrame[1, :].tolist()

                numPeds_data[dataset_index].append(len(pedsList))

                pedsWithPos = []

                for ped in pedsList:
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    pedsWithPos.append([ped, current_x, current_y])
                    all_peds[dataset_index].append((ped))

                if (ind >= valid_numFrames) or (self.infer):
                    all_frame_data[dataset_index][int((ind - valid_numFrames)/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)
                else:
                    valid_frame_data[dataset_index][int(ind/self.takeOneInNFrames), 0:len(pedsList), :] = np.array(pedsWithPos)

                frames[dataset_index][ind, 0:len(pedsList), :] = np.array(pedsWithPos)
                ind += self.takeOneInNFrames

            dataset_index += 1

        #passiamo ora al calcolo del "goal" di ogni pedone

        unique_all_peds = [] #array che conterra' per ogni video il numero totale di pedoni

        #Ciclo che per ogni video salva nell'array il numero di pedoni di quel video
        dir = 0
        while dir < len(data_dirs):
            unique_all_peds.append(np.unique(all_peds[dir]))
            dir += 1

        goal = []  # array contenente l'obbiettivo di ogni pedone
        # questo array ha dimensioni: (num_video, num_pedestrian_for_that_dir, 2), 2 si riferisce alla x e alla y del goal di ogni pedone

        # Inizializzazione dell'array goal con tutti 0
        dir = 0
        while dir < len(data_dirs):
            goal.append([])
            ped = 0
            #sembra che il valore len(unique_all_peds[dir]) non ritorni il numero di pedoni esatto in un video
            # e se non ci aggiungessimo una valore abbastanza alto darebbe errore.
            # Si e' quindi deciso di aggiungere una valore molto alto arbiratio per evitare errori, questa e' un
            # punto del codice che potrebbe assolutamente essere migliorato.
            while ped <= len(unique_all_peds[dir]) + 1000:
                goal[dir].append([0, 0])
                ped += 1
            dir += 1

        # per ogni frame dei video aggiornare l'ultima posizione conosciuta di quel pedone
        dir = 0
        while dir < len(frames):
            frame = 0
            while frame < len(frames[dir]):
                ped_n = 0
                #per ogni pedone in ogni frame di ogni video
                while ped_n < len(frames[dir][frame]):
                    ped_id = int(frames[dir][frame][ped_n][0]) #ricaviamo l'id del pedone attuale
                    goal[dir][ped_id][0] = frames[dir][frame][ped_n][1] #nell'array goal mettiamo le sue coordinate attuali
                    goal[dir][ped_id][1] = frames[dir][frame][ped_n][2]
                    ped_n += 1
                frame += 1
            dir += 1

        #A questo punto nell'array goal per ogni pedone di ogni frame dovremmo avere l'ultima posizione conosciuta

        # spiegazione di cosa viene salvato:
        # frameList_data[i] = tutti i numeri di frame dell'i-esimo dataset (se il dataset ha 700 frame ci sara un array di 700 elementi che vanno da 1 a 700
        # numpeds_data[i][j] = quandi pedoni ci sono nell'j-esimo frame dell'i-esimo dataset
        # all_frame_data[i][j]: della i-esima directory all j-esimo frame la lista di tutti i pedoni nell'ordine: [id,x,y], la lunghezza della lista e' maxNumPeds, e contiene i frame dopo l'ultimo valid_frame_data
        # valid frame data: uguale a all_frame_data come struttura solo che ha solo i frame di validzione
        # goal[i][j] = le coordinate x e y dell'obbiettivo pedone con id j del video i

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_frame_data, goal), f, protocol=2)
        f.close()

    #funzione modificata per caricare anche il goal
    def load_preprocessed(self, data_file):

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_data = self.raw_data[3]
        self.goals = self.raw_data[4] #prendo anche il goal dal file salvato
        counter = 0
        valid_counter = 0

        for dataset in range(len(self.data)):
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            print 'Training data from dataset', dataset, ':', len(all_frame_data)
            print 'Validation data from dataset', dataset, ':', len(valid_frame_data)
            counter += int(len(all_frame_data) / (self.seq_length+2))
            valid_counter += int(len(valid_frame_data) / (self.seq_length+2))

        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)
        self.num_batches = self.num_batches * 2

    #funzione modificata per fare in modo che nei dati del batch ci sia anche il goal del pedone
    def next_batch(self, randomUpdate=True):
        x_batch = []
        y_batch = []
        d = []
        i = 0

        while i < self.batch_size:
            frame_data = self.data[self.dataset_pointer]

            idx = self.frame_pointer

            if idx + self.seq_length < frame_data.shape[0]:

                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                #list of the ID of all the pedestrian in the current batch
                pedID_list = np.unique(seq_frame_data[:, :, 0])

                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # sia sourceData che targetData sono stati ampliati da 3 a 5
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                #per ogni frame della sequenza
                for seq in range(self.seq_length):
                    # frame attuale (ssqe_frame_data) e successivo (tseq_frame_data)
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    #per tutti i pedoni nel frame
                    for ped in range(numUniquePeds):
                        #prendere il pedID
                        pedID = pedID_list[ped]

                        #se il pedone non esiste andare avanti al prossimo giro
                        if pedID == 0:
                            continue
                        else:
                            tped = [] #target data per questo pedone
                            sped = [] #sequence data per questo pedone

                            #prendere la posizione del pedone nel frame
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            #se quel pedone e' presente nel frame, cioe' se ha una posizione(ed e' quindi salvata in temp_sped) allora si va avanti
                            if len(temp_sped) > 0 :
                                #aggiungere ai dati di input del pedone la posizione del pedone
                                iter= 0
                                while iter < len (temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter+=1

                                #e aggiungere i dati di input del pedone le coordinate del goal del pedone
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            #array temporameo che contiene i dati della posizione futura del pedone
                            temp_tped = tseq_frame_data[sseq_frame_data[:, 0] == pedID, :]
                            #se quel pedone ha dati target, cioe' se ha una posizione anche nel frame successivo (questa quindi sara' salvata in temp_sped)
                            if(len(temp_tped) > 0) :
                                iter = 0
                                #aggiungere ai dati target di quel pedone la sua posizione futura
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1
                                # e aggiungere i dati target di quel pedone anche le coordinate del goal
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            #se sono state inserite delle informazioni in sped e tped allora vengono aggiunti a sourceData e targetData
                            if len(sped) > 2:
                                sourceData[seq, ped, :] = sped
                            if len(tped) > 2:
                                targetData[seq, ped, :] = tped

                x_batch.append(sourceData)
                y_batch.append(targetData)

                if randomUpdate:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d

    # funzione modificata per fare in modo che nei dati del batch ci sia anche il goal del pedone, modifiche praticamente identiche alla funzione precedente
    def next_valid_batch(self, randomUpdate=True):
        x_batch = []
        y_batch = []
        d = []
        i = 0
        while i < self.batch_size:
            frame_data = self.valid_data[self.valid_dataset_pointer]
            idx = self.valid_frame_pointer

            if idx + self.seq_length < frame_data.shape[0]:
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]

                # list of the ID of all the pedestrian in the current batch
                pedID_list = np.unique(seq_frame_data[:, :, 0])
                # Number of unique peds the current batch
                numUniquePeds = pedID_list.shape[0]

                # sia sourceData che targetData sono stati ampliati da 3 a 5
                sourceData = np.zeros((self.seq_length, self.maxNumPeds, 5))
                targetData = np.zeros((self.seq_length, self.maxNumPeds, 5))

                # per ogni frame della sequenza
                for seq in range(self.seq_length):
                    # frame attuale (ssqe_frame_data) e successivo (tseq_frame_data)
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]

                    # per tutti i pedoni nel frame
                    for ped in range(numUniquePeds):
                        pedID = pedID_list[ped] #prendere il pedID

                        # se il pedone non esiste andare avanti al prossimo ciclo
                        if pedID == 0:
                            continue
                        else:
                            tped = [] #target data per questo pedone
                            sped = [] #sequence data per questo pedone

                            # array che contiene la posizione del pedone nel frame
                            temp_sped = sseq_frame_data[sseq_frame_data[:, 0] == pedID, :]

                            # se quel pedone e' presente nel frame, cioe' se ha una posizione(ed e' quindi salvata in temp_sped) allora si va avanti
                            if(len(temp_sped) > 0):
                                # aggiungere ai dati di input del pedone la posizione del pedone
                                iter = 0
                                while iter < len(temp_sped[0]):
                                    sped.append(temp_sped[0][iter])
                                    iter += 1

                                # e aggiungere i dati di input del pedone le coordinate del goal del pedone
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                sped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # array che contiene i dati della posizione futura del pedone
                            temp_tped = (tseq_frame_data[tseq_frame_data[:, 0] == pedID, :])

                            # se quel pedone ha dati target, cioe' se ha una posizione anche nel frame successivo (questa quindi sara' salvata in temp_sped)
                            if(len(temp_tped) > 0) :
                                # aggiungere ai dati target di quel pedone la sua posizione futura
                                iter = 0
                                while iter < len(temp_tped[0]):
                                    tped.append(temp_tped[0][iter])
                                    iter += 1

                                # e aggiungere i dati target di quel pedone anche le coordinate del goal
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][0])
                                tped.append(self.goals[self.dataset_pointer][int(pedID)][1])

                            # se sono state inserite delle informazioni in sped e tped allora vengono aggiunti a sourceData e targetData
                            if len(sped) > 2:
                                sourceData[seq, ped, : ] = sped
                            if len(tped) > 2:
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
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    #funzione non modificata
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

    #funzione non modificata
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




