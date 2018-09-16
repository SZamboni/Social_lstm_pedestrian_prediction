'''
Funzioni che calcolano l'array sociale

Modified by: Simone Zamboni
'''
import numpy as np
import math


def getGridMask(frame, dimensions, neighborhood_size, grid_size):
    '''
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    neighborhood_size : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    grid_size : Quanti pedoni saranno presenti nell'array
    '''

    # Estrarre e salvare il massimo numero di pedoni nel frame
    mnp = frame.shape[0]

    #Array che contiene per ogni pedone nel frame il suo array sociale
    my_array = np.zeros((mnp,grid_size*2))

    #Per ogni pedone nell'array viene creato il suo array sociale
    for pedindex in range(mnp):

        #se il pedone non esiste (quindi ha ID = 0) si passa al prossimo ciclo
        if(frame[pedindex,0] == 0):
           continue

        #prendere la posizione attuale del pedone preso in considerazione
        current_x,current_y =  frame[pedindex, 1], frame[pedindex, 2]
        other_peds_with_position = []

        #per ogni pedone nel frame
        for otherpedindex in range(mnp):

            #se il pedone non esiste (quindi ha ID = 0) si passa al prossimo ciclo
            if frame[otherpedindex, 0] == 0:
                continue

            #otherpedindex e' uguale a pedindex e quindi sono lo stesso pedone, quindi si passa al prossimo ciclo, perche' un pedone non puo' essere presente nel suo stesso array
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                continue

            #calcolare la distanza dal pedone attuale al pedone otherpedindex
            current_distance = math.sqrt( math.pow((current_x - frame[otherpedindex][1]),2) + math.pow((current_y - frame[otherpedindex][2]),2)  )

            #salvare nell'array other_peds_with_position ID,x,y,distanza_dal_pedone_attuale del pedone otherpedindex
            other_peds_with_position.append( [frame[otherpedindex][0],frame[otherpedindex][1],frame[otherpedindex][2],current_distance])

        #ora abbiamo un array contenente ID,x,y e distanza di tutti gli altri pedoni validi
        #se dopo aver controllato tutto il frame non vi sono altri pedoni all'infuori di quello attuale un pedone finto viene inserito
        if (len(other_peds_with_position) == 0):
            #questo pedone avra' coordinate x-2 e y-2 rispetto al pedone attuale, cosi' da essere molto lontano
            other_peds_with_position.append([0, frame[pedindex,1]-2, frame[pedindex,1]-2, 2,828427125])
            
        #numero di quanti altri pedoni sono stati trovati
        num_other_peds = len(other_peds_with_position)

        #scorro l'array sociale del pedone attuale e lo riempio
        j = 0 #indica il pedone j-esimo nell'array sociale, quindi j*2 indica la coordinata x e j*2+1 indica la coordinata y del j-esimo pedone nell'array sociale
        while j < len(my_array[pedindex]):
            x = 0 #usato per scorrerei pedoni nell'array other_peds_with_position

            # array che contiene in prima posizione la distanza minima trovata e in seconda la posizione del pedone con la minima distanza nell'array other_peds_with_position
            min_distance = [1000000,0]
            update = False

            # per ogni pedone nell'array other_peds_with_position si cerca il piu' vicino al pedone attuale
            while x < len(other_peds_with_position):
                # se il pedone x ha la distanza finora minore salvo la distanza in min_distance[0] e la posizione del pedone in min_distance[1]
                if(other_peds_with_position[x][3] < min_distance[0]):
                    min_distance[0] = other_peds_with_position[x][3]
                    min_distance[1] = x
                    update = True #indica che e' stato trovato un pedone con distanza minore di quella di default
                x+=1

            # solo se e' stato trovato un pedone con distanza minore di quella di default salviamo questo pedone nell'array sociale
            if(update == True):
                # salviamo nell'array alla posizone j-esima le coordinate del pedone con distanza minore
                my_array[pedindex][j] = other_peds_with_position[min_distance[1]][1] # x del pedone piu' vicino
                my_array[pedindex][j+1] = other_peds_with_position[min_distance[1]][2] # y del pedone piu' vicino
                # e poi eliminiamo dall'array quel pedone cosi' da non ripeterlo
                other_peds_with_position.remove(other_peds_with_position[min_distance[1]])
            j += 2

        #abbiamo ora nell'array sociale del pedone le coordinate dei pedoni vicini in ordine di vicinanza
        #se gli altri pedoni fossero >= grid_size ci potremmo fermare qui
        #la parte che segui copre l'eventualita' che gli altri pedoni siano <= grid_size e quindi
        # che nell'array sociale del pedone attuale ci siano ancora posizioni vuote

        #contiamo quanti pedoni mancheranno da inserire nell'array sociale
        num_peds_missing = - num_other_peds + (len(my_array[0])/2)

        # se e' un numero maggiore di 0 riempiremo l'array sociale con le ripetizioni degli altri pedoni
        if(num_peds_missing > 0):
            i = 0

            #per ogni spazio vuoto ripeto i pedoni gia' presenti nell'array a partire dal primo
            while i < num_peds_missing:
                my_array[pedindex][ (len(my_array[0])/2 - num_peds_missing + i) *2] = my_array[pedindex][i*2]
                my_array[pedindex][ (len(my_array[0])/2 - num_peds_missing + i) *2 + 1] = my_array[pedindex][i*2+1]
                i+=1

    #stampare l'array sociale
    i = 0
    while i < len(my_array):
        #if(frame[i,0] != 0):
            #print("pedestrian in frame n " + str(i) + " proximity array : " +str(my_array[i]))
        i+=1

    #print("Frame : " + str(frame))

    return my_array


def getSequenceGridMask(sequence, dimensions, neighborhood_size, grid_size):
    '''
    params:
    sequence : array con tutti i frame della sequenza, ha dimensioni SL x MNP x 3
    dimensions : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    neighborhood_size : inutile, tenuto per semplificare l'implementaizione e modificare solamente questo file
    grid_size : Quanti pedoni saranno presenti nell'array
    '''
    sl = sequence.shape[0] # estrarre e salvare il parametro sequence_length
    mnp = sequence.shape[1] #estrarre e salvare il parametro MaxNumPeds

    #l'array contentente l'array sociale di tutti i pedoni per ogni frame dela sequenza
    #ha dimensioni sequece_length X MaxNumPeds X grid_size*2
    sequence_mask = np.zeros((sl, mnp, grid_size*2))

    #per ogni frame della sequenza richiama la funzione getGridMark aggiungendo il risultato a sequence_mask
    for i in range(sl):
        sequence_mask[i, :, :] = getGridMask(sequence[i, :, :], dimensions, neighborhood_size, grid_size)

    return sequence_mask
