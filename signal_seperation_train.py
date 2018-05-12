from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import collections
import numpy as np
import os
import random

def get_next_batch(num_frequencies, gamma, batch_size):
    batch_indices = np.random.randint(low=0, high=len(experience),
                                      size=batch_size)
    batch = [experience[i] for i in batch_indices]
    #batch = experience[batch_indices]
    # batch is a list of experiences: (prev_state, action, score, state, continue_round)
    X = np.zeros((batch_size, 4, 4, 1)) 
    # X is a batch_size of frames.
    Y = np.zeros((batch_size, num_actions)) 
    # Y is a batch size of rewards (for each action).
    for i in range(len(batch)):
        #s_t, a_t, r_t, s_tp1, game_over = batch[i]
        prev_state, action, score, state, continue_round = batch[i]
        X[i] = prev_state
        Y[i] = model.predict(prev_state)[0]
        Q_sa = np.max(model.predict(state)[0])
        if continue_round == False:
            Y[i, action] = score
        else:
            Y[i, action] = score + gamma * Q_sa
    return X, Y

if __name__ == "__main__":   
############################# main ###############################

    # initialize parameters
    DATA_DIR = ""
    NUM_FREQUENCY_BINS = 100 # number of frequencies
    NUM_EPOCHS_TRAIN = 1000

    BATCH_SIZE = 32

    modelFile = "signal-seperation-network.h6"

    if os.path.exists(modelFile):
        model = load_model(os.path.join(DATA_DIR, modelFile))
    else:
        # build the model
        model = Sequential()
        model.add(Conv2D(16, kernel_size=2, strides=1,
                 kernel_initializer="normal", 
                 padding="same",
                 input_shape=(4, 4, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(32, kernel_size=2, strides=1, 
                 kernel_initializer="normal", 
                 padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(32, kernel_size=2, strides=1,
                 kernel_initializer="normal",
                 padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer="normal"))
        model.add(Activation("relu"))
        model.add(Dense(3, kernel_initializer="normal"))

    model.compile(optimizer=Adam(lr=1e-6), loss="mse")

    # train network
    game = blackjack.Blackjack()
    game.should_print = False
    experience = collections.deque(maxlen=MEMORY_SIZE)

    fout = open(os.path.join(DATA_DIR, "blackjack-simple-results.tsv"), "w")
    num_games, num_wins = 0, 0
    epsilon = INITIAL_EPSILON
    for e in range(NUM_EPOCHS):
        loss = 0.0
        game.reset()
        # get first state
        continue_game = game.start_round()
        while (continue_game):
            prev_state = preprocess_rounds(np.array(list(game.rounds)))
            if e <= NUM_EPOCHS_OBSERVE or np.random.rand() <= epsilon:
                action = random.randint(0, 2)
            else:
                q = model.predict(prev_state)[0]
                action = np.argmax(q)
                
            continue_game, continue_round, score, state = game.computer_turn(action)
            state = preprocess_rounds(state)
            if(score > 0):
                num_wins += 1

            experience.append((prev_state, action, score, state, continue_round))
            while(continue_round):
                prev_state = state
                if e <= NUM_EPOCHS_OBSERVE or np.random.rand() <= epsilon:
                    action = random.randint(0, 2)
                else:
                    q = model.predict(prev_state)[0]
                    action = np.argmax(q)

                continue_game, continue_round, score, state = game.computer_turn(action)
                state = preprocess_rounds(state)
                if(score > 0):
                    num_wins += 1

                experience.append((prev_state, action, score, state, continue_round))

            #train after every round
            if (e > NUM_EPOCHS_OBSERVE):
                X, Y = get_next_batch(experience, model, NUM_ACTIONS, 
                                  GAMMA, BATCH_SIZE)
                loss += model.train_on_batch(X, Y)
                
            if(continue_game):
                continue_game = game.start_round()

        # reduce epsilon gradually ever epoc
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / NUM_EPOCHS

        if e % 20 == 0:
            print("\nEpoch {:04d}/{:d} | Loss {:.5f} | Win Count: {:d}" 
                  .format(e+1, NUM_EPOCHS, loss, num_wins), end="\n")

        fout.write("{:04d}\t{:.5f}\t{:d}\n" 
                   .format(e+1, loss, num_wins))

        if e % 100 == 0:
            model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
                
    print("")
    fout.close()
    model.save(os.path.join(DATA_DIR, modelFile), overwrite=True)
