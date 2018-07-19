import gym
import random
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from keras.models import Sequential
from keras.layers import Dense
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0").env
env.reset()
goal_steps = 50000000
score_requirement = 50
initial_games = 10000
n_games = 1000

                
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

def neural_network_model(input_size):
    model = Sequential()
    
    #network = input_data(shape=[None, input_size, 1], name='input')
    model.add(Dense(units=2, activation='relu', input_dim=input_size))

    #network = fully_connected(network, 128, activation='relu')
    #network = dropout(network, 0.8)
    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(units=256, activation='relu'))
    
    model.add(Dense(units=512, activation='relu'))

    model.add(Dense(units=256, activation='relu'))

    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    #model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True, run_id='openai_learning')
    X = X[:,:,0]
    X = np.array(X)
    y = np.array(y)
    model.fit(X,y, epochs=5)
    return model


def n_population(model,media):
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(n_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            if len(prev_observation)==0:
                action = random.randrange(0,2)
            else:
                action = prev_observation.reshape(-1,len(prev_obs),1)
                action = action[:,:,0]
                action = np.array(action)
                action = np.argmax(model.predict(action))
            observation, reward, done, info = env.step(action)     
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break
        if score >= media:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

#some_random_games_first()

training_data = initial_population()
model = train_model(training_data)
for i in range(0,11):

    scores = []
    choices = []
    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            #env.render()
    
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = prev_obs.reshape(-1,len(prev_obs),1)
                action = action[:,:,0]
                action = np.array(action)
                action = np.argmax(model.predict(action))
    
            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break
    
        scores.append(score)
    print('Geracao: ',i)
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    next_score = score_requirement +  sum(scores)/len(scores)
    training_data = n_population(model,next_score)
    model = train_model(training_data)
