# Imports.
import numpy as np
import numpy.random as npr
import math
from collections import defaultdict

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self, method = "SARSA", e_greedy = True, gravity_inf = True):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.current_action = None
        self.current_state  = None
        self.method = method
        self.e_greedy = e_greedy
        self.gravity_inf = gravity_inf
        
        # parameters
        self.q = {}
        self.eps = 0.1
        self.gamma = 0.9
        self.directions = range(0,2) # action of jump or swing
        self.eta = 0.1
        self.iter = 0
        self.k = defaultdict(int)

        #segmentation:
        self.width_bin = 50
        self.top_bin = 50

        if gravity_inf == False:
            # segmentation:
            self.vel_bin = 20
        else:
            self.gravity = 0
            # segmentation:
            self.vel_bin = 20
            self.flag = 0


    def get_state(self,state):
        dist = math.floor(state['tree']['dist']/self.width_bin)
        vel = math.floor(state['monkey']['vel']/self.vel_bin)
        top = math.floor((state['tree']['top']-state['monkey']['top'])/self.top_bin)
        if np.abs(vel) > 2:  
            vel = np.sign(vel)*2
        if self.gravity_inf == False:
            return (dist,vel,top)
        else:
            gravity = self.gravity
            return (dist,vel,top,gravity)



    def getQ(self, state, action):
        #return the current Q value, if no value existed, return 0.0
        return self.q.get((state,action),0.0)

    #function to update the Q value for SARSA algorithm
    def updateQ_sarsa(self, state, action, reward, state_):
        oldQ = self.q.get((state,action))
        if oldQ is None:
            self.q[(state,action)] = reward
            self.k[(state,action)] = 1
        else:
            self.k[(state,action)] += 1
            #self.q[(state,action)] = oldQ - self.eta*(oldQ-(reward+self.gamma*self.getQ(state_, self.sarsa_action(state_))))
            self.q[(state,action)] = oldQ + 1/(self.k[(state,action)])*(reward+self.gamma*self.getQ(state_, self.sarsa_action(state_))-oldQ)

    #function to update the Q value for Q_learning algorithm
    def updateQ_learning(self, state, action, reward, state_):
        oldQ = self.q.get((state,action))
        if oldQ is None:
            self.q[(state,action)] = reward
            self.k[(state,action)] = 1
        else:
            self.k[(state,action)] += 1
            #self.q[(state,action)] = oldQ - self.eta*(oldQ-(reward+self.gamma*self.getQ(state_, self.Q_action(state_))))
            self.q[(state,action)] = oldQ + 1/(self.k[(state,action)])*(reward+self.gamma*self.getQ(state_, self.Q_action(state_))-oldQ)


    def sarsa_action(self, state):
        if self.e_greedy == True:
            eps = self.eps/(self.iter+1)
        else:
            eps = self.eps
        if npr.random() < eps:
            return 1 if npr.rand() < 0.5 else 0
        else:
            val = [self.getQ(state, a) for a in self.directions]
            i = val.index(max(val))
            return i

    def Q_action(self, state):
        val = [self.getQ(state, a) for a in self.directions]
        i = val.index(max(val))
        return i


    def reset(self):
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.iter += 1
        if self.gravity_inf == True:
            self.flag = 0
            self.gravity = 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        #select action
        self.last_state = self.current_state
        self.current_state = state
        if self.last_state != None and self.current_state != None and self.last_action != None:
            state = self.get_state(self.last_state)
            state_ = self.get_state(self.current_state)
            action = self.last_action
            if self.method =="SARSA":
                self.updateQ_sarsa(state, action, self.last_reward, state_)
            elif self.method == "Q_learning":
                self.updateQ_learning(state, action, self.last_reward, state_)

        if self.current_state == None:
            action = 1 if npr.rand() < 0.5 else 0
        else:
            action = self.sarsa_action(self.get_state(self.current_state))
        self.last_action = action
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    print("epoch", "\t", "score", "\t", "high", "\t", "avg")
    highscore, avgscore= 0.0,0.0
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)


        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        score = swing.score
        highscore = max([highscore, score])
        avgscore = (ii*avgscore+score)/(ii+1)

        print (ii, "\t", score, "\t", highscore, "\t", avgscore)

        # Reset the state of the learner.
        learner.reset()
        
    return


# if __name__ == '__main__':

#     # Select agent.
#     # agent = Learner(method = "Q_learning", e_greedy= False, gravity_inf = False)
#     # hist_q = []
#     # run_games(agent, hist_q, 400, 1)
#     # np.save('hist_q',np.array(hist_q))

#     # agent = Learner(method = "SARSA", e_greedy= False, gravity_inf = False)
#     # hist_s = []
#     # run_games(agent, hist_s, 400, 1)
#     # np.save('hist_s',np.array(hist_s))

#     # agent = Learner(method = "Q_learning", e_greedy= True, gravity_inf = False)
#     # hist_q_e = []
#     # run_games(agent, hist_q_e, 400, 1)
#     # np.save('hist_q_e',np.array(hist_q_e))

#     # agent = Learner(method = "SARSA", e_greedy= True, gravity_inf = False)
#     # hist_s_e = []
#     # run_games(agent, hist_s_e, 400, 1)
#     # np.save('hist_s_e',np.array(hist_s_e))

#     # agent = Learner(method = "Q_learning", e_greedy= True, gravity_inf = True)
#     # hist_q_e_g = []
#     # run_games(agent, hist_q_e_g, 400, 1)
#     # np.save('hist_q_e_g',np.array(hist_q_e_g))

#     agent = Learner(method = "SARSA", e_greedy= True, gravity_inf = True)
#     hist_s_e_g = []
#     run_games(agent, hist_s_e_g, 400, 1)
#     np.save('hist_s_e_g',np.array(hist_s_e_g))



