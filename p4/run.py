import sys
from stub_final import *
from optparse import OptionParser
import numpy as np
def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog -m method -e epsilon_modify -g gravity_inf(pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-m", "--method", dest="method",
                  help="The method we use for RL, choose between SARSA and Q_Learning")
    parser.add_option("-e", "--epsilon_modify", dest="epsilon_modify",
                  help="Indicator to choose whether we use the epsilon modified method, input True or False")
    parser.add_option("-g", "--gravity_inference", dest="gravity_inf",
                  help="Indicator to choose whether we consider gravity inference, input True or False")

    (options, args) = parser.parse_args(argv[1:])

    if options.method == "Q_learning" and options.epsilon_modify == 'False' and options.gravity_inf == 'False':
        agent = Learner(method = "Q_learning", e_greedy= False, gravity_inf = False)
        hist_q = []
        run_games(agent, hist_q, 400, 1)
        np.save('hist_q',np.array(hist_q))

    elif options.method == "Q_learning" and options.epsilon_modify == 'True' and options.gravity_inf == 'False':
        agent = Learner(method = "Q_learning", e_greedy= True, gravity_inf = False)
        hist_q_e = []
        run_games(agent, hist_q_e, 400, 1)
        np.save('hist_q_e',np.array(hist_q_e))

    elif options.method == "Q_learning" and options.epsilon_modify == 'True' and options.gravity_inf == 'True':
        agent = Learner(method = "Q_learning", e_greedy= True, gravity_inf = True)
        hist_q_e_g = []
        run_games(agent, hist_q_e_g, 400, 1)
        np.save('hist_q_e_g',np.array(hist_q_e_g))

    elif options.method == "SARSA" and options.epsilon_modify == 'False' and options.gravity_inf == 'False':
        agent = Learner(method = "SARSA", e_greedy= False, gravity_inf = False)
        hist_s = []
        run_games(agent, hist_s, 400, 1)
        np.save('hist_s',np.array(hist_s))

    elif options.method == "SARSA" and options.epsilon_modify == 'True' and options.gravity_inf == 'False':
        agent = Learner(method = "SARSA", e_greedy= True, gravity_inf = False)
        hist_s_e = []
        run_games(agent, hist_s_e, 400, 1)
        np.save('hist_s_e',np.array(hist_s_e))

    elif options.method == "SARSA" and options.epsilon_modify == 'True' and options.gravity_inf == 'True':
        agent = Learner(method = "Q_learning", e_greedy= True, gravity_inf = True)
        hist_q_e_g = []
        run_games(agent, hist_q_e_g, 400, 1)
        np.save('hist_q_e_g',np.array(hist_q_e_g))

    else:
        agent = Learner()
        hist = []
        run_games(agent, hist, 400, 1)
        np.save('hist',np.array(hist))



if __name__ == "__main__":
    sys.exit(main())
    