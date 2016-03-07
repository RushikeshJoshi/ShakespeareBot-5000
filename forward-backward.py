# Solution set for CS 155 Set 4, 2016
# Authors: Fabian Boemer
# Edited: Kevin Tang
#
# Remember to edit the filename to point to your sequence prediction files.
# Alternatively, import this file and run Viterbi and Forward yourself.


def main():
    """ This program computes the max-probability state sequence and probability
        of the observed sequences stored in ./data/
        The Viterbi algorithm is used to compute the maximum-probability state
        sequence, and the Forward algorithm is used to compute the probability
        of observing the given sequence. Output is stored to question1D.txt
    """
    num_files = 5                      # number of sequences
    w = open('1F.txt', 'w')   # text output file
    latex_table = 'Dataset & Sequence & Max-probability state ' + \
        'sequence & Probability of sequence  \\\\ \hline \n'

    for fileid in range(num_files):
        filename = '../data/sequenceprediction' + str(fileid + 1) + '.txt'

        A, O, sequences = LoadHMM(filename)
        num_states = len(A)

        for obs in sequences:
            latex_table += str(fileid + 1) + ' & '
            latex_table += ''.join([str(x) for x in obs]) + ' & '
            latex_table += Viterbi(num_states, obs, A, O) + ' & '
            latex_table += '%.3e\\\\\n' % (Forward(num_states, obs, A, O))
        latex_table = latex_table[:-3]
        latex_table += '\n \\\\ \\hline\n'

    w.write(latex_table)
    w.close()


def LoadHMM(filename):
    """ Loads a HMM file. Returns in format A, O, sequences
    """
    A = []  # transition matrix
    O = []  # observation matrix
    sequences = []

    with open(filename, 'r') as f:
        num_states, num_obs = [int(x)
                               for x in f.readline().strip().split('\t')]
        for i in range(num_states):
            A.append([float(x) for x in f.readline().strip().split('\t')])
        for i in range(num_states):
            O.append([float(x) for x in f.readline().strip().split('\t')])
        for i in range(5):
            sequences.append([int(x) for x in list(f.readline().strip())])
    return (A, O, sequences)


def Viterbi(states, obs, A, O):
    """ Finds the max-probability state sequence for a given HMM and observation
        using the Viterbi Algorithm. This is a dynamic programming approach.
        The function uses 'prob' and 'seq' to store the probability and the
        sequence, respectively, of the most-likely sequences at each length.
        Arguments: states the number of states
                   obs    an array of observations
                   A      the transition matrix
                   O      the observation matrix
        Returns the most-likely sequence
    """
    len_ = len(obs)
    # stores p(best_seqence)
    prob = [[[0] for i in range(states)] for j in range(len_)]

    # stores most-likely sequence
    seq = [[[''] for i in range(states)] for i in range(len_)]

    # initializes uniform state distribution
    prob[0] = [staterow[obs[0]] / len(A) for staterow in O]

    # initialize best sequence of length 1
    seq[0] = [str(i) for i in range(states)]

    # We iterate through all indices in the data
    for length in range(1, len_):   # length + 1 to avoid initial condition
        for state in range(states):
            max_state = 0
            best_prob = 0

            # We iterate through all possible transitions from previous state
            for prev in range(states):

                # cur_prob is the probability of transitioning to 'state'
                # from 'prev' state and observing the correct state.
                cur_prob = prob[length - 1][prev] * A[prev][state] *\
                    O[state][obs[length]]
                if cur_prob > best_prob:
                    max_state, best_prob = prev, cur_prob

                # update best probability
                prob[length][state] = best_prob
                # update sequence
                seq[length][state] = seq[length - 1][max_state] + str(state)

        prob[length] = prob[length][:]   # copies by value
        seq[length] = seq[length][:]

    max_ind = 0
    for i in range(states):  # find most-likely index of entire sequence
        if prob[len_ - 1][i] > prob[len_ - 1][max_ind]:
            max_ind = i

    # returns most-likely sequence
    return seq[len_ - 1][max_ind]


def Forward(num_states, obs, A, O):
    """Computes the probability a given HMM emits a given observation using the
        forward algorithm. This uses a dynamic programming approach, and uses
        the 'prob' matrix to store the probability of the sequence at each length.
        Arguments: num_states the number of states
                   obs        an array of observations
                   A          the transition matrix
                   O          the observation matrix
        Returns the probability of the observed sequence 'obs'
    """
    len_ = len(obs)                   # number of observations
    # stores p(seqence)
    prob = [[[0.] for i in range(num_states)] for i in range(len_)]

    # initializes uniform state distribution, factored by the
    # probability of observing the sequence from the state (given by the
    # observation matrix)
    prob[0] = [(1. / num_states) * O[j][obs[0]] for j in range(num_states)]

    # We iterate through all indices in the data
    for length in range(1, len_):   # length + 1 to avoid initial condition
        for state in range(num_states):
            # stores the probability of transitioning to 'state'
            p_trans = 0

            # probabilty of observing data in our given 'state'
            p_obs = O[state][obs[length]]

            # We iterate through all possible previous states, and update
            # p_trans accordingly.
            for prev_state in range(num_states):
                p_trans += prob[length - 1][prev_state] * A[prev_state][state]

            prob[length][state] = p_trans * p_obs  # update probability

        prob[length] = prob[length][:]  # copies by value

    # return total probability
    return sum(prob[len_ - 1])

if __name__ == '__main__':
    main()
