import numpy as np
import random
import re
import string

obs = []

filename = './data/shakespeare.txt'
SONNET_LINE_COUNT = 14

def parse_data(filename, num_poems=154):
    global obs
    obs = []
    obs_seqs = []
    rhyme_dict = {}

    with open(filename, 'r') as f:
        line = 'BEGIN'

        while line:
            title = int(f.readline().rstrip().split(' ')[-1]) # Skip Sonnet Titles
            line_count = 0

            limit = SONNET_LINE_COUNT

            if title > num_poems:
                break

            skip = False
            # Exceptions
            if title == 99: # Sonnet 99 consists of 15 lines
                limit = SONNET_LINE_COUNT + 1
                skip = True
            if title == 126: # Sonnet 125 consists of 12 lines
                limit = SONNET_LINE_COUNT - 2
                skip = True

            poem = []
            while line_count < limit:

                line = f.readline().rstrip().lstrip()

                if not skip:
                    # Removed all punctuation for now
                    line = "".join(c for c in line if c not in string.punctuation)

                    words = line.split(' ')
                    #words.append('\n') # Keep track of end of line
                    words = [w.lower() for w in words if w]

                    obs += words
                    obs_seqs.append(words)

                    poem.append(words)

                line_count += 1

            #obs_seqs.append(poem)
            if not skip:
               for a in range(0, int((limit - 2) / 4.0)):
                   idx = a * 4
                   #print idx
                   for i in range(idx, idx+2):
                       word1 = poem[i][-1]
                       word2 = poem[i+2][-1]

                       if rhyme_dict.get(word1):
                           rhyme_dict[word1].append(word2)
                       else:
                          rhyme_dict[word1] = [word2]

                       if rhyme_dict.get(word2):
                           rhyme_dict[word1].append(word1)
                       else:
                          rhyme_dict[word2] = [word1]

               word1 = poem[12][-1]
               word2 = poem[13][-1]

               if rhyme_dict.get(word1):
                   rhyme_dict[word1].append(word2)
               else:
                  rhyme_dict[word1] = [word2]

               if rhyme_dict.get(word2):
                   rhyme_dict[word1].append(word1)
               else:
                  rhyme_dict[word2] = [word1]

               poem = []
            line = f.readline() # Skip Sonnet End

    obs = list(set(obs))
    obs_map = {obs[i] : i for i in range(len(obs))}

    return obs_seqs, obs_map, rhyme_dict

def main():
    num_states = 10
    (obs_seqs, obs_map, rhyme_dict) = parse_data(filename, num_poems=154)

    A, O = baum_welch(num_states, len(obs), [[obs_map[a] for a in obs_seq] for obs_seq in obs_seqs])

    lines = generate_poem(num_states, O, A, rhyme_dict, obs_map, rhyme=True)
    for line in lines: print line

    # Viterbi, not necessary
    '''
    obs_seq = [obs_map[a] for a in obs_seqs[0]]
    seq = Viterbi(num_states, obs_seq, A, O)
    print obs_seqs[0]
    print seq

    obs_seq = [obs_map[a] for a in obs_seqs[1]]
    seq = Viterbi(num_states, obs_seq, A, O)
    print obs_seqs[1]
    print seq
    '''

# Poem Generation
def generate_poem(num_states, O, A, rhyme_dict=None, obs_map=None, rhyme=False):
    print rhyme_dict
    words_per_line = 7
    min_considered_prob = 1e-5

    gen_seqs = []

    if not rhyme:
        for line in range(SONNET_LINE_COUNT):
            curr_state = random.randint(0, num_states - 1)
            print curr_state
            gen_seq = []

            for word_idx in range(words_per_line):
                row = O[curr_state]

                pos_words = [(i,x) for i,x in enumerate(row) if x > min_considered_prob]

                if pos_words:
                    # Generate Bag of Words to choose from
                    pos_words.sort(key=lambda x: x[1])
                    min_prob = pos_words[0][1]
                    sample = []
                    for word in pos_words:
                        sample += [word[0]] * (word[1] / min_prob)

                    gen_seq.append(random.choice(sample))

                    # Generate Bag of States to transition to
                    pos_trans = [x for x in A[curr_state] if x > min_considered_prob]
                    min_prob = min(pos_trans)
                    sample = []
                    for i,x in enumerate(pos_trans):
                        sample += [i] * (x / min_prob)

                    curr_state = random.choice(sample)

            gen_seqs.append([obs[word] for word in gen_seq])

    elif rhyme:
        a_prev_end = ''
        b_prev_end = ''
        words_per_line = 6

        # End State
        for line in range(SONNET_LINE_COUNT):
            print line
            gen_seq = []

            if line in [0,1,4,5,8,9,12]:
                rhymes = [obs_map[a] for a in rhyme_dict.keys()]
            elif line in [2,3,6,7,10,11,13]:
                if line in [2,6,10]:
                    prev_end = a_prev_end
                else:
                    prev_end = b_prev_end

                rhymes = [obs_map[a] for a in rhyme_dict[prev_end]]

            pos_words = []
            while not pos_words:
                curr_state = random.randint(0, num_states - 1)
                pos_words = [(word, O[curr_state][word]) for word in rhymes if O[curr_state][word] > min_considered_prob]

            pos_words.sort(key=lambda x: x[1])
            min_prob = pos_words[0][1]
            sample = []
            for word in pos_words:
                sample += [word[0]] * (word[1] / min_prob)

            chosen_word = random.choice(sample)
            gen_seq.append(chosen_word)
            if line in [0,1,4,5,8,9,12]:
                if line in [0,4,8]:
                    a_prev_end = obs[chosen_word]
                else:
                    b_prev_end = obs[chosen_word]

            # Generate Bag of States to transition to
            pos_trans = [x for x in A[curr_state] if x > min_considered_prob]

            min_prob = min(pos_trans)
            sample = []
            for i,x in enumerate(pos_trans):
                sample += [i] * (x / min_prob)

            curr_state = random.choice(sample)

            for word_idx in range(words_per_line):
                row = O[curr_state]

                pos_words = [(i,x) for i,x in enumerate(row) if x > min_considered_prob]

                if pos_words:
                    # Generate Bag of Words to choose from
                    pos_words.sort(key=lambda x: x[1])
                    min_prob = pos_words[0][1]
                    sample = []
                    for word in pos_words:
                        sample += [word[0]] * (word[1] / min_prob)

                    chosen_word = random.choice(sample)

                    gen_seq.append(chosen_word)

                    # Generate Bag of States to transition to
                    pos_trans = [x for x in A[curr_state] if x > min_considered_prob]

                    min_prob = min(pos_trans)
                    sample = []
                    for i,x in enumerate(pos_trans):
                        sample += [i] * (x / min_prob)

                    curr_state = random.choice(sample)

            gen_seq.reverse()
            gen_seqs.append([obs[word] for word in gen_seq])
            print gen_seqs
            print

    return gen_seqs

def fb_alg(A_mat, O_mat, observ):
    k = len(observ)
    (n,m) = O_mat.shape
    prob_mat = np.zeros( (n,k) )
    fw = np.zeros( (n,k+1) )
    bw = np.zeros( (n,k+1) )

    # forward part
    fw[:, 0] = 1.0/n
    for obs_ind in xrange(k):
        f_row_vec = np.matrix(fw[:,obs_ind])
        fw[:, obs_ind+1] = f_row_vec * \
                           np.matrix(A_mat) * \
                           np.matrix(np.diag(O_mat[:,observ[obs_ind]]))

        # Normalize fw vector
        denom = np.sum(fw[:,obs_ind+1])
        if denom:
            fw[:,obs_ind+1] = fw[:,obs_ind+1] / denom
        else:
            fw[:,obs_ind+1] = 0

    # backward part
    bw[:,-1] = 1.0
    for obs_ind in xrange(k, 0, -1):
        b_col_vec = np.matrix(bw[:,obs_ind]).transpose()
        bw[:, obs_ind-1] = (np.matrix(A_mat) * \
                            np.matrix(np.diag(O_mat[:,observ[obs_ind-1]])) * \
                            b_col_vec).transpose()

        # Normalize bw vector
        denom = np.sum(bw[:,obs_ind-1])
        if denom:
            bw[:,obs_ind-1] = bw[:,obs_ind-1] / denom
        else:
            bw[:,obs_ind-1] = 0

    # combine it
    prob_mat = np.array(fw)*np.array(bw)
    return prob_mat, fw, bw

def baum_welch( num_states, num_obs, observ_seqs, convergence=.01 ):
    #A_mat = np.ones((num_states, num_states))
    A_mat = np.random.uniform(size=(num_states, num_states))
    A_mat = A_mat / np.sum(A_mat,1)[:, np.newaxis] # Normalize

    #O_mat = np.ones((num_states, num_obs))
    O_mat = np.random.uniform(size=(num_states, num_obs))
    O_mat = O_mat / np.sum(O_mat,1)[:, np.newaxis] # Normalize

    while True:
        old_A = A_mat
        old_O = O_mat

        A_mat_num = np.zeros((num_states, num_states))
        A_mat_denom = np.zeros((num_states, num_states))
        O_mat_num = np.zeros((num_states, num_obs))
        O_mat_denom = np.zeros((num_states, num_obs))

        # expectation step, forward and backward probabilities
        for i in xrange(len(observ_seqs)):
            observ = observ_seqs[i]
            print i, observ

            theta = np.zeros( (num_states, num_states, len(observ)) )
            P,F,B = fb_alg( old_A, old_O, observ)

            for a_ind in xrange(num_states):
                for b_ind in xrange(num_states):
                    for t_ind in xrange(len(observ)):
                        theta[a_ind,b_ind,t_ind] = \
                        F[a_ind,t_ind] * \
                        B[b_ind,t_ind+1] * \
                        old_A[a_ind,b_ind] * \
                        old_O[b_ind, observ[t_ind]]

            # Form A matrix
            for a_ind in xrange(num_states):
                for b_ind in xrange(num_states):
                    A_mat_denom[a_ind, b_ind] += float(np.sum(P[a_ind,:]))
                    A_mat_num[a_ind, b_ind] += float(np.sum( theta[a_ind, b_ind, :] ))

            # Form O matrix
            for a_ind in xrange(num_states):
                for o_ind in xrange(num_obs):
                    right_obs_ind = [i + 1 for i, x in enumerate(observ) if x == o_ind]

                    O_mat_denom[a_ind, o_ind] += float(np.sum( P[a_ind,1:]))
                    O_mat_num[a_ind, o_ind] += float(np.sum(P[a_ind,right_obs_ind]))
                    #print P[a_ind, right_obs_ind], a_ind, o_ind

        A_mat = A_mat_num / A_mat_denom
        A_mat[np.isnan(A_mat)] = 0

        O_mat = O_mat_num / O_mat_denom
        O_mat[np.isnan(O_mat)] = 0

        # Normalize A, O matrix
        O_denom = np.sum(O_mat,1)[:, np.newaxis]
        A_denom = np.sum(A_mat,1)[:, np.newaxis]
        for a_ind in xrange(num_states):
            if O_denom[a_ind]:
                O_mat[a_ind] = O_mat[a_ind] / O_denom[a_ind]
                O_mat[np.isnan(O_mat)] = 0
            if A_denom[a_ind]:
                A_mat[a_ind] = A_mat[a_ind] / A_denom[a_ind]
                A_mat[np.isnan(A_mat)] = 0

        print O_mat
        print
        for row in O_mat:
            ids = [(i,x) for i,x in enumerate(row) if x]
            ids.sort(key=lambda x: x[1])
            for idx in ids: print obs[idx[0]], idx
            print "END ROW"
            print
        print
        print A_mat
        print "NORM values"
        print "A_NORM", np.linalg.norm(old_A-A_mat)
        print "O_NORM", np.linalg.norm(old_O-O_mat)
        print

        if np.linalg.norm(old_A-A_mat) < convergence and np.linalg.norm(old_O-O_mat) < convergence:
            break

    return A_mat, O_mat

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
                seq[length][state] = seq[length - 1][max_state] + ' -> ' + str(state)

        prob[length] = prob[length][:]   # copies by value
        seq[length] = seq[length][:]

    max_ind = 0
    for i in range(states):  # find most-likely index of entire sequence
        if prob[len_ - 1][i] > prob[len_ - 1][max_ind]:
            max_ind = i

    # returns most-likely sequence
    return seq[len_ - 1][max_ind]

if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    main()
