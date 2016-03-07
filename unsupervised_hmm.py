import numpy as np
import re
import string

def main():
    obs = []
    obs_seqs = []

    SONNET_LINE_COUNT = 14
    num_poems = 154

    with open('./data/shakespeare.txt', 'r') as f:
        line = 'BEGIN'

        while line:
            title = int(f.readline().rstrip().split(' ')[-1]) # Skip Sonnet Titles
            line_count = 0

            limit = SONNET_LINE_COUNT

            # Exceptions
            if title > num_poems:
                break
            if title == 99: # Sonnet 99 consists of 15 lines
                limit = SONNET_LINE_COUNT + 1
            if title == 126: # Sonnet 125 consists of 12 lines
                limit = SONNET_LINE_COUNT - 2

            while line_count < limit:
                line = f.readline().rstrip()
                line = "".join(c for c in line if c not in string.punctuation)

                words = line.split(' ')
                words.append('\n') # Keep track of end of line

                obs += words
                obs_seqs.append(words)

                line_count += 1
            line = f.readline() # Skip Sonnet End

    obs = list(set(obs))
    obs_map = {obs[i] : i for i in range(len(obs))}
    num_states = 5

    A, O = baum_welch(num_states, len(obs), [[obs_map[a] for a in obs_seq] for obs_seq in obs_seqs])

    A_str = latex_matrix(A)
    O_str = latex_matrix(O)
    with open('hmm_out.txt', 'w') as f:
        f.write(A_str)
        f.write(O_str)

def fb_alg(A_mat, O_mat, observ):
    # set up
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

        denom = np.sum(bw[:,obs_ind-1])
        if denom:
            bw[:,obs_ind-1] = bw[:,obs_ind-1] / denom
        else:
            bw[:,obs_ind-1] = 0

    # combine it
    prob_mat = np.array(fw)*np.array(bw)
    return prob_mat, fw, bw

def baum_welch( num_states, num_obs, observ_seqs ):
    A_mat = np.random.rand(num_states, num_states)
    A_mat = A_mat / np.sum(A_mat,1)[:, np.newaxis]

    O_mat = np.random.rand(num_states, num_obs)
    O_mat = O_mat / np.sum(O_mat,1)[:, np.newaxis]

    while True:
        old_A = A_mat
        old_O = O_mat
        A_mat = np.ones( (num_states, num_states) )
        O_mat = np.ones( (num_states, num_obs) )

        # expectation step, forward and backward probabilities
        for i in xrange(len(observ_seqs)):
            observ = observ_seqs[i]
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

            for a_ind in xrange(num_states):
                for b_ind in xrange(num_states):

                    denom = np.sum(P[a_ind,:])
                    if denom:
                        A_mat[a_ind, b_ind] = np.sum( theta[a_ind, b_ind, :] ) / denom
                    else:
                        A_mat[a_ind, b_ind] = 0

            denom = np.sum(A_mat,1)[:, np.newaxis]
            for a_ind in xrange(num_states):
                if denom[a_ind]:
                    A_mat[a_ind] = A_mat[a_ind] / denom[a_ind]

            for a_ind in xrange(num_states):
                for o_ind in xrange(num_obs):
                    right_obs_ind = [i for i, x in enumerate(observ) if x == o_ind]

                    denom = np.sum( P[a_ind,1:])
                    if denom:
                        O_mat[a_ind, o_ind] = np.sum(P[a_ind,right_obs_ind]) / denom
                    else:
                        O_mat[a_ind, o_ind] = 0

            denom = np.sum(A_mat,1)[:, np.newaxis]
            for a_ind in xrange(num_states):
                if denom[a_ind]:
                    O_mat[a_ind] = O_mat[a_ind] / denom[a_ind]

        if np.linalg.norm(old_A-A_mat) < .00001 and np.linalg.norm(old_O-O_mat) < .00001:
            break

    return A_mat, O_mat

if __name__ == '__main__':
    main()
