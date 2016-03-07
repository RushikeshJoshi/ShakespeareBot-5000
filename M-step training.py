def main():
    raw_moods = []
    raw_genres = []

    with open('./data/ron.txt', 'r') as f:  # read in Ron's data
        for line in f.readlines():
            mood, genre = line.strip().split('\t')
            raw_moods.append(mood)
            raw_genres.append(genre)

    # maps moods to numbers
    moods = {'happy': 0, 'mellow': 1, 'sad': 2, 'angry': 3}

    # list of music genres
    # maps genres to numbers
    genres = {'rock': 0, 'pop': 1, 'house': 2, 'metal': 3, 'folk': 4,
              'blues': 5, 'dubstep': 6, 'jazz': 7, 'rap': 8, 'classical': 9}

    # numerical data of Ron's moods and music genres
    state_seq = [moods[x] for x in raw_moods]
    obs_seq = [genres[x] for x in raw_genres]

    A, O = MStep(moods, genres, state_seq, obs_seq)

    A_str = latex_matrix(A)
    O_str = latex_matrix(O)
    with open('1G.txt', 'w') as f:
        f.write(A_str)
        f.write(O_str)


def latex_matrix(matrix):
    matrix_str = '\\begin{bmatrix}\n'
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_str += str("{0:.3f}".format(matrix[i][j])) + ' & '
        matrix_str = matrix_str[:-2] + '\\\\\n'
    matrix_str += '\\end{bmatrix} \n'
    return matrix_str


# Uses a single Maximization step to compute A (state-transition) and
# O (observation) matrices. See Lecture 6 Slide 65.
def MStep(pos_states, pos_observations, state_seq, obs_seq):

    A = [[1. / len(pos_states) for i in pos_states] for j in pos_states]
    O = [[1. / len(pos_observations) for i in pos_observations]
         for j in pos_states]

    # create transition matrix
    for prev_state in pos_states:
        for state in pos_states:
            num = 0.0
            den = 0.0
            for j in range(len(state_seq) - 1):
                if (state_seq[j] == pos_states[prev_state]):
                    den += 1
                    if state_seq[j + 1] == pos_states[state]:
                        num += 1
            A[pos_states[state]][pos_states[prev_state]] = num / \
                den if den != 0 else 0

    # create observation matrix
    for state in pos_states:
        for obs in pos_observations:
            num = 0.0
            den = 0.0
            num = sum([int(obs_seq[j] == pos_observations[obs]) and
                       int(state_seq[j] == pos_states[state]) for j in range(len(state_seq))])
            den = sum([int(state_seq[j] == pos_states[state])
                       for j in range(len(state_seq))])

            O[pos_states[state]][pos_observations[obs]] = float(num) / den

    return A, O

if __name__ == '__main__':
    main()
