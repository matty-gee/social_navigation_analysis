import numpy as np
import sklearn as sk
from sklearn import metrics
import itertools, functools
import pandas as pd
import re

#--------------------------------------------------------------------------------------------
# checking 
#--------------------------------------------------------------------------------------------


def all_equal(iterable):
    ''' check if all items in a list are identical '''
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)


def all_equal_arrays(L):
    ''' check if difference in List L along axis of stacking == 0  '''
    return (np.diff(np.vstack(L).reshape(len(L), -1), axis=0) == 0).all()


def is_numeric(input_str):
    ''' check if all characters in a string are numeric '''
    return all(char.isdigit() for char in input_str)


def is_finite(array):
    return array[np.isfinite(array)]


def get_unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


#--------------------------------------------------------------------------------------------
# text parsing
#--------------------------------------------------------------------------------------------


def remove_multiple_strings(string, replace_list):
  for string_ in replace_list:
    string = string.replace(string_, '')
  return string


def remove_nontext(string):
    return re.sub(r'[^a-zA-Z]', '', string)


def remove_nonnumeric(string):
    return re.sub(r'[^0-9]', '', string)


def get_strings_matching_pattern(strings, pattern, verbose=0):
    if '*' in pattern:
        if verbose: print('Replacing wildcard * with regex .+')
        pattern = pattern.replace('*','.+')
    regex = re.compile(pattern)
    return [s for s in strings if re.match(pattern, s)]


def get_strings_matching_substrings(strings, substrings):
    ''' return strings in list that partially match any substring '''
    matches = [any(ss in s for ss in substrings) for s in strings]
    return list(np.array(strings)[matches])


def substring_in_strings(substring, strings):
    return substring in '\t'.join(strings)


#--------------------------------------------------------------------------------------------
# neutral character
#--------------------------------------------------------------------------------------------


def remove_neutrals(arr):
    ''' remove neutral charactr trials from array (trials 15, 16, 36) '''
    return np.delete(arr, np.array([14, 15, 35]), axis=0)


def add_neutrals(arr, add=[0, 0]):
    ''' add values to the neutral character trial positions (trials 15, 16, 36) '''
    neu_arr = arr.copy()
    for row in [14, 15, 35]: # ascending order, to ensure no problems w/ shifting the array
        neu_arr = np.insert(neu_arr, row, add, axis=0)
    return neu_arr


#--------------------------------------------------------------------------------------------
# math
#--------------------------------------------------------------------------------------------


def combos(arr, k=2):
    """ 
        arr: np.array to get combos from
        r: num of combos
    """
    return list(itertools.combinations(arr, k))


def exponential_decay(init, decay, time_steps):
    # f(t) = i(1 - r) ** t
    return np.array([init * (1 - decay) ** step for step in range(time_steps)])


def linear_decay(init, decay, time_steps):
    # f(t) = i - r * t
    # dont let it decay past 0
    return np.array([max((0, init - decay * step)) for step in range(time_steps)])


def _coincident_vectors(u, v):
    ''' Checks if vectors (u & v) are the same or scalar multiples of each other'''
    return np.dot(u, v) * np.dot(u, v) == np.dot(u, u) * np.dot(v, v)


def angle_between_vectors(u, v, direction=None, verbose=False):
    '''
        Compute elementwise angle between sets of vectors u & v
            
        uses np.arctan2(y,x) which computes counterclockwise angle [-π, π] between origin (0,0) and x, y
        clockwise v. counterclockwise: https://itecnote.com/tecnote/python-calculate-angle-clockwise-between-two-points/  
        included: ADD LINK

        TODO: make pairwise..?

        Arguments
        ---------
        u : array-like
            vector
        v : array-like
            another vector
        direction : None, True or False (optional, default=None)
            None == Included
            True == Clockwise 360
            False == Counterclockwise 360 
        verbose : bool (optional, default=False)
             
        Returns
        -------
        float 
            angle in radians 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    #     if U.shape != V.shape:
    #         if verbose: print(f'Different shape vectors: U={U.shape}, V={V.shape}. Assuming smaller is reference.')
    #         if len(U) < len(V): U = np.repeat(np.expand_dims(U, 0), len(V), axis=0)
    #         else:               V = np.repeat(np.expand_dims(V, 0), len(U), axis=0)
    #     rads = []
    #     for u, v in zip(U, V):            
    # if one of vectors is at origin, the angle is undefined but could be considered as orthogonal (90 degrees)
    if ((u==0).all()) or ((v==0).all()): 
        if verbose: print(u, v, 'at least 1 vector at origin; treating as orthogonal')
        rad = np.pi/2

    # if same vectors (or scalar multiples of each other) being compared, no angle between (0 degrees)
    # -- b/c 0-360 degrees, direction matters: make sure the signs are the same too
    elif (_coincident_vectors(u, v)) & all(np.sign(u) == np.sign(v)):
        if verbose: print(u, v, 'same vectors, no angle in between')
        rad = 0 # 0 degrees == 360 degrees == 2*pi radians 

    else:

        if direction is None: 

            # "included" angle from [0,180], [0, π] 
            rad = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
            # equivalent: np.arctan2(l2_norm(np.cross(u, v)), np.dot(u, v))

        elif direction is True: 

            # clockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*v[::-1]) - np.arctan2(*u[::-1])) % (2 * np.pi)  
        
        elif direction is False:

            # counterclockwise angle from [0,360], [0, 2π]
            # -- compute vector angles from origin & take difference, then convert to 360 degrees
            rad = (np.arctan2(*u[::-1]) - np.arctan2(*v[::-1])) % (2 * np.pi)
            
    return rad


def calculate_angle(U, V=None, direction=None, force_pairwise=False, verbose=False):
    '''
        Calculate angles between n-dim vectors 
        If V == None, calculate U pairwise
        Else, calculate elementwise
        
        TODO: more explanation; find more elegant ways to do this; also adapt other pairwise like functions to have structure

        Arguments
        ---------
        U : array-like
            shape (n_vectors, n_dims)
        V : array-like
            shape (n_vectors, n_dims)
        direction : optional (default=None)
            None : included 180
            False : counterclockwise 360 (wont give a symmetrical matrix)
            True : clockwise 360
            
        Returns
        -------
        numeric 
            pairwise angles in radians

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''
    
    # # testing (10-12-22)
    # U = np.random.randint(100, size=(4,2))
    # for V in [None, np.random.randint(100, size=U.shape), 
    # np.random.randint(100, size=(1, U.shape[1])), np.random.randint(100, size=(7, U.shape[1]))]:

    messages = []

    # check/fix shapes
    if U.ndim == 1: 
        U = np.expand_dims(U, 0)
        messages.append('Added a dimension to U')
        
    if V is not None:
        if V.ndim == 1: 
            V = np.expand_dims(V, 0)
            messages.append('Added a dimension to V')

            
    # determine output shape 
    # - 1 set of vectors: pw, square, symm
    if V is None: 
        default = 'pairwise'
        V = U 
        
    # - 2 vectors of same shape
    # -- pw, square, non-symm
    # -- ew, vector
    elif U.shape == V.shape: 
        default = 'elementwise' 

    # - 2 vectors, 1 w/ length==1 & is reference
    # -- pw, vector shape (1,u)
    elif (U.shape[0] > 1) & (V.shape[0] == 1): 
        V = np.repeat(V, len(U), 0) 
        default = 'reference'  
    
    # -- pw, vector shape (v,1)
    elif (U.shape[0] == 1) & (V.shape[0] > 1): 
        U = np.repeat(U, len(V), 0) 
        default = 'reference' 
        
    # - 2 vectors, different lengths
    # -- pw, rectangle 
    else: 
        default = 'pairwise' 
        
    messages.append(f'Calculated {default}')
    
    
    # calculate angles
    radians = np.zeros((U.shape[0], V.shape[0]))
    for i in range(U.shape[0]):
        for j in range(V.shape[0]):
            radians[i, j] = angle_between_vectors(U[i,:], V[j,:], direction=direction)

            
    # output
    if default == 'pairwise': cols = 'U'
    else:                     cols = 'V'
    radians = pd.DataFrame(radians, index=[f'U{i+1:02d}' for i in range(len(U))], columns=[f'{cols}{i+1:02d}' for i in range(len(V))])

    if not force_pairwise:
        if default == 'reference':
            radians = radians.iloc[:,0].values
        elif default == 'elementwise':
            radians = np.diag(radians)
    if verbose: [print(m) for m in messages]
        
    return radians


def cosine_distance(u, v=None):
    ''' 
        cosine distance of (u, v) = 1 - (dot(u,v) / dot(l2_norm(u), l2_norm(v)))
        returns similarity measure [0,2]
    '''
    return sk.metrics.pairwise_distances(u, v, metric='cosine')


def cosine_similarity(u, v=None):
    ''' 
        cosine similarity of (u, v) = dot(u,v) / dot(l2_norm(u), l2_norm(v))
        returns similarity measure [-1,1], equivalent to [0 degrees,18 0degrees]
        maybe issue: small angles tend to get very similar values(https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance)

    '''
    return 1 - sk.metrics.pairwise_distances(u, v, metric='cosine')


def angular_distance(u, v=None):
    ''' 
        angular distance of (u, v) = arccos(cosine_similarity(u, v)) / pi
        returns dissimilarity measure [0,2]
    '''
    return np.arccos(cosine_similarity(u, v))/np.pi


def angular_similarity(u, v=None):
    ''' 
        angular similarity between two vectors = 1 - (arccos(cosine_similarity(u, v)) / pi)
        returns similarity measure [-1,1]
    '''
    return 1 - (np.arccos(cosine_similarity(u, v))/np.pi)


#--------------------------------------------------------------------------------------------
# list & array manipulation
# TODO: simplify these into a smaller set of more robust functions
#--------------------------------------------------------------------------------------------


def flatten_nested_lists(L):
    return [i for sL in L for i in sL]


def bootstrap_matrix(matrix, random_state=None):
    ''' shuffles similarity matrix based on recommendation by Chen et al., 2016 '''
    s = skl.utils.check_random_state(random_state)
    n = matrix.shape[0]
    bs = sorted(s.choice(np.arange(n), size=n, replace=True))
    return matrix[bs, :][:, bs]


def digitize_matrix(matrix, n_bins=10): 
    '''
        Digitize an input matrix to n bins (10 bins by default)
        [By Matthew Schafer, github: @matty-gee; 2020ish]
    '''
    matrix_bins = [np.percentile(np.ravel(matrix), 100/n_bins * i) for i in range(n_bins)] # compute the bins 
    matrix_vec_digitized = np.digitize(np.ravel(matrix), bins = matrix_bins) * (100 // n_bins) # compute the vector digitized value 
    matrix_digitized = np.reshape(matrix_vec_digitized, np.shape(matrix)) # reshape to matrix
    matrix_digitized = (matrix_digitized + matrix_digitized.T) / 2  # force symmetry in the plot
    return matrix_digitized


def symmetrize_matrix(a):
    """
        Return a symmetrized version of NumPy array a.

        Values 0 are replaced by the array value at the symmetric
        position (with respect to the diagonal), i.e. if a_ij = 0,
        then the returned array a' is such that a'_ij = a_ji.

        Diagonal values are left untouched.

        a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
        for i != j.
    """
    return a + a.T - np.diag(a.diagonal())


def fill_in_upper_tri(sym_mat, diagonal=0):
    sym_mat = sym_mat + sym_mat.T - np.diag(np.diag(sym_mat))
    np.fill_diagonal(sym_mat, diagonal)
    return sym_mat


def symm_mat_labels_to_vec(labels, upper=True):
    '''
        lower or upper triangle label pairs from symm matrices, excl. diagonal
    '''
    n = len(labels)
    mat = pd.DataFrame(np.zeros((n, n)))
    for r in range(0, n):
        for c in range(r+1, n):
            mat.loc[r, c] = labels[r] + '_and_' + labels[c]
            mat.loc[c, r] = labels[c] + '_and_' + labels[r]
    if upper:
        ut  = pd.DataFrame(np.triu(mat, k=1)) # matches mat df
        vec = ut.values.flatten()
    else:
        lt  = pd.DataFrame(np.tril(mat, k=-1)) 
        vec = lt.values.flatten()
    return vec[vec!=0] 


def sort_symm_mat(mat, vec):
    
    ''' Sorts rows/columns of a symmetrical matrix according to a separate vector '''
    sort = vec.argsort()
    return mat[sort][:, sort]


def make_symm_mat_mask(orig_ixs, size=(63)):
    '''
        make a boolean mask for a symmetrical matrix of a certain size using the col/row ixs from original variables

        Arguments
        ---------
        ixs : list-like
            the ixs of the original variable to be set to True
        size : tuple (optional, default=(63))
            size of the symmetrical matrix

        Returns
        -------
        boolean mask 

        [By Matthew Schafer; github: @matty-gee; 2020ish]
    '''

    if type(size) is not int: 
        assert size[0] == size[1], 'need a symmetrical matrix; change size parameter'
        size = size[0]
    mask = np.ones((size, size))
    for ix in orig_ixs:
        mask[ix,:] = 0
        mask[:,ix] = 0
    return symm_mat_to_ut_vec(mask == 1)


def ut_vec_pw_dist(x, metric='euclidean'):
    x = np.array(x)
    if x.ndim == 1:  x = x.reshape(-1,1)
    return symm_mat_to_ut_vec(sk.metrics.pairwise_distances(x, metric=metric))

 
def symm_mat_to_ut_vec(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    vec_ut = mat[np.triu_indices(len(mat), k=1)]
    return vec_ut


def ut_mat_to_symm_mat(mat):
    ''' go from upper tri matrix to symmetrical matrix '''
    for i in range(0, np.shape(mat)[0]):
        for j in range(i, np.shape(mat)[1]):
            mat[j][i] = mat[i][j]
    return mat


def ut_vec_to_symm_mat(ut_vec):
    '''
        go from vectorized/flattened upper tri (to upper tri matrix) to symmetrical matrix
    '''
    ut_mat   = ut_vec_to_ut_mat(ut_vec)
    symm_mat = ut_mat_to_symm_mat(ut_mat)
    return symm_mat


def ut_vec_to_ut_mat(vec):
    '''
        go from vectorized/flattened upper tri to a upper tri matrix
            1. solve to get matrix size: matrix_len**2 - matrix_len - 2*vector_len = 0
            2. then populate upper tri of a m x m matrix with the vector elements 
    '''
    
    # solve quadratic equation to find size of matrix
    from math import sqrt
    a = 1; b = -1; c = -(2*len(vec))   
    d = (b**2) - (4*a*c) # discriminant
    roots = (-b-sqrt(d))/(2*a), (-b+sqrt(d))/(2*a) # find roots   
    if False in np.isreal(roots): # make sure roots are not complex
        raise Exception('Roots are complex') # dont know if this can even happen if not using cmath...
    else: 
        m = int([root for root in roots if root > 0][0]) # get positive root as matrix size
        
    # fill in the matrix 
    mat = np.zeros((m,m))
    vec = vec.tolist() # so can use vec.pop()
    c = 0  # excluding the diagonal...
    while c < m-1:
        r = c + 1
        while r < m: 
            mat[c,r] = vec[0]
            vec.pop(0)
            r += 1
        c += 1
    return mat


def remove_diag(arr):
    arr = arr.copy()
    np.fill_diagonal(arr, np.nan)
    return arr[~np.isnan(arr)].reshape(arr.shape[0], arr.shape[1] - 1)


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------


def merge_dfs(df_list, on=None, how='inner'):
    return functools.reduce(lambda x, y: pd.merge(x, y, on=on, how=how), df_list)


def move_cols_to_front(df, cols):
    return df[cols + [c for c in df if c not in cols]]