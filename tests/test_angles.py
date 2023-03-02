#!/usr/bin/env python3

#------------------------------------------------
# test angle functions - MAKE MUCH FASTER
#------------------------------------------------

import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(f'{str(Path(__file__).parent.absolute())}/../social_navigation_task')))
import utils
import numpy as np

class TestAngles(unittest.TestCase):

    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''
    n_vectors = 5

    # test from the origin against a reference of [1,0]
    # TODO: test diff origins
    def test_00_angle_between_vectors_angles(self):
        coords  = np.array([[1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])
        degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315])[:,np.newaxis]
        degrees2 = np.array([np.rad2deg(utils.angle_between_vectors(crd[0:2], np.array([1,0]), direction=False)) for crd in coords])
        self.assertEqual(degrees.all(), degrees2.all())       

    # 1 set of 2d vectors w/ itself: pairwise
    def test_01_calculate_angle_1vector_output_shape(self):
        for r in np.random.randint(10, size=self.n_vectors):
            U = np.random.randint(10, size=(r, 2))
            res = utils.calculate_angle(U, V=None)
            self.assertEqual(res.shape, (r,r))

    # 2 sets of 2d vectors (of possible different lengths), forced to pairwise output
    def test_02_calculate_angle_2vectors_pairwise_output_shape(self):

        for _ in range(self.n_vectors):
            U = np.random.randint(10, size=(np.random.randint(1,10), 2))
            V = np.random.randint(10, size=(np.random.randint(1,10), 2))
            res = utils.calculate_angle(U, V, force_pairwise=True)
            if (U.shape[0] == 1) or (V.shape[0] == 1): 
                if U.shape[0] > V.shape[0]: exp_shape = (U.shape[0], U.shape[0]) 
                else:                       exp_shape = (V.shape[0], V.shape[0])
            else:
                exp_shape = (U.shape[0], V.shape[0])
            self.assertEqual(res.shape, exp_shape)

    # 2 sets of 2d vectors of same shape: elementwise & pairwise
    def test_03_calculate_angle_2vectors_output_shape(self):
        for n in np.random.randint(10, size=self.n_vectors): # 10 random integers
            for force_pairwise, exp_shape in {True:(n,n), False:(n,)}.items():
                U = np.random.randint(10, size=(n,2))
                V = np.random.randint(10, size=(n,2))
                res = utils.calculate_angle(U, V, force_pairwise=force_pairwise)
                self.assertEqual(res.shape, exp_shape)

    # same vectors flipped -> a transposed matrix
    def test_04_calculate_angle_order_inputs_output_shape(self):

        U = np.random.randint(10, size=(10,2))
        V = np.random.randint(10, size=(1,2))
        res1 = utils.calculate_angle(U, V, force_pairwise=True)
        res2 = utils.calculate_angle(V, U, force_pairwise=True)
        self.assertEqual(res1.T.values.all(), res2.T.values.all())

if __name__ == '__main__':
    unittest.main()