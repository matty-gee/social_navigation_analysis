import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(f'{str(Path(__file__).parent.absolute())}/../social_navigation_task')))
import preprocess as preprc
import info 
import numpy as np
import pandas as pd
import random

# want to make meaningful assertions about the state after calling method 

class TestBehavior(unittest.TestCase):
    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''
    # TODO: add shape, type, etc testing for inputs
    # TDOD: test other methods - eg compute_character,compute_acrosscharacters 

    def fake_decisions_2d(self, n_trials=12):
        return np.array([random.choice([-1,1]) for _ in range(n_trials*2)]).reshape(-2,2)

    def fake_data(self):      
        
        # make better to test what certain patterns would look like in the behavior
        dimension = np.array([['affil', 'affil', 'affil', 'power', 'affil', 'power', 'power',
                            'affil', 'affil', 'power', 'affil', 'power', 'power', 'power',
                            'neutral', 'neutral', 'affil', 'power', 'affil', 'power', 'power',
                            'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                            'power', 'affil', 'power', 'power', 'power', 'affil', 'affil',
                            'neutral', 'power', 'power', 'affil', 'affil', 'affil', 'affil',
                            'power', 'affil', 'affil', 'power', 'affil', 'affil', 'affil',
                            'affil', 'affil', 'power', 'power', 'power', 'affil', 'power',
                            'affil', 'affil', 'power', 'power', 'power', 'power', 'affil']]).reshape(-1,1)
        char_role_nums = np.array([[1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 4, 2, 1, 1, 9, 9, 1, 2, 4, 2, 4, 4,
                                    4, 4, 1, 2, 2, 2, 1, 5, 5, 3, 5, 3, 3, 9, 3, 3, 3, 3, 5, 5, 5, 5,
                                    5, 5, 2, 2, 3, 2, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3]]).reshape(-1,1)
        char_dec_nums = np.array([[1,2, 3, 4, 1, 2, 5, 6, 7,  3,  1,  4,  8,  9,  1,  2, 10,
                                    5,2, 6, 3, 4, 5, 6, 11,  7,  8,  9, 12,  1,  2,  1,  3,  2,
                                    3,3, 4, 5, 6, 7, 4, 5, 6,  7,  8,  9, 10, 11,  8, 12, 10,
                                    11,12, 7, 8, 9, 10, 11, 12,  9, 10, 11, 12]]).reshape(-1,1)
        # fake decisions
        button_press = np.array([random.choice([1,2 ]) for _ in range(63)]).reshape(-1,1)
        decisions    = np.array([random.choice([-1,1]) for _ in range(63)]).reshape(-1,1)
        fake_data    = pd.DataFrame(np.hstack([dimension, char_role_nums, char_dec_nums, button_press, decisions]), 
                                     columns=['dimension', 'char_role_num', 'char_decision_num', 'button_press', 'decision'])
        return fake_data
     
    iters    = 5
    n_trials = 12
    compute = preprc.ComputeBehavior(file=None)   

    #------------------
    # test input/output
    #------------------
    
    def test_xx_check_input_shape(self):
        exp_shapes = [(2,2), (3,2)]
        to_check   = np.array([(3,2),[3,2]])

        bool = self.compute.check_input_shape(to_check, exp_shapes)
        self.assertEqual(True, bool, 'Shape checker is off')

    def test_xx_run_output_shape(self):
        data    = self.fake_data()
        compute = preprc.ComputeBehavior(file=data)
        compute.run()
        self.assertEqual(compute.behavior.shape[0], 63, 'There are not 63 rows in the outputted dataframe')
        # self.assertEqual(self.compute.behavior.shape[1], 15*num_models, f'There are not {} columns in the outputted dataframe')

    #----------------------
    # test decision getting
    #----------------------

    def test_xx_current_decisions(self):
        for _ in range(self.iters):

            decisions  = self.fake_decisions_2d(n_trials=self.n_trials)
            decisions2 = self.compute.current_decisions(decisions=decisions)
            self.assertListEqual(decisions.tolist(),  decisions2.tolist(), 'Current decisions are off')

    def test_xx_previous_decisions(self):
        for _ in range(self.iters):

            decisions      = self.fake_decisions_2d(n_trials=self.n_trials)
            prev_decisions = self.compute.previous_decisions(decisions, shift_by=1)
            self.assertEqual(prev_decisions.shape, (self.n_trials,2))

            # previous_decisions/coordinates row 1 should = 0, 0
            self.assertListEqual(prev_decisions[0,:].tolist(), [0,0], 'Previous decisions in row 0 do not equal [0,0]')

            # previous_decisions/coordinates [1:end] = decisions/coordinates [0:end-1]
            self.assertListEqual(prev_decisions[1:].tolist(), decisions[0:-1].tolist(), 'Previous decisions in row 1:end do not equal decisions row 0:end-1')

    #-------------------------
    # test deccision weighting
    #-------------------------

    def test_xx_weight_decisions_constant(self):
        for _ in range(self.iters):
            decisions = self.fake_decisions_2d(n_trials=self.n_trials)
            decisions2 = self.compute.weight_decisions(decisions, decay='constant') # shouldnt change the decision values at all
            self.assertListEqual(decisions.tolist(), decisions2.tolist(), 'The constant weighting is off: raw decisions should equal constant weighted (weight=1) decisions')

    # def test_xx_weight_decisions_linear(self)
    # def test_xx_weight_decisions_exponential(self)
    
    #------------------------
    # test coordinate summing
    #------------------------

    def test_xx_actual_coords(self):
        for _ in range(self.iters):
            decisions = self.fake_decisions_2d(n_trials=self.n_trials)
            coords = np.cumsum(decisions, axis=0)
            coords2 = self.compute.actual_coords(decisions)
            self.assertListEqual(coords.tolist(), coords2.tolist(), 'Coordinates are off')
            self.assertEqual(np.sum(decisions[:, 0]), coords2[-1, 0], 'The last affil coordinate is not what it should be')
            self.assertEqual(np.sum(decisions[:, 1]), coords2[-1, 1], 'The last power coordinate is not what it should be')

    def test_xx_counterfactual_coords(self):
        for _ in range(self.iters):

            decisions = self.fake_decisions_2d(n_trials=self.n_trials)
            
            coords    = np.cumsum(decisions, axis=0)
            cf_coords = coords-(2*decisions)

            cf_coords2 = self.compute.counterfactual_coords(decisions)

            self.assertEqual(cf_coords2.shape, (self.n_trials, 2))
            self.assertListEqual(cf_coords.tolist(), cf_coords2.tolist(), 'Countefactual decisions are wrongly computed')

    def test_xx_cumulative_mean(self):

        decisions = self.fake_decisions_2d(n_trials=self.n_trials)
        resp_mask = decisions != 0
        cum_mean, (cum_sum, cum_count) = self.compute.cumulative_mean(decisions, resp_mask)
        self.assertEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], 'Cumulative mean for 1st dimension (ie, affiliation) is off')
        self.assertEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], 'Cumulative mean for 2nd dimension (ie, power) is off')

    #---------------------
    # test other functions
    #---------------------

    def test_xx_simulate_consistent_behavior(self):

        decisions = self.fake_decisions_2d(n_trials=self.n_trials)       
        incon_decs, con_decs, resp_mask = self.compute.simulate_consistent_decisions(decisions)

        self.assertEqual(incon_decs.shape, (self.n_trials, 2))
        self.assertEqual(con_decs.shape, (self.n_trials, 2))
        self.assertEqual(resp_mask.shape, (self.n_trials, 2))

        incon_coords = np.cumsum(incon_decs, axis=0)
        con_coords = np.cumsum(con_decs, axis=0)
        self.assertEqual(np.sum(incon_coords <= con_coords), self.n_trials*2, 'Some of the inconsistent pattern coordinates (min) are not equal or smaller to consistent pattern (max)')

    # def test_xx_cumulative_consistency(self):
    
    def test_xx_add_3rd_dimension_output_shape(self):
        for _ in range(10):
            U = np.random.randint(1,10, size=(np.random.randint(1,10),2)) # all the outputs should have this shape
            V = np.random.randint(1,10, size=(2,))
            ori = np.array([[0,0]])   
            U, V, ori = self.compute.add_3rd_dimension(U, V, ori)
            self.assertEqual(U.shape, V.shape, f'Adding in 3rd dimension made mismatched vector shapes {U.shape} != {V.shape}')  

    def test_xx_quadrant_overlap_sum1_and_correct_quad(self):

        q1_coords = np.array([[4,4], [-1,4], [-1,-1], [4,-1]])
        q2_coords = q1_coords - np.repeat([[3,0]], 4, axis=0) # x-3
        q3_coords = q2_coords - np.repeat([[0,3]], 4, axis=0) # y-3
        q4_coords = q3_coords + np.repeat([[3,0]], 4, axis=0) # x+3

        for q, qC in enumerate([q1_coords, q2_coords, q3_coords, q4_coords]):

            overlap = self.compute.quadrant_overlap(qC)

            summed = np.sum(overlap)
            self.assertEqual(summed, 1, f'The overlap sum {summed}!=1')

            max_quad = np.where(overlap == np.max(overlap))[0][0]
            self.assertEqual(max_quad, q, f'The quarant with the max value {max_quad}!={q}')
        
    # ori & U should have z-axis coordinates that increase sequentially; V should have a fixed z-axis coordinate

if __name__ == '__main__':
    unittest.main()