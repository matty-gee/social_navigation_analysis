import unittest
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd
 
# my modules
curr_dir = str(Path(__file__).parent.absolute())
sys.path.append(str(Path(f'{curr_dir}/../social_navigation_analysis')))
from preprocess import ComputeBehavior2
from test_utils import *

# test_subject_fname = f'{curr_dir}/../data/example_files/snt_18001.xslx'
test_subject_fname = '/Users/matty_gee/Dropbox/Projects/social_navigation_analysis/data/example_files/snt_18001.xlsx'
# import unittest
# sys.path.insert(0, f'{user}/Dropbox/Projects/social_navigation_analysis/tests')
# from test_utils import *
# test_subject_fname = '/Users/matthew/Dropbox/Projects/social_navigation_analysis/data/example_files/snt_18001.xlsx'

class TestBehavior(unittest.TestCase):
    ''' 
        inherits unittest.TestCase 
        any method with test_ in prefix will be considered a test    
    '''

    iters      = 5
    n_trials   = 12
    almost_tol = 5 # maybe limited precision

    #------------------
    # input/output
    #------------------
    
    def test_check_input(self):
        exp_shapes = [(2,2), (3,2)]
        to_check = np.array([(3,2),[3,2]])
        bool = ComputeBehavior2.check_input(to_check, exp_shapes)
        self.assertEqual(True, bool, 'Shape checker is off')

    # def test_run_output_shape(self):
    #     compute = ComputeBehavior2(file=random_behavior_data())
    #     compute.run()
    #     self.assertEqual(compute.dfs['current_constant_actual']['character_coords'].shape[0], 63, 'There are not 63 rows in the outputted dataframe')
    #     # self.assertEqual(self.compute.behavior.shape[1], 15*num_models, f'There are not {} columns in the outputted dataframe')

    #------------------------------------
    # decision weighting etc
    #------------------------------------
    
    def test_get_decisions_current(self):
        decisions_raw = random_behavior_data()['decision'].values.astype(int)
        decisions = ComputeBehavior2.get_decisions(decisions_raw, which='current')
        self.assertListEqual(decisions_raw.tolist(), decisions.tolist(), 'Current decisions off')

    def test_get_previous_decisions(self):
        decisions      = fake_decisions_2d(n_trials=self.n_trials)
        prev_decisions = ComputeBehavior2.get_decisions(decisions, which='previous', shift_by=1)
        self.assertEqual(prev_decisions.shape, (self.n_trials,2))
        self.assertListEqual(prev_decisions[0,:].tolist(), [0,0], 'Previous decisions in row 0 do not equal [0,0]')
        self.assertListEqual(prev_decisions[1:].tolist(), decisions[0:-1].tolist(), 'Previous decisions in row 1:end do not equal decisions row 0:end-1')

    def test_weight_decisions_constant(self):
        decisions = random_behavior_data()['decision'].values.astype(float)
        decisions_weighted = ComputeBehavior2.weight_decisions(decisions, weights='constant')[0]
        self.assertListEqual(decisions.tolist(), decisions_weighted.tolist(), 'Constant weighted decisions are off')

    # def test_weight_decisions_exponential(self): 

    # def test_weight_decisions_linear(self): 

    def test_get_coords_actual(self):
        decisions = fake_decisions_2d(n_trials=self.n_trials)
        coords2   = ComputeBehavior2.get_coords(decisions, which='actual')
        self.assertEqual(coords2.shape, (self.n_trials, 2))
        self.assertListEqual(np.cumsum(decisions, axis=0).tolist(), coords2.tolist(), 'Coordinates do not equal the cumulative sum of the decisions')
        self.assertEqual(np.sum(decisions[:, 0]), coords2[-1, 0], 'The last affil coordinate is not what it should be')
        self.assertEqual(np.sum(decisions[:, 1]), coords2[-1, 1], 'The last power coordinate is not what it should be')

    def test_get_coords_counterfactual(self):
        decisions  = fake_decisions_2d(n_trials=self.n_trials)
        coords     = np.cumsum(decisions, axis=0)
        cf_coords  = coords-(2*decisions)
        cf_coords2 = ComputeBehavior2.get_coords(decisions, which='counterfactual')
        self.assertEqual(cf_coords2.shape, (self.n_trials, 2))
        self.assertListEqual(cf_coords.tolist(), cf_coords2.tolist(), 'Countefactual decisions are wrongly computed')

    def test_get_coords_demeaning(self):
        coords2 = ComputeBehavior2.get_coords(fake_decisions_2d(n_trials=self.n_trials), which='actual', demean=True)
        self.assertAlmostEqual(float(np.mean(coords2, axis=0)[0]), 0, self.almost_tol, 'The demeaning isnt accurate within 5 decimal places')
        coords2 = ComputeBehavior2.get_coords(fake_decisions_2d(n_trials=self.n_trials), which='counterfactual', demean=True)
        self.assertAlmostEqual(float(np.mean(coords2, axis=0)[0]), 0, self.almost_tol, 'The demeaning isnt accurate within 5 decimal places for counterfactual')
    
    #------------------------------------
    #------------------------------------

    def test_cumulative_mean(self):

        decisions = fake_decisions_2d(n_trials=self.n_trials)
        resp_mask = decisions != 0
        cum_sum   = np.cumsum(decisions, axis=0)
        cum_count = np.cumsum(resp_mask, axis=0)
        mean      = np.mean(decisions, axis=0)

        # with resp_mask arg
        cum_mean  = ComputeBehavior2.calc_cumulative_mean(decisions, resp_mask)
        self.assertAlmostEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], self.almost_tol, 'Cumulative mean for affiliation is off')
        self.assertAlmostEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], self.almost_tol, 'Cumulative mean for power is off')

        self.assertAlmostEqual(cum_mean[-1,0], mean[0], self.almost_tol, 'Cumulative mean for affiliation is off: last value does not equal overall mean')
        self.assertAlmostEqual(cum_mean[-1,1], mean[1], self.almost_tol, 'Cumulative mean for power is off: last value does not equal overall mean')
    
        # without resp_mask arg
        cum_mean  = ComputeBehavior2.calc_cumulative_mean(decisions, resp_mask=None)
        self.assertAlmostEqual(cum_mean[-1,0], cum_sum[-1,0] / cum_count[-1,0], self.almost_tol, 'Cumulative mean for affiliation is off')
        self.assertAlmostEqual(cum_mean[-1,1], cum_sum[-1,1] / cum_count[-1,1], self.almost_tol, 'Cumulative mean for power is off')

    def test_quadrant_overlap_sum1_and_correct_quad(self):

        q1_coords = np.array([[4,4], [-1,4], [-1,-1], [4,-1]])
        q2_coords = q1_coords - np.repeat([[3,0]], 4, axis=0) # x-3
        q3_coords = q2_coords - np.repeat([[0,3]], 4, axis=0) # y-3
        q4_coords = q3_coords + np.repeat([[3,0]], 4, axis=0) # x+3

        for q, qC in enumerate([q1_coords, q2_coords, q3_coords, q4_coords]):

            overlap = ComputeBehavior2.calc_quadrant_overlap(qC)
            summed = np.sum(overlap)
            self.assertAlmostEqual(summed, 1, self.almost_tol, f'The overlap sum {summed}!=1')
            
            max_quad = np.where(overlap == np.max(overlap))[0][0]
            self.assertAlmostEqual(max_quad, q, self.almost_tol, f'The quarant with the max value {max_quad}!={q}')

    def test_centroids(self):
        # not sure what else to tes there
        decisions = fake_decisions_2d(n_trials=self.n_trials)
        coords    = np.cumsum(decisions, 0)
        centroids = ComputeBehavior2.calc_centroid(coords)

        self.assertGreater(np.max(coords, 0)[0], centroids[0][0])
        self.assertGreater(np.max(coords, 0)[1], centroids[0][1])

    def test_real_data(self):
        
        subj_data = pd.read_excel(test_subject_fname)
        coords = []

        for c in np.unique(subj_data['char_role_num']): 
            ixs = np.where((subj_data['char_role_num']==c) == True)[0]
            crds = ComputeBehavior2.calc_coords(ixs, subj_data.iloc[ixs,:])

            if c in [1,2,3,4,5]:

                # coujtn trials
                self.assertEqual(len(ixs), 12, f'Number of trials are off for character {c}')

                # check means
                end_mean = pd.DataFrame(crds[['affil_mean', 'power_mean']])
                avg = pd.DataFrame(crds[['affil_coord', 'power_coord']])/6
                self.assertListEqual(avg.iloc[-1, :].values.tolist(), end_mean.iloc[-1, :].values.tolist(),
                                     'Mean affiliation & powert is off')

            coords.append(crds)
        df = pd.DataFrame(np.hstack(coords))
        df.sort_values(by='trial_index', inplace=True)

# if running in jupyter nb -> unittest.main(argv=['first-arg-is-ignored'], exit=False) 
if __name__ == '__main__':
    unittest.main()