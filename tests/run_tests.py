#!/usr/bin/env python3

# discovers unit test files in /tests & runs them automatically

import unittest
import sys
def run_tests(out=sys.stderr, verbosity=2):
    # helps us capture unittest results in an out file
    testsuite = unittest.TestLoader().discover('.')
    unittest.TextTestRunner(out, verbosity=verbosity).run(testsuite)

if __name__ == '__main__':

    import os, sys
    from pathlib import Path
    import datetime

    # log the test
    user = os.getlogin()
    # out_log = str(Path(f'{str(Path(__file__).parent.absolute())}/unit_testing.txt'))
    out_log = 'unit_tests.txt'
    if os.path.exists(out_log): mode = 'a' # append
    else:                       mode = 'w' # write 
    with open(out_log, mode) as f:

        f.write(f'User: {user}\n')
        f.write(f'Test datetime: {datetime.datetime.now()}\n')
        f.write(f'Python version: {sys.version}\n\n')
        run_tests(f) # run & output test results
        f.write('\n......................................................................\n')
        f.write('......................................................................\n\n')
        