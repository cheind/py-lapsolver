import os
import numpy as np
import lapsolver as lap
import glob

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def test_files_for_dense(): 
    files = glob.glob(os.path.join(DATA_DIR, 'dense', '*.npz'))
    print(DATA_DIR)
    assert len(files) > 0
    for f in files:
        data = np.load(f)
        rids, cids = lap.solve_dense(data['costs'])

        assert data['costs'][rids, cids].sum() == data['total_cost']
