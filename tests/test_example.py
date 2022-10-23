import sys
sys.path.append('..')
sys.path.append('.')

from utils import get_all_h_param_comb, get_pred_test_list

def test_prediction_check_for_all_classes():
    gamma_list = [0.001,0.05,0.003,0.0002,0.00001]
    c_list = [0.1,0.8,0.3,2,8]
    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)
