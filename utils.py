import code
def get_all_h_param_comb(params):
    h_param_comb = [{'gamma':g,'C':c} for g in params['gamma'] for c in params['C']]
    return h_param_comb

def get_pred_test_list():
    return pred_test_list()



