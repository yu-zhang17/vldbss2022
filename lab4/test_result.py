import pytest
import json

class Test_simple():
    def test_cost(self):
        json_file = 'lab1/eval/results.json'
        with open(json_file, 'r') as f:
            data = json.load(f)
        act_cost = data.get('avi_cost')
        est_cost = data.get('est_cost')
        act_cardinality = data.get('act_cardinality')
        est_cardinality = data.get('est_cardinality')
        
        assert(len(act_cost) == len(est_cost))
        assert(len(act_cost) == len(act_cardinality))
        assert(len(act_cost) == len(est_cardinality))
        for v in est_cost:
            assert(v > 0)

if __name__ == '__main__':
    pytest.main(['test_result.py'])