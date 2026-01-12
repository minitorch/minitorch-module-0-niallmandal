import math
from minitorch import operators as ops

# is_close
assert ops.is_close(0.1, 0.105)
assert not ops.is_close(0.1, 0.2)
print('is_close checks OK')

# sigmoid properties
for x in [-100.0, -5.0, -1.0, 0.0, 1.0, 5.0, 100.0]:
    s = ops.sigmoid(x)
    assert 0.0 <= s <= 1.0
    # 1 - sigmoid(x) == sigmoid(-x)
    assert abs((1.0 - s) - ops.sigmoid(-x)) < 1e-6
print('sigmoid checks OK')

# log_back
x = 10.0
d = 2.0
assert abs(ops.log_back(x, d) - (d / (x + ops.EPS))) < 1e-12
print('log_back checks OK')

# relu and relu_back
assert ops.relu(5.0) == 5.0
assert ops.relu(-3.0) == 0.0
assert ops.relu_back(5.0, 3.0) == 3.0
assert ops.relu_back(-3.0, 3.0) == 0.0
print('relu checks OK')

print('ALL QUICK CHECKS PASS')
