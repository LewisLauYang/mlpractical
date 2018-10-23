from mlp.tests import test_adam_with_weight_decay

check_functionality, correct, outputs = test_adam_with_weight_decay()

assert check_functionality == 1.0, (
'The weight decay functionality test failed',
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(correct, outputs, outputs-correct)
)

print("Adam with Weight Decay Learning Rule Functionality Test Passed")