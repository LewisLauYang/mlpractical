from mlp.tests import test_cosine_scheduler

check_functionality, functionality_correct, functionality_output, check_continue_from_checkpoint, continuation_correct, continuation_output = test_cosine_scheduler()

assert check_functionality == 1.0, (
'The scheduler functionality test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(functionality_correct, functionality_output, functionality_output-functionality_correct)
)

print("Scheduler Functionality Test Passed")

assert check_continue_from_checkpoint == 1.0, (
'The scheduler continue-from feature test failed'
'Correct output is \n\n{0}\n\n but returned output is \n\n{1}\n\n difference is \n\n{2}'
.format(continuation_correct, continuation_output, continuation_output-continuation_correct)
)

print("Scheduler Continue-from Test Passed")