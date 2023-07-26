from chains.NewFunctionChain import load_new_function_chain
from chains.TestFunctionChain import load_test_function_chain
from chains.utils.extract_code import extract_code_blocks

def add_new_skill():
    new_function_chain = load_new_function_chain()
    test_function_chain = load_test_function_chain()
    function_description = input("Enter a description for the function: ")
    new_func=new_function_chain.run(function_description)
    outputs = extract_code_blocks(new_func)
    function = outputs[0]
    function_header = function.split('def ')[1].split('\n')[0]
    tool = outputs[1]
    print("\n\n=== NEW FUNCTION ===")
    print(outputs[0] + "\n\n")
    print("=== NEW TOOL ===")
    print(outputs[1] + "\n\n")
    tests = test_function_chain.run(new_func)
    unique_filename = 'output_log_' + function_header.replace(' ', '') + '.txt'
    with open(unique_filename, 'w') as f:
        f.write(str(new_func))
        f.write(str(tests))