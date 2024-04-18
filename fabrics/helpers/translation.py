import re

def extract_result_dimension(file_content: str):
    lines = file_content.split('\n')
    for line in lines:
        if '->' in line:
            important_line = line
            res = re.findall('\((.*)\)', important_line.split('>')[1])
            if '[' in res[0]:
                return int(re.findall('\[(\d+)\]', res[0])[0])
            else:
                return 1


def extract_c_function(file_content: str):
    """Extracts the function starting with 'static int casadi_' from the input file."""

    # Extract the function starting with 'static int casadi_'
    match = re.search(r'static int casadi_[^\{]*\{(?:[^{}]*\{[^{}]*\})*[^{}]*\}', file_content)
    if match:
        return match.group(0)
    else:
        print("Error: Couldn't find the function starting with 'static int casadi_'")
        return None

def c2np(input_file: str, output_file: str):
    """Translates generated c-code to numpy function."""
    with open(input_file, 'r') as file:
        file_content = file.read()
    result_shape = extract_result_dimension(file_content)
    print(result_shape)
    c_code = extract_c_function(file_content)
    if c_code is None:
        return

    # Extract variable names and operations from C code
    variables = re.findall(r'arg\[\d\]\[(\d)\]', c_code)
    operations = re.findall(r'(?<=\=).*?(?=\;)', c_code)

    # Generate Python code
    python_code_lines = [
        "import numpy as np\n",
        "def casadi_f0_numpy(*args):",
        f"    res = np.zeros((1, {result_shape}, args[0].shape[1]))",
    ]

    # Translate each operation from C to Python


    for line in c_code.split('\n')[2:-2]:
        line = line.replace(';', '')
        variable_name = line.split('=')[0]
        if '?' in line:
            line = variable_name + " =" + line.split('?')[1].split(':')[0]
            line = line.replace('arg', 'args')
        if "!" in line:
            line = '  ' + line.split(' ')[-1]
        if 'casadi_sq' in line:
            line = line.split('=')[0] + ' = ' + re.findall('\((.*)\)', line)[0] + ' ** 2'
        if 'sqrt' in line:
            line = line.split('=')[0] + ' = ' + re.findall('\((.*)\)', line)[0] + ' ** 0.5'
        if 'casadi_sign' in line:
            line = line.split('=')[0] + ' = np.sign(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'exp' in line:
            line = line.split('=')[0] + ' = np.exp(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'tanh' in line:
            line = line.split('=')[0] + ' = np.tanh(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'fmax' in line:
            line = line.split('=')[0] + ' = np.fmax(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'cos' in line:
            line = line.split('=')[0] + ' = np.cos(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'sin' in line:
            line = line.split('=')[0] + ' = np.sin(' + re.findall('\((.*)\)', line)[0] + ')'
        if 'casadi_fabs' in line:
            line = line.split('=')[0] + ' = np.abs(' + re.findall('\((.*)\)', line)[0] + ')'
        line = line.replace('  ', '    ')




        python_code_lines.append(line)
    python_code_lines.append('    return np.transpose(res[0])')

    python_code = "\n".join(python_code_lines)



    # Write Python code to output file
    with open(output_file, 'w') as file:
        file.write(python_code)




