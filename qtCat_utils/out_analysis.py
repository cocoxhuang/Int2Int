'''
This module provides 
1. functions to decode the output from a model into a DataFrame of integers. 
The output is expected to be in a specific format from Int2Int, and the functions will convert it into a more usable form.
''' 
import pandas as pd

def decode_integer_array(encoded_str):
    '''
    decode from Int2Int token strings to regular strings
    e.g.
    from "V5 + 1 + 0 + 1 + 1 + 0 + 0"
    to "1,0,1,1,0,0"
    '''
    # Split the string by spaces and filter out empty strings
    tokens = [token for token in encoded_str.split() if token not in ('+', '-', '<eos>')]
    # also ingnore 'V22' or any other prefix
    tokens = [token for token in tokens if not token.startswith('V')]
    # Join the integers into a single string seperated by commas
    return ','.join(tokens)

def decode_output(output_str):
    '''
    Decode a line of output string from Int2Int.
    The output string is expected to be in the format:
    "V5 + 1 + 0 + 1 + 1 + 0 + 0 \t V5 1 + 0 + 1 + 1 + 0 + 0 \t V5 1 + 0 + 1 + 1 + 0 + 0"
    and will be converted to:
    "1,0,1,1,0,0 \t 1,0,1,1,0,0 \t 1,0,1,1,0,0"
    '''
    # Split the output string by tab characters
    parts = output_str.strip().split('\t')
    
    # Decode each part
    decoded_parts = [decode_integer_array(part) for part in parts]
    
    return decoded_parts

def decode_output_to_dataframe(int2int_output_path):
    '''
    Decode the output string from the model into a DataFrame with columns for input, output, and target.
    Return:
        A DataFrame with columns ['input', 'predict', 'target'] where each column contains the decoded integer strings.
        e.g.
        input, predict, target
        1,0,1,1,0,0, 1,0,1,1,0,0, 1,0,1,1,0,0
        ..., ..., ...
    '''
    decoded_df = []

    with open(int2int_output_path, 'r') as f:
        int2int_output_file = f.read()

    for line in int2int_output_file.strip().split('\n'):
        decoded_parts = decode_output(line)
        decoded_df.append(decoded_parts)
    
    df = pd.DataFrame(decoded_df, columns=['input', 'predict', 'target'])    
    return df