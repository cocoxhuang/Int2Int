import argparse
import random
import os
import pandas as pd
from sage.all import DyckWords
from dyckwords import area_data, dinv_data, bounce_data
import time

def encode_integer(val, base=1000, digit_sep=" "):
    if val == 0:
        return '+ 0'
    sgn = '+' if val >= 0 else '-'
    val = abs(val)
    r = []
    while val > 0:
        r.append(str(val % base))
        val = val//base
    r.append(sgn)
    r.reverse()
    return digit_sep.join(r)    

def encode_integer_array(x, base=1000):
    return " ".join(encode_integer(int(z), base) for z in x)

def format_data(task, seq_len, input_file):
    # Read lines from input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # create a table for the first list and another table for the second list
    input = []
    output = []
    if task == "seq2seq":
        for line in lines:
            # remove the brackets and split by comma
            line = line.replace('[', '').replace(']', '').replace(',', '')
            # split by space
            line = line.split()
            # add the first list to table1 and the second list to table2
            input.append(encode_integer_array(line[0:seq_len]))
            output.append(encode_integer_array(line[seq_len:seq_len*2]))
        
        # put all in a file
        df = pd.DataFrame()
        df['x'] = [f"V{seq_len} " + x for x in input]
        df['y'] = [f"V{seq_len} " + x for x in output]
    if task in ["area", "dinv", "bounce"]:
        # a line is of format" [[1, 0, 1, 1, 0, 0], 1]
        for line in lines:
            # remove the brackets and split by comma
            line = line.replace('[', '').replace(']', '').replace(',', '')
            # split by space
            line = line.split()
            # add the first list to table1 and the second list to table2
            input.append(encode_integer_array(line[0:seq_len]))
            output.append(encode_integer_array(line[seq_len:seq_len+1]))

        # put all in a file
        df = pd.DataFrame()
        df['x'] = [f"V{seq_len} " + x for x in input]
        df['y'] = [f"V1 " + x for x in output]
        # df['y'] = output

    # output df as txt, sep by \t
    if input_file.endswith('.txt'):
        output_file = input_file[:-4] + "_formatted.txt"
    else:
        output_file = input_file + "_formatted.txt"
        
    df.to_csv(output_file, sep='\t', index=False, header=False)

    return output_file
    

def shuffle_and_split(task, input_file, seq_len, train_file, test_file, val_file, test_ratio, val_ratio):
    start = time.time()

    # first format input file
    input_file = format_data(task, seq_len, input_file)

    # Read lines from input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Shuffle lines
    random.shuffle(lines)

    # Split lines
    val_index = int(len(lines) * (1 - test_ratio))
    train_index = int(len(lines) * (1 - test_ratio - val_ratio))
    train_lines = lines[:train_index]
    val_lines = lines[train_index:val_index]
    test_lines = lines[val_index:]

    # Write train file
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    # write val file
    with open(val_file, 'w') as f:
        f.writelines(val_lines)

    # Write test file
    with open(test_file, 'w') as f:
        f.writelines(test_lines)

    # write out a recommned command file to run the model
    if task == "seq2seq":
        cmd = f"python train.py --operation data --data_types int[{seq_len}]:int[{seq_len}] --train_data {train_file} --eval_data {val_file},{test_file} --dump_path results --exp_name qtCat --exp_id 1 --epoch_size 10000 --eval_size 1000"
    if task in ["area", "dinv", "bounce"]:
        cmd = f"python train.py --operation data --data_types int[{seq_len}]:int[1] --train_data {train_file} --eval_data {val_file},{test_file} --dump_path results --exp_name qtCat --exp_id 1 --epoch_size 10000 --eval_size 1000"
    with open("qtCat_CMD.txt", "w") as f:
        f.write(cmd)
        f.write("\n\nYou might need to change parameters in the command such as  --exp_id, --epoch_size, --eval_size, -optimizer, --architecture, --n_enc_layers, etc.")
        f.write("\nCheck https://github.com/cocoxhuang/Int2Int/blob/main/train.py for more details.")


    print("")
    print(f"Done! {len(train_lines)} lines in {train_file}, {len(val_lines)} lines in {val_file}, {len(test_lines)} lines in {test_file}.")
    print("")
    print(f"Command to run the model saved at qtCat_CMD.txt: {cmd}\nYou might need to add additional parameters to the command.")

    print(f"Time taken to run shuffle_and_split: {time.time() - start} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a text file and split into train/test files.")
    parser.add_argument("--input_file", help="Path to the input text file")
    parser.add_argument("--seq_len", type=int, help="Length of the input and output sequence")
    parser.add_argument("--task", type=str, help="Prediction task name: area, dinv, bounce, seq2seq")
    parser.add_argument("--train_file", help="Path to output file for training data")
    parser.add_argument("--test_file", help="Path to output file for testing data")
    parser.add_argument("--val_file", help="Path to output file for validation data")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of data for validation (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of data for testing (default: 0.2)")
    args = parser.parse_args()

    gen_data_tasks = ["area", "dinv", "bounce"] # tasks that the script can generate data for

    assert args.task, "Task name is required."
    if args.task not in gen_data_tasks:
        assert args.input_file, "Input file is required."
    assert args.seq_len, "Sequence length is required."

    if args.input_file:
        assert args.input_file.endswith('.txt'), "Input file must be a .txt file."
        input_file = args.input_file
    else:
        assert args.task in gen_data_tasks, "We can only generate area and dinv data."
        input_file = f"./data/{args.task}_{args.seq_len // 2}.txt"
        # generate dyck words data
        assert (args.seq_len % 2) == 0, "Sequence length must be even as they are the length of Dyckwords."
        eval(args.task + f"_data({args.seq_len // 2}, '{input_file}')")

    # Set default train and test filenames if not provided
    train_file = args.train_file if args.train_file else f"{input_file[:-4]}_train.txt"
    test_file = args.test_file if args.test_file else f"{input_file[:-4]}_test.txt"
    val_file = args.val_file if args.val_file else f"{input_file[:-4]}_val.txt"

    shuffle_and_split(args.task, input_file, args.seq_len, train_file, test_file, val_file, args.test_ratio, args.val_ratio)