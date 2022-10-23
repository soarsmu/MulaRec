import random
random.seed(42)

import pandas as pd
from tqdm import tqdm
import os
    
def split_two_files_annotation(file, variant):
    df = pd.read_csv(file, usecols=['annotation', 'api_seq', 'target_api'])
    os.makedirs('data/3-lines/annotation-only', exist_ok=True)
    with open('data/3-lines/annotation-only/{}.annotation'.format(variant), 'w') as f1:
        with open('data/3-lines/annotation-only/{}.all.seq'.format(variant), 'w') as f2:
            with open('data/3-lines/annotation-only/{}.target.seq'.format(variant), 'w') as f3:
                for index, row in tqdm(df.iterrows()):
                    f1.write(row['annotation'].strip())
                    f1.write('\n')
                    f2.write(row['api_seq'])
                    f2.write('\n')
                    f3.write(row['target_api'])
                    f3.write('\n')

def split_two_files_code(file, variant):
    df = pd.read_csv(file, usecols=['source_code', 'target_api'])
    os.makedirs('data/3-lines/code-only', exist_ok=True)
    with open('data/3-lines/code-only/{}.code'.format(variant), 'w') as f1:
        with open('data/3-lines/code-only/{}.seq'.format(variant), 'w') as f2:
            for index, row in tqdm(df.iterrows()):
                f1.write(' '.join(row['source_code'].strip().split()))
                f1.write('\n')
                f2.write(row['target_api'])
                f2.write('\n')

def split_two_files_annotation_code(file, variant):
    df = pd.read_csv(file, usecols=['annotation', 'source_code', 'target_api'])

    os.makedirs('data/3-lines/bimodal', exist_ok=True)
    with open('data/3-lines/bimodal/{}.bimodal'.format(variant), 'w') as f1:
        with open('data/3-lines/bimodal/{}.seq'.format(variant), 'w') as f2:
            for index, row in tqdm(df.iterrows()):
                bimodal_info = row['annotation'] + ' ' + row['source_code']
                bimodal_info = ' '.join(bimodal_info.strip().split())
                f1.write(bimodal_info)
                f1.write('\n')
                f2.write(row['target_api'])
                f2.write('\n')

def get_avg_tokens(token_list, token_type):
    num_code_tokens = []
    for code_snippet in token_list:
        num_code_tokens.append(len(code_snippet.split()))
    print('average num of {} tokens: {}'.format(token_type, sum(num_code_tokens) / len(num_code_tokens)))
    
def data_statistics(file_name, variant):
    df = pd.read_csv(file_name)
    df = df.fillna("")
    code = df['source_code'].astype("string").tolist()
    ant = df['annotation'].astype("string").tolist()
    post = df['related_so_question'].astype("string").tolist()
    print('{}'.format(variant))
    get_avg_tokens(code, 'code')
    get_avg_tokens(ant, 'annotation')
    get_avg_tokens(post, 'similar post')
    
if __name__ == '__main__':
    # split_two_files_annotation('data/train_3_lines.csv', 'train')
    # split_two_files_annotation('data/validate_3_lines.csv', 'valid')
    # split_two_files_annotation('data/test_3_lines.csv', 'test')
    # split_two_files_code('data/train_3_lines.csv', 'train')
    # split_two_files_code('data/validate_3_lines.csv', 'valid')
    # split_two_files_code('data/test_3_lines_dedup.csv', 'test')

    # split_two_files_annotation_code('data/train_3_lines.csv', 'train')
    # split_two_files_annotation_code('data/validate_3_lines.csv', 'valid')
    split_two_files_annotation_code('data/test_3_lines_dedup.csv', 'test')
    # data_statistics('data/train_3_lines.csv', 'train')
    # data_statistics('data/validate_3_lines.csv', 'valid')
    # data_statistics('data/test_3_lines_dedup.csv', 'test')