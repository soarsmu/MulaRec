"""
ex.: python calculate_bleu_score.py --reference codebert/saved_models_our_data/test_-1.gold --candidate codebert/saved_models_our_data/test_-1.output
python calculate_bleu_score.py --reference codebert/saved_models_our_data-10-epochs/test_-1.gold --candidate codebert/saved_models_our_data-10-epochs/test_-1.output
python calculate_bleu_score.py --reference codebert/saved_models_our_data-30-epochs/test_-1.gold --candidate codebert/saved_models_our_data-30-epochs/test_-1.output
"""

from nltk.translate.bleu_score import sentence_bleu
import argparse

weights = {
    '1': [1],
    '2': [1./2., 1./2.],
    '3': [1./3., 1./3., 1./3.],
    '4': [1./4., 1./4., 1./4., 1./4.]
}
 
def calculate_blue(weight_index):        

    reference = open(args.reference, 'r').readlines()
    candidate = open(args.candidate, 'r').readlines()

    if len(reference) != len(candidate):
        raise ValueError('The number of sentences in both files do not match.')

    score = 0.

    for i in range(len(reference)):
        score += sentence_bleu([
            reference[i].strip().lower().split()], 
            candidate[i].strip().lower().split(),
            weights = weights[weight_index]
        )

    score /= len(reference)
    print("The bleu score BLEU-{} is: {}".format(weight_index, str(score)))
    
def remove_line_num(file, new_file_name):
    with open(file) as f:
        lines = f.readlines()
    with open(new_file_name, 'w') as f:
        for line in lines:
            new_line = ' '.join(line.split()[1:])
            f.write(new_line)
            f.write('\n')
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--reference', type=str, default='summaries.txt', help='Reference File')
    argparser.add_argument('--candidate', type=str, default='candidates.txt', help='Candidate file')
    args = argparser.parse_args()    
    
    # remove_line_num('codebert/3-lines-bimodal/test_-1.gold', 'codebert/3-lines-bimodal/test.gold')
    # remove_line_num('codebert/3-lines-bimodal/test_-1.output', 'codebert/3-lines-bimodal/test.output')
    
    # remove_line_num('codebert/3-lines-code/test_-1.gold', 'codebert/3-lines-code/test.gold')
    # remove_line_num('codebert/3-lines-code/test_-1.output', 'codebert/3-lines-code/test.output')
    
    calculate_blue('1')
    calculate_blue('2')
    calculate_blue('3')
    calculate_blue('4')