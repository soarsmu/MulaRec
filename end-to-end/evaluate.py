from models.dual_model import MulaRecDual
from models.single_code_model import MulaRecCode
from models.single_model import MulaRecAnt
from models.tri_model import MulaRecTri
import os
import argparse
from utils import set_logger
import logging

from datetime import datetime
import pytz

model_type_dict = {
    0: 'single_ant',
    1: 'single_code',
    2: 'dual_model',
    3: 'tri_model'
}

if __name__ == '__main__':     
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, required=True,
                        help="model type")
    parser.add_argument("--max_length", type=int, required=True,
                        help="max length")
    parser.add_argument("--load_model_path", type=str, required=True,
                        help="the fine-tuned model path")
    parser.add_argument("--test_filename", type=str, required=True,
                        help="the test file name")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory")    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)    
    set_logger('./log/eval_{}_{}.log'.format(model_type_dict[args.model_type], datetime.now(pytz.timezone('Asia/Singapore'))))
    logging.info(args)
    
    if args.model_type == 0:
        logging.info('***' * 10)
        logging.info('single annotation')
        
        model = MulaRecAnt(
            codebert_path = 'microsoft/codebert-base', 
            decoder_layers = 6, 
            fix_encoder = False, 
            beam_size = 5,
            max_source_length = args.max_length, 
            max_target_length = args.max_length, 
            load_model_path = args.load_model_path,
            l2_norm = True
        )
    elif args.model_type == 1:
        logging.info('***' * 10)
        logging.info('single code')
        
        model = MulaRecCode(
            codebert_path = 'microsoft/codebert-base', 
            decoder_layers = 6, 
            fix_encoder = False, 
            beam_size = 5,
            max_source_length = args.max_length, 
            max_target_length = args.max_length, 
            load_model_path = args.load_model_path,
            l2_norm = True
        )
    elif args.model_type == 2:
        logging.info('***' * 10)
        logging.info('dual model')
        
        model = MulaRecDual(
            codebert_path = 'microsoft/codebert-base', 
            decoder_layers = 6, 
            fix_encoder = False, 
            beam_size = 5,
            max_source_length = args.max_length, 
            max_target_length = args.max_length, 
            load_model_path = args.load_model_path,
            l2_norm = True,
            fusion = True
        )
    elif args.model_type == 3:        
        logging.info('***' * 10)
        logging.info('tri model')
        
        model = MulaRecTri(
            codebert_path = 'microsoft/codebert-base', 
            decoder_layers = 6, 
            fix_encoder = False, 
            beam_size = 5,
            max_source_length = args.max_length, 
            max_target_length = args.max_length, 
            load_model_path = args.load_model_path,
            l2_norm = True,
            fusion = True
        )

    # test model
    model.test(test_filename = args.test_filename, \
        test_batch_size = 128, \
        output_dir = args.output_dir
    )