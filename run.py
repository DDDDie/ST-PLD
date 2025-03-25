from configs.base import ParamManager
from dataloaders.base import DataManager
from backbones.base import ModelManager
from manager import STPLDManager
from utils.functions import save_results, set_seed
import logging
import argparse
import sys
import os
import datetime
import itertools
import torch

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='open_intent_discovery', help="Type for methods")

    parser.add_argument('--logger_name', type=str, default='Discovery', help="Logger name for open intent discovery.")

    parser.add_argument('--log_dir', type=str, default='logs', help="Logger directory.")

    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
    
    parser.add_argument("--num_workers", default=8, type=int, help="The number of known classes")

    parser.add_argument("--window_size", default=10, type=int, help="The size of the window")

    parser.add_argument("--bottom_k_ratio", default=0.01, type=float, help="The ratio of the bottom_k")
    
    parser.add_argument("--bottom_k_num", default=10, type=int, help="The num of the bottom_k")
    
    parser.add_argument("--top_k_ratio", default=0.5, type=float, help="The ratio of the top_k")
    
    parser.add_argument("--alpha", default=0.3, type=float, help="The alpha")
    
    parser.add_argument("--beta", default=0.5, type=float, help="The beta")
    
    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples in the training set")
    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")
   
    parser.add_argument("--wo_self", type=bool, default=True, help="Whether to pre-train the model")
    
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    
    parser.add_argument("--tune", action="store_true", help="Whether to tune the model")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for open intent detection")

    parser.add_argument("--backbone", type=str, default='bert', help="which backbone to use")
    
    parser.add_argument("--config_file_name", type=str, default='DeepAligned.py', help = "The name of the config file.")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='1', help="Select the GPU id")

    parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
    
    parser.add_argument("--data_dir", default = sys.path[0]+'/data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--output_dir", default= '../outputs', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")

    parser.add_argument("--save_results", action="store_true", help="save final results for open intent detection")

    args = parser.parse_args()

    return args

def set_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.dataset}_{args.backbone}_{args.known_cls_ratio}_{args.labeled_ratio}_{time}.log"
    args.logger_file_name =  f"{args.dataset}_{args.backbone}_{time}"
    print('logger_file_name', args.logger_file_name)
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def run(args, logger, debug_args = None): # 参数配置、日志记录器、可选的调试参数
    
    set_seed(args.seed)
    logger.info('Data and Model Preparation...')
    data = DataManager(args) # 数据管理器对象:用于处理数据
    model = ModelManager(args, data) # 模型管理器对象:用于处理模型
    
    method = STPLDManager(args, data, model, logger_name = args.logger_name) # 将参数、数据和模型传入方法对象中
    
    if args.train:
        
        logger.info('Training Begin...')
        method.train(args, data)
        logger.info('Training Finished...')

    logger.info('Testing begin...')
    outputs = method.test(args, data)
    logger.info('Testing finished...')

    if args.save_results:
        logger.info('Results saved in %s', str(os.path.join(args.result_dir, args.results_file_name)))
        save_results(args, outputs, debug_args=debug_args)

if __name__ == '__main__':
    
    sys.path.append('.')
    
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger = set_logger(args)
    
    logger.info('Open Intent Discovery Begin...')
    logger.info('Parameters Initialization...')
    param = ParamManager(args)
    args = param.args

    logger.debug("="*30+" Params "+"="*30) # 由等号组成的分隔线
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}") # 遍历 args 字典中的所有键
    logger.debug("="*30+" End Params "+"="*30)

    if args.tune:
        logger.info('Tuning begins...') # 进行参数调优：遍历不同参数组合并运行相应的操作
        debug_args = {}

        for k,v in args.items():
            if isinstance(v, list):
                debug_args[k] = v

        logger.info("***** Tuning parameters: *****")
        for key in debug_args.keys():
            logger.info("  %s = %s", key, str(debug_args[key]))
            
        for result in itertools.product(*debug_args.values()):
            for i, key in enumerate(debug_args.keys()):
                args[key] = result[i]         
            
            run(args, logger, debug_args=debug_args)

    else:
        run(args, logger) # 直接运行

