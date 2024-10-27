from SFT_NPA_model.SFT_NPA_Train_Test import *
from SFT_NRMS_model.SFT_NRMS_Train_Test import *
from SFT_NAML_model.SFT_NAML_Train_Test import *
from SFT_MRNN_model.SFT_MRNN_Train_Test import *

from DataLoad_MIND import load_data
import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_data_mode', type=int, default=3)
    parser.add_argument('--news_data_mode', type=int, default=3)
    parser.add_argument('--mode', type=str, default='NRMS_Bert')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='./out/save_model/', help='模型保留位置')

    parser.add_argument('--user_num', type = int, default=15427)
    parser.add_argument('--user_clicked_num', type=int, default=50)
    parser.add_argument('--warm_user_num', type=int, default=9925, help='热用户数')
    parser.add_argument('--cold_user_num', type=int, default=5502, help='冷用户数')
    parser.add_argument('--news_num', type=int, default=35855, help='新闻总数')
    parser.add_argument('--warm_news_num', type=int, default=6174, help='冷新闻数')
    parser.add_argument('--cold_news_num', type=int, default=29681, help='冷新闻数')
    parser.add_argument('--category_num', type=int, default=18, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=251, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=40300, help='单词总数')
    parser.add_argument('--news_entity_num', type=int, default=21343, help='新闻实体特征个数')
    parser.add_argument('--total_entity_num', type=int, default=111979, help='总实体特征个数')
    parser.add_argument('--total_relation_num', type=int, default=405, help='总关系特征个数')
    parser.add_argument('--news_entity_size', type=int, default=20, help='单个新闻最大实体个数')
    parser.add_argument('--title_word_size', type=int, default=39, help='每个新闻标题中的单词数量')
    parser.add_argument('--entity_neigh_num', type=int, default=5, help='邻居节点个数')

    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--embedding_dim', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--title_embedding_dim', type=int, default=400, help='新闻初始向量维数')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_embedding_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    
    return parser.parse_args()

def main(path, device):
    args = parse_args()
    data = load_data(args,path)
    if args.mode == "SFT_NPA":
        model = SFT_NPA_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "SFT_NRMS":
        model = SFT_NRMS_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "SFT_NAML":
        model = SFT_NAML_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "SFT_MRNN":
        model = SFT_MRNN_Train_Test(args, data, device)
        model.Train()
        model.Test()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
