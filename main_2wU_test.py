from MRNN_model.MRNN_Train_Test import *
from NRMS_model.NRMS_Train_Test import *
from NAML_model.NAML_Train_Test import *
from MNN4Rec_model.MNN4Rec_Train_Test import *
from LSTUR_model.LSTUR_Train_Test import *
from DKN_model.DKN_Train_Test import *
from GAN_exp1_model.GAN_exp1_Train_Test import *
from LightGCN_model.LightGCN_Train_Test import *
from LightGCN_VAE_model.LightGCN_VAE_Train_Test import *
from FM_model.FM_Train_Test import *
from MNN4Rec_update_model.MNN4Rec_update_Train_Test import *
from MetaKG_model.MetaKG_Train_Test import *
from NPA_model.NPA_Train_Test import *
from KIM_model.KIM_Train_Test import *
from FIM_model.FIM_Train_Test import *
from Zeroshot_NPA_model.Zeroshot_NPA_Train_Test import *
from Zeroshot_MRNN_model.Zeroshot_MRNN_Train_Test import *
from Zeroshot_NRMS_model.Zeroshot_NRMS_Train_Test import *
from Zeroshot_IDvae_model.Zeroshot_IDvae_Train_Test import *
from Zeroshot_base_model.Zeroshot_base_Train_Test import *
from Zeroshot_baseID_model.Zeroshot_baseID_Train_Test import *
from Zeroshot_GAN_MRNN_model.Zeroshot_GAN_MRNN_Train_Test import *
from Zeroshot_NAML_model.Zeroshot_NAML_Train_Test import *
from Zeroshot_GAN_MRNN_ATT_model.Zeroshot_GAN_MRNN_ATT_Train_Test import *
from Zeroshot_GAN_NPA_model.Zeroshot_GAN_NPA_Train_Test import *
from Zeroshot_GAN_NPA_ATT_model.Zeroshot_GAN_NPA_ATT_Train_Test import *
from Zeroshot_GAN_NRMS_model.Zeroshot_GAN_NRMS_Train_Test import *
from Zeroshot_GAN_NRMS_ATT_model.Zeroshot_GAN_NRMS_ATT_Train_Test import *
from Zeroshot_GAN_NAML_model.Zeroshot_GAN_NAML_Train_Test import *
from Zeroshot_GAN_NAML_ATT_model.Zeroshot_GAN_NAML_ATT_Train_Test import *

from Zeroshot_NPA_ATT_model.Zeroshot_NPA_ATT_Train_Test import *
from Zeroshot_NRMS_ATT_model.Zeroshot_NRMS_ATT_Train_Test import *
from Zeroshot_NAML_ATT_model.Zeroshot_NAML_ATT_Train_Test import *
from Zeroshot_MRNN_ATT_model.Zeroshot_MRNN_ATT_Train_Test import *

from Zeroshot_Contrastive_NPA_ATT_model.Zeroshot_Contrastive_NPA_ATT_Train_Test import *
from Zeroshot_Contrastive_NRMS_ATT_model.Zeroshot_Contrastive_NRMS_ATT_Train_Test import *
from Zeroshot_Contrastive_NAML_ATT_model.Zeroshot_Contrastive_NAML_ATT_Train_Test import *
from Zeroshot_Contrastive_MRNN_ATT_model.Zeroshot_Contrastive_MRNN_ATT_Train_Test import *

from Zeroshot_GAZ_NPA_ATT_model.Zeroshot_GAZ_NPA_ATT_Train_Test import *
from Zeroshot_GAZ_NRMS_ATT_model.Zeroshot_GAZ_NRMS_ATT_Train_Test import *
from Zeroshot_GAZ_NAML_ATT_model.Zeroshot_GAZ_NAML_ATT_Train_Test import *
from Zeroshot_GAZ_MRNN_ATT_model.Zeroshot_GAZ_MRNN_ATT_Train_Test import *

from Prompt_NRMS_model.Prompt_NRMS_Train_Test import *
from Prompt_NAML_model.Prompt_NAML_Train_Test import *
from Prompt_MRNN_model.Prompt_MRNN_Train_Test import *

from NRMS_Bert_model.NRMS_Bert_Train_Test import *

from DataLoad_MIND import load_data
import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_data_mode', type=int, default=3)
    parser.add_argument('--news_data_mode', type=int, default=3)
    parser.add_argument('--mode', type=str, default='Zeroshot_Contrastive_NAML_ATT')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='./out/save_model/', help='模型保留位置')

    parser.add_argument('--user_num', type=int, default=15427)
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

    # MRNN
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
    # DKN参数
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='卷积核个数')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2, 3, 4], help='窗口大小')
    parser.add_argument('--use_context', type=bool, default=None, help='自主题总数')
    # NAML参数
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='卷积核个数')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='丢弃概率')
    # LSTUR参数
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='卷积核个数')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='窗口大小')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='遮掩概率')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # MNN4Rec参数
    parser.add_argument('--topk_implicit', type=int, default=5, help='隐式邻居采样数')
    parser.add_argument('--topk_explicit', type=int, default=5, help='显式邻居采样数')
    parser.add_argument('--use_news_relation', type=bool, default=True, help='是否利用新闻关系')
    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=50, help='meta batch size')
    parser.add_argument('--n_hops', type=int, default=10, help='gcn hop')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--select_idx', type=bool, default=False, help='是否选择样本')
    # KIM参数
    parser.add_argument('--feature_dim', type=int, default=300, help='新闻特征维度')
    return parser.parse_args()

def main(path, device):
    args = parse_args()
    data = load_data(args,path)
    if args.mode == "MRNN":
        model = MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NRMS":
        model = NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NAML":
        model = NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MNN4Rec":
        model = MNN4Rec_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LSTUR":
        model = LSTUR_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "DKN":
        model = DKN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "GAN_exp1":
        model = GAN_exp1_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LightGCN":
        model = LightGCN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LightGCN_VAE":
        model = LightGCN_VAE_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "FM":
        model = FM_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MNN4Rec_update":
        model = MNN4Rec_update_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MetaKG":
        model = MetaKG_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NPA":
        model = NPA_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NPA":
        model = Zeroshot_NPA_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_MRNN":
        model = Zeroshot_MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NRMS":
        model = Zeroshot_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_IDvae":
        model = Zeroshot_IDvae_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_base":
        model = Zeroshot_base_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_baseID":
        model = Zeroshot_baseID_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NAML":
        model = Zeroshot_NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_MRNN":
        model = Zeroshot_GAN_MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_MRNN_ATT":
        model = Zeroshot_GAN_MRNN_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NAML":
        model = Zeroshot_GAN_NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NAML_ATT":
        model = Zeroshot_GAN_NAML_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NRMS":
        model = Zeroshot_GAN_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NRMS_ATT":
        model = Zeroshot_GAN_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NPA":
        model = Zeroshot_GAN_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NPA_ATT":
        model = Zeroshot_GAN_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "KIM":
        model = KIM_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "FIM":
        model = FIM_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NPA_ATT":
        model = Zeroshot_NPA_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NRMS_ATT":
        model = Zeroshot_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NAML_ATT":
        model = Zeroshot_NAML_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_MRNN_ATT":
        model = Zeroshot_MRNN_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_Contrastive_NPA_ATT":
        model = Zeroshot_Contrastive_NPA_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_Contrastive_NRMS_ATT":
        model = Zeroshot_Contrastive_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_Contrastive_NAML_ATT":
        model = Zeroshot_Contrastive_NAML_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_Contrastive_MRNN_ATT":
        model = Zeroshot_Contrastive_MRNN_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAZ_NPA_ATT":
        model = Zeroshot_GAZ_NPA_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAZ_NRMS_ATT":
        model = Zeroshot_GAZ_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAZ_NAML_ATT":
        model = Zeroshot_GAZ_NAML_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAZ_MRNN_ATT":
        model = Zeroshot_GAZ_MRNN_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Prompt_NRMS":
        model = Prompt_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Prompt_NAML":
        model = Prompt_NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Prompt_MRNN":
        model = Prompt_MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NRMS_Bert":
        news_info_df = pd.read_csv('./Data_MIND/df/news_info_df.csv')
        news_title = news_info_df['title'].tolist()
        news_title.append('padding')
        news_title = np.array(news_title)
        model = NRMS_Bert_Train_Test(args, data, news_title, device)
        model.Test_load()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
