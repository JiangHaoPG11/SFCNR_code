from SFT_NAML_model.SFT_NAML import SFT_NAML
from SFT_NAML_model.SFT_NAML_Trainer import Trainer
import torch

class SFT_NAML_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        SFT_NAML_model = SFT_NAML(args, entity_embedding, relation_embedding,
                                            news_entity_dict, entity_adj, relation_adj, news_title_word_index,
                                            word_embedding, news_category_index, news_subcategory_index, device).to(device)

        optimizer_Zeroshot_news = torch.optim.Adam(SFT_NAML_model.zeroshot_news_tower.parameters(), lr=0.0001)
        optimizer_Zeroshot_user = torch.optim.Adam(SFT_NAML_model.zeroshot_user_tower.parameters(), lr=0.0001)


        optimizer = torch.optim.Adam([{"params": SFT_NAML_model.news_encoder.parameters()},
                                          {"params": SFT_NAML_model.user_encoder.parameters()},
                                           ], lr=0.0001)
            
        # optimizer_base = torch.optim.Adam([{"params": SFT_NAML_model.news_encoder.parameters()},
        #                                    {"params": SFT_NAML_model.user_encoder.parameters()}
        #                                    ], lr=0.0001)
            
        optimizer_base = torch.optim.Adam([{"params": SFT_NAML_model.news_encoder.parameters()},
                                           {"params": SFT_NAML_model.user_encoder.parameters()},
                                           {"params": SFT_NAML_model.news_encoder.parameters()},
                                           {"params": SFT_NAML_model.user_encoder.parameters()},
                                           {"params": SFT_NAML_model.zeroshot_news_tower.parameters()},
                                           {"params": SFT_NAML_model.zeroshot_user_tower.parameters()}
                                           ], lr=0.0001)

        for para in SFT_NAML_model.named_parameters():
            print(para[0])
        self.trainer = Trainer(args, SFT_NAML_model, optimizer, optimizer_base, optimizer_Zeroshot_news, optimizer_Zeroshot_user, data)
        self.Zeroshot_model = SFT_NAML_model
        self.args = args

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def Test_load(self):
        self.Zeroshot_model.load_state_dict(torch.load(self.args.checkpoint_dir + 'checkpoint-' + self.args.mode + '-epochfinal.pth', map_location='cpu'))
        self.trainer.test()
