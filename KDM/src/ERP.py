import os
import argparse
from torch_geometric.loader import DataLoader
import torch
from utils import *
from model.classify_model import *
from datasets.graph_dataset import *
from add_role_to_events import *

def acc(preds, target):
    p = metrics.precision_score(target, preds, average="macro")
    r = metrics.recall_score(target, preds, average="macro")
    f1 = metrics.f1_score(target, preds, average="macro")

    p_i = metrics.precision_score(target, preds, average="micro")
    r_i = metrics.recall_score(target, preds, average="micro")
    f1_i = metrics.f1_score(target, preds, average="micro")
    print(f1, f1_i)
    matrix = metrics.confusion_matrix(target, preds)
    length = len(matrix)
    return f1, length, f1_i

def train(model, optimizer, dataloader, epoch, args):
    all_loss = 0
    model.train_loss.reset()
    optimizer.zero_grad()
    with tqdm(total=len(dataloader), desc='training') as pbar:
        for i, data in enumerate(tqdm(dataloader)):
            loss = model.loss(data, i)
            pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()))
            loss.backward()    
            if i % args.update_size == 0 or i == len(dataloader.dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()
            all_loss += float(loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, all_loss / len(dataloader)))
    return all_loss


def test(model, dataloader, args):
    model.test_loss.reset()
    model.eval()

    pred_all = []
    true_all = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc='testing') as pbar:
            for i, data in enumerate(tqdm(dataloader)):
                t_loss, pred, true = model.test(data)
                pred_all.append(pred)
                true_all.append(true)
                pbar.set_description("t_loss:{}".format(t_loss))

    pred_all = torch.cat(pred_all, dim=0)
    true_all = torch.cat(true_all, dim=0)
    acc(pred_all, true_all)
    torch.cuda.empty_cache()

def main(args):
    if args.predict:
        model = torch.load(args.save+"/model_near_role_enriched.pt").cuda()
        train_list, test_list = load_graphs_with_emb()
        [test_adj_init_all, test_adj_true_all, test_features_all, test_len, test_dis, test_role] = test_list
        test_set = c_GraphDataset(args, test_features_all, test_adj_true_all, test_adj_init_all, test_len, test_dis, test_role)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)
        result = test(model, test_loader, args)

        for scenario in ["suicide_ied", 'wiki_ied_bombings', 'wiki_mass_car_bombings']:
            test_list = pickle.load(open('../../data/'+scenario+'_test_pruned_with_bert_max_50_set_enriched_wo_event.pkl', 'rb'))
            [test_adj_init_all, test_adj_true_all, test_features_all, test_len, test_dis, test_role] = test_list
            test_set = c_GraphDataset(args, test_features_all, test_adj_true_all, test_adj_init_all, test_len, test_dis, test_role)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)
            print(scenario)
            result = test(model, test_loader, args)

            # A_init, x_features, train_len, all_node_dict, dis, e_r, role = read_dataset(scenario)
            # p_dataset = c_predict_GraphDataset(args, x_features, A_init, train_len, dis, role)
            # loader = DataLoader(p_dataset, batch_size=args.batch_size, num_workers=2)
            # rel_list = []
            # model.eval()
            # with torch.no_grad():
            #     with tqdm(total=len(loader), desc='predicting '+scenario) as pbar:
            #         for i, data in enumerate(tqdm(loader)):
            #             new_E = model.predict(data)
            #             rel_list.append(new_E)
            # with open("../result_IGE/dataset_{}_seed_{}".format(scenario, args.seed)+'/preditct_rel_final.pkl', 'wb') as handle:
            #     pickle.dump(rel_list, handle)
            
    else:    
        # get the dataset and dataloader
        train_list, test_list = load_graphs_with_emb()
        input_dims = {'X':args.X_dim, 'E': args.E_dim}
        output_dims = {'X':int(args.X_dim/2), 'E':args.E_dim}

        print("********creating dataset*********")
        [train_adj_init_all, train_adj_true_all, train_features_all, train_len, train_dis, train_role] = train_list
        [test_adj_init_all, test_adj_true_all, test_features_all, test_len, test_dis, test_role] = test_list

        train_set = c_GraphDataset(args, train_features_all, train_adj_true_all, train_adj_init_all, train_len, train_dis, train_role)
        test_set = c_GraphDataset(args, test_features_all, test_adj_true_all, test_adj_init_all, test_len, test_dis, test_role)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)

        model = c_model(args=args, input_dims = input_dims, output_dims = output_dims).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True,
                                    weight_decay=args.weight_decay)


        for epoch in range(args.epochs):
            print("\n-------EPOCH {}-------".format(epoch))
            train(model, optimizer, train_loader, epoch, args)
            if epoch % 50 == 0:
                result = test(model, test_loader, args)
            torch.save(model, args.save+"/model_near_role_enriched.pt")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='using diffusion model to generate complex event schema')
    parser.add_argument('--event_node_type', type=int, default=67, help='number of different event node types')
    parser.add_argument('--entity_node_type', type=int, default=24, help='number of different entity node types')
    parser.add_argument('--dataset', type= str, default='suicide_ied', help='graph dataset name')
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--X_dim", type=int, default=768)
    parser.add_argument("--E_dim", type=int, default=134)
    parser.add_argument("--use_classifer", type = bool, default=True)
    parser.add_argument("--predict", type = bool, default=False)

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='weight_decay')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size during training')
    parser.add_argument('--update_size', type=int, default=8, help='batch size during training')

    parser.add_argument('--infer_batch_size', type=int, default=1, help='batch size during inference')
    parser.add_argument('--seed', type=int, default=6666, help='random seed (default: 1)')
    parser.add_argument('--res_dir', type = str, help = "store the result")
    parser.add_argument('--save', type = str, help = "store the result")
    parser.add_argument('--name', type=str, default = "schema_generation")
    parser.add_argument('--n_layers', type=int, default = 2)
    parser.add_argument('--hidden_dims', type=dict, default = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128})
    parser.add_argument('--hidden_mlp_dims', type=dict, default = {'X':256, 'E':128, 'y':128})
    

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    args.res_dir = "../result_IGE/dataset_{}_seed_{}".format(args.dataset, args.seed)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
    
    args.save = "../result_IGE/seed_{}".format(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    print(args)
    print("add role to the generated events")
    
    # print("suicide_ied")
    # add_role('suicide_ied', args.seed)
    # print("wiki_ied_bombings")
    # add_role('wiki_ied_bombings', args.seed)
    # print("wiki_mass_car_bombings")
    # add_role('wiki_mass_car_bombings',args.seed)

    main(args)


