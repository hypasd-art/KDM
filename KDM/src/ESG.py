import os
import argparse
import torch
from utils import *
from datasets.graph_dataset import *
from datasets.datamodule import *
from model.diffusion_model_discrete import *
from eval_res import *

def train(model, optimizer, dataloader, epoch, args):
    all_loss = 0
    model.train_loss.reset()
    
    with tqdm(total=len(dataloader), desc='training') as pbar:
        for i, data in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            loss, loss_X, loss_E = model.loss(data)
            pbar.set_description('Epoch: %d, loss: %0.4f, node_loss: %0.4f, edge_loss: %0.4f' % (epoch, loss.item(), loss_X.item(), loss_E.item()))
            loss.backward()    
            optimizer.step()
            all_loss += float(loss)
    
    model.train_loss.log_epoch_metrics(epoch)
    return all_loss


def test(model, dataloader, args, idx):
    model.eval()
    with torch.no_grad():
        model.sample_test(idx, args.training_sample_to_save)
    torch.cuda.empty_cache()

def main(args):
    if args.predictor:
        if args.data_type == "IGE":
            model = torch.load(args.save+"/model_expansion.pt")
        elif args.data_type == "IGE":
            model = torch.load(args.save+"/model_expansion_json.pt")
        elif args.data_type == "None":
            model = torch.load(args.save+"/model.pt")

        model.eval()
        _, test_list = load_graphs_with_type_label(args, name = args.dataset)

        test_dataset = [None for i in range(len(test_list))]
        test_dataloader = [None for i in range(len(test_list))]
                    
        with torch.no_grad():
            for i in range(len(data_dict)):
                model.sample_test(i, args.training_sample_to_save)
                evaluation_generation_res(data_dict[i], args, seed=args.seed)
        print("generate done")
    else:
        train_list, test_list = load_graphs_with_type_label(args, name = args.dataset)
        print("*****************************creating dataset******************************")
        train_dataset = GraphDataset(args, train_list)
        info_depth = train_dataset.calcu_info_depth()
        info_type = train_dataset.calcu_info_type()
        info_num = train_dataset.calcu_info_num()

        print("margin node type")
        print(info_type)
        print("margin node num")
        print(info_num)
        print("margin node depth")
        print(info_depth)

        test_dataset = [None for i in range(len(test_list))]
        test_dataloader = [None for i in range(len(test_list))]

        for i in range(len(test_list)):
            test_dataset[i] = GraphDataset(args, test_list[i])
            test_dataloader[i] = DataLoader(test_dataset[i], batch_size=args.batch_size, num_workers=2)

        datasets = {"train": train_dataset, "val": None, "test": None}

        datamodule = DataModule(args)
        datamodule.prepare_data(datasets)

        dataset_infos = datainfos(datamodule=datamodule, args=args)

        # 计算输入输出维度
        dataset_infos.compute_input_output_dims(datamodule=datamodule)

        depth_sampler_tools = depth_sampler(info_depth)

        # get model info
        model_kwargs = {'dataset_infos': dataset_infos, 'depth_sampler':depth_sampler_tools}
        
        model = DiscreteDenoisingDiffusion(args=args, **model_kwargs).cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True,
                                    weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            print("\n-------EPOCH {}-------".format(epoch))
            train(model, optimizer, datamodule.train_dataloader(), epoch, args)
            if epoch % 100 == 0:

                for i in range(len(data_dict)):
                    result = test(model, test_dataloader[i], args, i)
                    evaluation_generation_res(data_dict[i], args, seed=args.seed)
                if args.data_type == 'None':
                    torch.save(model, args.save+"/model.pt")
                elif args.data_type == 'IGE':
                    torch.save(model, args.save+"/model_expansion.pt")
                elif args.data_type == 'json':
                    torch.save(model, args.save+"/model_expansion_json.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='using diffusion model to generate complex event schema')
    parser.add_argument('--event_node_type', type=int, default=67, help='number of different event node types')
    parser.add_argument('--entity_node_type', type=int, default=24, help='number of different entity node types')
    parser.add_argument('--dataset', type= str, default='suicide_ied', help='graph dataset name')
    parser.add_argument('--num_vertex_type', type=int)
    parser.add_argument('--num_edge_type', type=int, default=2)
    parser.add_argument('--max_n', type=int)
    parser.add_argument('--START_TYPE', type=int)
    parser.add_argument('--END_TYPE', type=int)
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--predictor", type=bool, default=False)
    parser.add_argument("--use_pos", type=bool, default=True)
    parser.add_argument("--data_type", type=str, default="IGE")
    parser.add_argument("--save_process_pic", type=bool, default=False)

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-12, help='weight_decay')
    parser.add_argument('--epochs', type=int, default=2501, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size during training')
    parser.add_argument('--infer_batch_size', type=int, default=1, help='batch size during inference')
    parser.add_argument('--seed', type=int, default=6666, help='random seed')
    parser.add_argument('--res_dir', type = list, help = "store the result")
    parser.add_argument('--save', type = str, help = "store the model")
    parser.add_argument('--lambda_train', default = [3, 0])
    parser.add_argument('--name', type=str, default = "schema_generation")
    parser.add_argument('--diffusion_steps', type=int, default = 500)
    parser.add_argument('--n_layers', type=int, default = 12)
    parser.add_argument('--hidden_dims', type=dict, default = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128})
    parser.add_argument('--hidden_mlp_dims', type=dict, default = {'X':256, 'E':128, 'y':128})
    parser.add_argument('--diffusion_noise_schedule', type=str, default = 'cosine')
    parser.add_argument('--number_chain_steps', type=int, default=50)
    
    parser.add_argument('--training_sample_to_save', type=int, default=20)
    parser.add_argument('--testing_sample_to_save', type=int, default=500)
    

    args = parser.parse_args()
    data_dict = {0:"suicide_ied", 1:"wiki_ied_bombings", 2:"wiki_mass_car_bombings"}

        
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"

    args.res_dir = [None for i in range(len(data_dict))]
    for i in range(len(data_dict)):
        args.res_dir[i] = "../result_" + args.data_type + "/dataset_{}_seed_{}".format(data_dict[i], args.seed)
        if not os.path.exists(args.res_dir[i]):
            os.makedirs(args.res_dir[i]) 
    print(args.res_dir)

    args.save = "../result_" + args.data_type + "/seed_{}".format(args.seed)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)
    main(args)


