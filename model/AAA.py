from datautils import PairTreeFolder
import argparse
from vocab import Vocab, common_atom_vocab
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str, default="../data/tensors.pkl", help='data path to training data')
parser.add_argument('--valid', type=str, default="../data/valid.csv", help='data path to validation data')

parser.add_argument('--vocab', type=str, default="../data/vocab.txt", help='data path to substructure vocabulary')
parser.add_argument('--save_dir', type=str, default="../result/center_models", help='data path to the directory used to save trained models')
parser.add_argument('--load_epoch', type=int, default=0, help='an interger used to control the loaded model (i.e., if load_epoch==1000, '+\
                    'the model save_dir+1000.pkl would be loaded)')
parser.add_argument('--ncpu', type=int, default=8, help='the number of cpus')

parser.add_argument('--size', type=int, default=70000, help='size of training data')
parser.add_argument('--hidden_size', type=int, default=256, help='the dimension of hidden layers')
parser.add_argument('--batch_size', type=int, default=256, help='the number of molecule pairs in each batch')
parser.add_argument('--latent_size', type=int, default=32, help='the dimention of latent embeddings')
parser.add_argument('--embed_size', type=int, default=32, help='the dimention of substructure embedding')
parser.add_argument('--depthG', type=int, default=5, help='the depth of message passing in graph encoder')
parser.add_argument('--depthT', type=int, default=3, help='the depth of message passing in tree encoder')

parser.add_argument('--use_atomic', action="store_false", help='whether to use atomic number as feature (default value is True)')
parser.add_argument('--use_node_embed', action="store_true", help='whether to use the substructure embedding in the prediction functions (default value is False)')
parser.add_argument('--use_brics', action="store_true", help='whether to use brics substructures in the encoder (default value is False)')    
parser.add_argument('--use_feature', action='store_false', help='whether to use the atom features or not')
parser.add_argument('--use_class', action='store_true', help='whether the reaction types are known')
parser.add_argument('--update_embed', action='store_true')
parser.add_argument('--shuffle', action='store_false')
parser.add_argument('--use_product', action='store_false')
parser.add_argument('--use_attachatom', action='store_true')
parser.add_argument('--use_latent_attachatom', action='store_true')
parser.add_argument('--sum_pool', action='store_false')
parser.add_argument('--use_mess', action='store_true')
    
parser.add_argument('--use_atom_product', action='store_true')

parser.add_argument('--network_type', type=str, default='gcn')
parser.add_argument('--clip_norm', type=float, default=10.0, help='')
parser.add_argument('--use_tree', action='store_true')

# control the learning process
parser.add_argument('--epoch', type=int, default=150, help='the number of epochs')
parser.add_argument('--total_step', type=int, default=-1, help='the number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate')
parser.add_argument('--print_iter', type=int, default=20)
parser.add_argument('--save_iter', type=int, default=3000)

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]    #83个词汇表
vocab = Vocab(vocab)
avocab = common_atom_vocab  #内容为['B','C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S', 'Cl', 'Cu', 'Zn', 'Se', 'Br', 'Sn', 'I']的类

loader = PairTreeFolder(args.train, vocab, avocab, args, is_train_center=True)
print(loader)