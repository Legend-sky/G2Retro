import torch 
import torch.nn as nn
import rdkit
import numpy as np
import rdkit
import sys
import rdkit.Chem as Chem
from model.mol_tree import MolTree
from model.chemutils import is_sim
from model.molsynthon import MolSynthon
from argparse import ArgumentParser
from model.mol_enc import MolEncoder
from model.datautils import MolTreeFolder
from model.vocab import Vocab, common_atom_vocab
from model.config import device, SUB_CHARGE_NUM, SUB_CHARGE_OFFSET, SUB_CHARGE_CHANGE_NUM, BOND_SIZE, \
             VALENCE_NUM, HYDROGEN_NUM, IS_RING_NUM, IS_CONJU_NUM, REACTION_CLS, IS_AROMATIC_NUM

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = ArgumentParser()
#dest参数是一个字符串，用来保存解析后的参数
parser.add_argument("-t", "--test", dest="test_path", default="../data/test_8_mol.txt")
parser.add_argument("-m1", "--mod_cen_path", dest="mod_cen_path", default="../trained_models/center_models/model_center_optim.pt")
parser.add_argument("-m2", "--mod_syn_path", dest="mod_syn_path", default="../trained_models/synthon_models/model_synthon_optim.pt")
parser.add_argument("-d", "--save_dir", dest="save_dir", default="../result/test_8_mol/")
parser.add_argument("-o", "--output", dest="output", default="output")
parser.add_argument("-st", "--start", type=int, dest="start", default=0)
parser.add_argument("-si", "--size", type=int, dest="size", default=8)

parser.add_argument("--vocab", type=str, default="../data/vocab.txt")
parser.add_argument("--ncpu", type=int, default=10)
parser.add_argument("--decode_type", type=int, default=2)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--knum", type=int, default=10)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_sizeS', type=int, default=512)
parser.add_argument('--hidden_sizeC', type=int, default=256)
parser.add_argument('--embed_sizeS', type=int, default=32)
parser.add_argument('--embed_sizeC', type=int, default=32)
parser.add_argument('--latent_sizeS', type=int, default=32)
parser.add_argument('--latent_sizeC', type=int, default=32)
parser.add_argument('--depthGS', type=int, default=10)
parser.add_argument('--depthGC', type=int, default=10)
parser.add_argument('--depthT', type=int, default=3)

parser.add_argument('--reduce_dim', action="store_true")
parser.add_argument('--use_atomic', action="store_false")
parser.add_argument('--sum_pool', action="store_false")
parser.add_argument('--use_class', action="store_true")
parser.add_argument('--use_edit', action="store_true")
parser.add_argument('--use_node_embed', action="store_true")
parser.add_argument('--use_brics', action="store_true")
parser.add_argument('--update_embed', action="store_true")
parser.add_argument('--use_product', action='store_false')
parser.add_argument('--network_type', type=str, default='gcn')
parser.add_argument('--use_latent_attachatom', action="store_true")
parser.add_argument('--use_attachatom', action='store_true')
parser.add_argument('--use_feature', action="store_false")    
parser.add_argument('--use_match', action="store_true")
parser.add_argument('--use_mess', action="store_true")
parser.add_argument('--use_tree', action="store_true")
parser.add_argument('--without_target', action="store_true", help='whether the input data includes ground truth reactants')

parser.add_argument('--num', type=int, default=20)

args = parser.parse_args()

def make_cuda(tensors, product=True, ggraph=None):
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x, requires_grad=False)
    
    if len(tensors) == 2:
        graph_tensors, tree_tensors = tensors
        new_tree_tensors1 = [x if x is None else make_tensor(x).to(device).long() for x in tree_tensors[:6]]
        new_tree_tensors = new_tree_tensors1 + [tree_tensors[-1]]
    else:
        graph_tensors = tensors[0]
        
    new_graph_tensors1 = [make_tensor(x).to(device).long() for x in graph_tensors[:6]]
    
    if len(graph_tensors) > 8:
        new_graph_tensors2 = [graph_tensors[-3], make_tensor(graph_tensors[-2]).to(device), make_tensor(graph_tensors[-1]).to(device)]
        new_graph_tensors = new_graph_tensors1 + new_graph_tensors2
    elif not product:
        new_graph_tensors = new_graph_tensors1 + [graph_tensors[-1], None, None]
    else:
        new_graph_tensors = new_graph_tensors1 + [graph_tensors[-1]]
    
    if len(tensors) == 2:
        return new_graph_tensors, new_tree_tensors, make_tensor(ggraph).to(device)
    else:
        return [new_graph_tensors]

class MolCenter(nn.modules):

    def __init__(self, vocab, avocab, args):
        super(MolCenter, self).__init__()
        self.vocab = vocab  #分子词汇表
        self.avocab = avocab    #原子词汇表
        self.hidden_size = args.hidden_size #隐藏层大小256
        self.latent_size = args.latent_size #潜在空间大小
        self.atom_size = atom_size = avocab.size()  #原子词汇表大小,16
        self.use_brics = args.use_brics #是否使用BRICS特征
        self.use_feature = args.use_feature     #是否使用特征
        self.use_tree = args.use_tree   #是否使用树结构
        self.use_latent_attachatom = args.use_latent_attachatom
        self.use_class = args.use_class     #是否使用类别
        self.use_mess  = args.use_mess  # 是否使用键特征

        self.charge_offset = SUB_CHARGE_OFFSET  #1
        self.charge_change_num = SUB_CHARGE_CHANGE_NUM  #3
        self.charge_num = SUB_CHARGE_NUM    #3
        
        # embedding for substructures and atoms
        self.E_a = torch.eye(atom_size).to(device)  #原子嵌入层，使用单位矩阵初始化,16x16单位矩阵
        if args.use_feature:    #特征嵌入层
            self.E_fv = torch.eye( VALENCE_NUM ).to(device) #8
            self.E_fg = torch.eye( self.charge_num ).to(device)
            
            self.E_fh = torch.eye( HYDROGEN_NUM ).to(device)    #6
            self.E_fr = torch.eye( IS_RING_NUM ).to(device)     #2
            self.E_fc = torch.eye( IS_CONJU_NUM ).to(device)    #2
            self.E_fa = torch.eye( IS_AROMATIC_NUM ).to(device) #2
        
        self.E_b = torch.eye(BOND_SIZE).to(device)  #键嵌入层，使用单位矩阵初始化，3x3单位矩阵

        if self.use_feature:
            feature_embedding = (self.E_a, self.E_fv, self.E_fg, self.E_fh, self.E_fr, self.E_fc, self.E_fa, self.E_b)
        else:
            feature_embedding = (self.E_a, self.E_b) 

        if self.use_class:
            self.reactions = torch.eye( REACTION_CLS ).to(device)   #10
            feature_embedding += (self.reactions, )
        
        # encoder：使用 MolEncoder 类初始化编码器，用于编码分子结构
        self.encoder = MolEncoder(self.atom_size, feature_embedding, args=args)
    
    def encode(self, tensors, product=False, classes=None, use_feature=False, usemask=False):
        """ Encode the molecule during the test
        
        Args:
            tensors: input embeddings
        """
        tensors[0][0][:, 2] = tensors[0][0][:, 2] + self.charge_offset
        mol_rev_vecs, mol_atom_vecs, mol_mess_vecs = self.encoder(tensors, product=product, classes=classes, use_feature=use_feature, usemask=usemask)
        
        tensors[0][0][:, 2] = tensors[0][0][:, 2] - self.charge_offset
        return mol_rev_vecs, mol_atom_vecs, mol_mess_vecs
    
    def test(self, classes, product_batch, product_trees, react_smiles, decode_type=1, has_gt=False, knum=1):
        product_graphs, product_tensors, product_ggraph, _ = product_batch
        
        product_tensors = make_cuda(product_tensors, ggraph=product_ggraph)
        product_scope = product_tensors[0][-1]
                
        # encoding
        product_embed_vecs, product_atom_vecs, product_mess_vecs = self.encode(product_tensors, product=True, use_feature=self.use_feature, classes=classes, usemask=False)
        
        product_data = (product_embed_vecs, product_tensors, product_atom_vecs, product_trees)
        
        
    

vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)
avocab = common_atom_vocab

# ======== 加载反应中心识别模型 =========
args.depthG = args.depthGC
args.hidden_size = args.hidden_sizeC
args.latent_size = args.latent_sizeC
args.embed_size = args.embed_sizeC

model_center = MolCenter(vocab, avocab, args)
try:
    model_center.load_state_dict(torch.load(args.mod_cen_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")

# ======== 加载合成子完成模型 ===========
args.depthG = args.depthGS
args.hidden_size = args.hidden_sizeS
args.latent_size = args.latent_sizeS
args.embed_size = args.embed_sizeS

# synthon completion module does not use brics
tmp1 = args.use_brics
tmp2 = args.use_tree
args.use_tree = False
args.use_brics = False
model_synthon = MolSynthon(vocab, avocab, args)
try:
    model_synthon.load_state_dict(torch.load(args.mod_syn_path, map_location=torch.device(device)))
except:
    raise ValueError("model does not exist")
model_synthon.charge_set = 0
args.use_brics = tmp1
args.use_tree = tmp2
args.with_target = not args.without_target

# ============== 加载测试数据集 ==================
data = []
with open(args.test_path) as f:
    for line in f.readlines()[1:]:
        s = line.strip("\r\n ").split(",")
        if args.with_target:
            smiles = s[2].split(">>")
            data.append((int(s[0]), s[1], smiles[1], smiles[0]))
        else:
            data.append((0, '', s[0], ''))

output = []
start = int(args.start)
end = int(args.size) + start if args.size > 0 else len(data)

# ============= 准备输出文件 =======================
output_path = args.save_dir + args.output + "_" + str(args.knum) + "_%d" %(start) + "_%d" % (end)

res_file = open("%s_res.txt" % (output_path), 'w') # result file with top-10 exact match labels
error_file = open("%s_error.txt" % (output_path), 'w') # error file with logs
pred_file = open("%s_pred.txt" % (output_path), 'w') # prediction file with all the predicted reactants

loader = MolTreeFolder(data[start:end], vocab, avocab, args.ncpu, args.batch_size, with_target=args.with_target, test=True, use_atomic=True, use_class=args.use_class, use_brics=args.use_brics, use_feature=True, del_center=True, usepair=False)

top_10_bool = np.zeros((len(loader.prod_list), 10))
prod_list = []
num = 0

# ============= 开始测试 =======================
for batch in loader:
    classes, product_batch, product_tree, reacts_smiles, product_smiles, synthon_smiles, skip_idxs = batch

    # print(len(product_batch))
    # print(product_batch)

    with torch.no_grad():
        top_k_trees, buffer_log_probs = model_center.test(classes, product_batch, product_tree, reacts_smiles, knum=args.knum)
        #打印top_k_trees信息
        # print(len(top_k_trees)) #10
        # print(top_k_trees)
        # print('------------')
        # print(len(buffer_log_probs))    #7
        # print(buffer_log_probs)

        top_k_synthon_batch = [None for _ in top_k_trees]
        
        test_product_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(tree.smiles)) for tree in product_tree]
        for i, trees in enumerate(top_k_trees):
            _, tensors = MolTree.tensorize(trees, vocab=vocab, istest=True, use_atomic=True, use_feature=args.use_feature, avocab=avocab, product=False)
            top_k_synthon_batch[i] = tensors

        pre_react_smiles, pre_react_logs = model_synthon.test_synthon_beam_search(classes, product_batch, product_tree, top_k_trees, top_k_synthon_batch, \
                                                                                  buffer_log_probs, knum=10, product_smiles=test_product_smiles)
        #输出结果数量以及结果
        # print('------------')
        # print(len(pre_react_smiles))    # 7
        # print(pre_react_smiles)
        # print('------------')
        # print(len(pre_react_logs))   #7
        # print(pre_react_logs)
    if args.with_target:
        idx = 0
        for i, react_smile in enumerate(reacts_smiles):
            if i in skip_idxs:
                continue
            
            for j in range(len(pre_react_smiles[idx])):
                pre_smile = pre_react_smiles[idx][j]
                
                if is_sim(pre_smile, react_smile):
                    top_10_bool[num+i, j:] = 1
                    print("%s match (%.2f)" % (product_smiles[i][0], pre_react_logs[idx][j]))
                    break
                else:
                    string = "%s: %s fail to match %s with %s (%.2f)\n" % (product_smiles[i][0], product_smiles[i][1], react_smile, pre_smile, pre_react_logs[idx][j])
                    print(string)
                    error_file.write(string)
            idx += 1
            
        batch_10_acc = np.mean(top_10_bool[num:num+len(product_smiles), :], axis=0)
        print("iter: top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (batch_10_acc[0], batch_10_acc[1], batch_10_acc[2], batch_10_acc[3], batch_10_acc[4], batch_10_acc[-1]))
    
        cumu_10_acc = np.mean(top_10_bool[:num+len(product_smiles), :], axis=0)
        print("iter: top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (cumu_10_acc[0], cumu_10_acc[1], cumu_10_acc[2], cumu_10_acc[3], cumu_10_acc[4], cumu_10_acc[-1]))

        
        for i, (idx, prod) in enumerate(product_smiles):
            string = "%s %s %s\n" % (idx, prod, " ".join([str(top_10_bool[num+i, j]) for j in range(10)]))
            res_file.write(string)

    idx = 0
    for i, react_smile in enumerate(reacts_smiles):
        if i in skip_idxs:
            pred_file.write("%d %s %s --- --- 0.00\n" % (i + num, product_smiles[i][0], product_smiles[i][1]))
            continue
        for j in range(len(pre_react_smiles[idx])):
            pre_smile = pre_react_smiles[idx][j]
            pred_file.write("%d %s %s %s %s %.2f\n" % (i + num, product_smiles[i][0], product_smiles[i][1], react_smile, pre_smile, pre_react_logs[idx][j]))
        idx += 1
        
    num += len(product_smiles)
    sys.stdout.flush()
    error_file.flush()
    res_file.flush()
    pred_file.flush()
    
res_file.close()
error_file.close()
pred_file.close()

if args.with_target:
    top_10_acc = np.sum(top_10_bool, axis=0) / len(loader.prod_list)
    print("top 10 accuracy: %.4f  %.4f  %.4f  %.4f  %.4f  %.4f" % (top_10_acc[0], top_10_acc[1], top_10_acc[2], top_10_acc[3], top_10_acc[4], top_10_acc[-1]))









