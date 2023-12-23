import numpy as np
import tensorflow as tf
from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input PDB file")
parser.add_argument("-out","--path_output", type=str, help="the path of output portTrans file")


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat
    
def load_sec_struct_model():
    checkpoint_dir = "ProtTrans/protT5/sec_struct_checkpoint/secstruct_checkpoint.pt"
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)

    return model



def read_fasta(fasta_path, split_char="!", id_field=0):
    seq = ''
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if not line.startswith('>'):
                seq += line.strip()

    seq_id = os.path.splitext(os.path.basename(fasta_path))[0] # Get only the file name without path and extension
    seqs = [(seq_id, seq)]

    return seqs





def get_T5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    # Rostlab/prot_t5_xl_uniref50
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer



def get_embeddings(model, tokenizer, seqs, max_residues=4000, max_seq_len=1000, max_batch=100):
    results = {"residue_embs": dict()}

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs, key=lambda x: len(x[1]), reverse=True)
    start = time.time()
    batch = []

    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = []

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if "residue_embs" in results:
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"])

    return results

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, tokenizer = get_T5_model()

def save_port_map(port_data,output_file):
    np.savetxt(output_file, port_data)


def main(fasta_file, output_file):
    filename = os.path.splitext(os.path.basename(fasta_file))[0]
    seqs = read_fasta(fasta_file)
    results = get_embeddings( model, tokenizer, seqs)
    embeddings = results['residue_embs'][filename]
    save_port_map(embeddings, output_file)
    
    
# if __name__ == "__main__":
#     args = parser.parse_args()
#     input=os.listdir(args.path_input)
#     str=".fasta"
#     for i in input:
#         if i.endswith(str):
#             file_name="/"+i.split(".")[0]
#             print(args.path_input+"/"+i)
#             main(args.path_input+"/"+i, args.path_output+file_name+".prottrans")



if __name__ == "__main__":
    args = parser.parse_args()
    input = os.listdir(args.path_input)
    str = ".fasta"
    for i in input:
        if i.endswith(str):
            try:
                file_name = "/" + i.split(".")[0]
                print(args.path_input + "/" + i)
                main(args.path_input + "/" + i, args.path_output + file_name + ".prottrans")
            except Exception as e:
                log_mode = 'a' if os.path.exists("pt_log.txt") else 'w'
                with open("pt_log.txt", log_mode) as pt_log:
                    pt_log.write(i + "\n")
                print(f"處理 {i} 時發生錯誤：{e}")
                continue