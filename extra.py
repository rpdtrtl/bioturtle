
# FASTA to dictionary parser
def parse_fasta_file(x):
    parsed_seqs = {}
    curr_seq_id = None
    curr_seq = []

    for line in x:
        line = line.strip()

        if line.startswith(">"):
            if curr_seq_id is not None:
                parsed_seqs[curr_seq_id] = ''.join(curr_seq)

            curr_seq_id = line[1:]
            curr_seq = []
            continue

        curr_seq.append(line)

    parsed_seqs[curr_seq_id] = ''.join(curr_seq)
    return parsed_seqs

def tokenize(input):
    a = {}
    return a

# Tokenization
all_aan = "ARNDCQEGHILKMFPSTWYVUX"
extra_n = ['Other', 'End', 'Start', 'Pad']

aa_index = {aa: i for i, aa in enumerate(all_aan)}
extra_index = {token: i + len(all_aan) for i, token in enumerate(extra_n)}
ttoi = {**aa_index, **extra_index}
itot = {index: token for token, index in ttoi.items()}
n_tokens = len(ttoi)

def tokenize_seq(seq):
    other_token_index = aa_index['<OTHER>']
    return [extra_index['<START>']] + [aa_index.get(aa, other_token_index) for aa in seq] + \
            [extra_index['<END>']]