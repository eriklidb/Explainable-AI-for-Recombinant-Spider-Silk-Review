from Bio import SeqIO
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import pairwise2

def local_align_score(seq1, seq2):
    alignments = pairwise2.align.localms(
        seq1, seq2,
        2,    # match
        -1,   # mismatch
        -5,   # gap open
        -0.5  # gap extend
    )
    return alignments[0].score if alignments else 0

def normalized_similarity(seq1, seq2):
    raw = local_align_score(seq1, seq2)
    return raw / min(len(seq1), len(seq2))

def match_sequences(ds_id, ds_seq, seqs_silkome, top_k=5):
    matches = {}

    scores = []

    for silk_id, silk_seq in tqdm(seqs_silkome.items(), desc=f'Aligning {ds_id}'):
        sim = normalized_similarity(ds_seq, silk_seq)
        #print("Similarity score: ", sim)
        scores.append((silk_id, sim))

    # Sort by similarity (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Keep top-k
    matches[ds_id] = scores[:top_k]
        
    return matches

def matches_to_dataframe(matches):
    rows = []

    for ds_id, hits in matches.items():
        for rank, (silk_id, score) in enumerate(hits, start=1):
            rows.append({
                "ds_id": ds_id,
                "rank": rank,
                "silkome_id": silk_id,
                "similarity": score
            })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    seqs = pd.read_csv('../data/sequences.csv', index_col=0).squeeze()
    seqs_ds = {}
    for prot in ['NT2RepCT', 'A3IA', 'Rep1', 'Rep2', 'Rep3', 'Rep5', 'Rep7', '4A 2rep', 'VN-A3IA', 'fNT A3IA', 'Br_MaSp2_300', 'Br_MaSp2_400', 'Br_MaSp2_long', 'Br_MaSp2_short']:
        seqs_ds[prot] = seqs[prot]

    fasta_path = "../data/spider-silkome-database.v1.prot.fasta"

    seqs_silkome = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id          # e.g. "SeqID_123"
        sequence = str(record.seq)  # amino acid sequence
        seqs_silkome[seq_id] = sequence

    print(f"Loaded {len(seqs_silkome)} sequences")# Store sequences by ID

    k=100
    for ds_id, ds_seq in seqs_ds.items():
        matches = match_sequences(ds_id, ds_seq, seqs_silkome, top_k=k)
        df_matches = matches_to_dataframe(matches)
        df_matches.to_csv(f"../data/sequence_matches/top_{k}_sequence_matches_{ds_id}.csv", index=False)