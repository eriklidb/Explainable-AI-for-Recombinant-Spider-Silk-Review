import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
try:
    import torch
    from transformers.models.auto.modeling_auto import AutoModel
    from transformers.models.auto.tokenization_auto import AutoTokenizer
except:
    print('Torch not loaded')

class EmbeddingGenerator():
    def __init__(self, MODEL_NAME: str) -> None:
        self._sequences = pd.read_csv('../data/sequences.csv', index_col=0).squeeze()
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Load model and tokenizer
        self._model = AutoModel.from_pretrained(MODEL_NAME)
        self._model.eval()  # set to evaluation mode

        # Move model to GPU if available
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def get_embedding(self, sequence: str) -> None:
        """
        Compute the mean-pooled embedding of a protein sequence using ESM-2.
        """
        with torch.no_grad():
            tokens = self._tokenizer(sequence, return_tensors="pt")
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            outputs = self._model(**tokens)
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # shape: (seq_len, hidden_dim)
            attention_mask = tokens["attention_mask"].squeeze(0)

            # Mean-pooling over valid tokens (ignoring padding)
            valid_embeddings = token_embeddings[attention_mask.bool()]
            sequence_embedding = valid_embeddings.mean(dim=0).cpu().numpy()
        return sequence_embedding

    def get_embeddings(self, save_embeddings: bool=True) -> pd.DataFrame:
        # Generate embeddings
        embedding_dict = {}
        for name, seq in tqdm(self._sequences.items(), desc="Embedding sequences", total=len(self._sequences)):
            try:
                embedding = self.get_embedding(seq)
                embedding_dict[name] = embedding
            except Exception as e:
                print(f"Failed to embed {name}: {e}")

        # Convert to DataFrame
        embedding_df = pd.DataFrame.from_dict(embedding_dict, orient="index")
        embedding_df.index.name = "Protein"
        embedding_df.reset_index(inplace=True)

        prots = ['A3IA', 'NT2RepCT', 'A3IA', 'A3IA', 'A3IA', 'NT2RepCT', 'A3IA', 'A3IA', 'A3IA', 'A3IA', 'NT2RepCT', 'NT2RepCT', 'NT2RepCT']
        prots_copies = ['50% Mcherry -A3IA, 50% A3IA', 'NT2RepCT 0.1% MCherry', 'A3I-A +0.1% Carbon Black', '25% Mcherry -A3IA, 75% A3IA', '12.5% Mcherry -A3IA, 87.5% A3IA', 'NT2RepCT + 50 uM Tht', 'A3IA 1 mM Imidazol', 'Mcherry -A3IA', 'A3IA 10 mM Imidazol', 'A3IA + 0.3% AcZ', 'NT2RepCT -Magnetic nanoparticles', 'NT2RepCT 0.1% Bri2MCherry', 'NT2RepCT 0.5% Bri2MCherry']
        for prot, prot_copy in zip(prots, prots_copies):
            ind = int(np.where(embedding_df.loc[:, 'Protein'] == prot)[0][0])
            embedding_df.loc[len(embedding_df)] = embedding_df.loc[ind]
            embedding_df.loc[len(embedding_df) - 1, 'Protein'] = prot_copy

        prots_combos = [('Rep7', 'NT2RepCT'), ('Rep7', 'NT2RepCT'), ('IdeS-A3IA', 'A3IA'), ('VN-NT', 'A3IA')]
        prots_new = ['Rep7 Tusp (30%) & NT2RepCT (70%)', 'Rep7 Tusp (57%) & NT2repCT (43%)', '20% IdeS-A3IA, 80% A3I_A', '5% VN-NT & A3IA']
        fracs = [(.3, .7), (.57, .43), (.2, .8), (.05, .95)]
        for prots, prot_new, fracs in zip(prots_combos, prots_new, fracs):
            inds = [int(np.where(embedding_df.loc[:, 'Protein'] == prot)[0][0]) for prot in prots]
            embedding_df.loc[len(embedding_df)] = 0
            embedding_df.loc[len(embedding_df) - 1, 'Protein'] = prot_new
            embedding_df.iloc[len(embedding_df) - 1, 1:] = fracs[0]*embedding_df.iloc[inds[0], 1:] + fracs[1]*embedding_df.iloc[inds[1], 1:]

        if save_embeddings:
            embedding_df.to_csv(os.path.join(os.pardir, 'data', 'protein_embeddings.csv'), index=False)
        return embedding_df

def pca(embedding_df: pd.DataFrame, n_components=15, save_embeddings: bool=True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Drop the protein identifier column (first column) to extract only numeric data
    X = embedding_df.drop(columns=["Protein"])

    # Standardize the data before PCA (important for PCA to work properly)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    print(X_scaled.shape)
    X_pca = pca.fit_transform(X_scaled)

    # Create a new DataFrame with PCA components
    pca_columns = [f"pca_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df.insert(loc=0, column="Protein", value=embedding_df["Protein"].to_numpy())

    # Optional: Save or inspect
    if save_embeddings:
        pca_df.to_csv(os.path.join(os.pardir, 'data', 'protein_embeddings_pca.csv'), index=False)
    return pca_df, pca.explained_variance_ratio_, pca.singular_values_


if __name__ == '__main__':
    MODEL_NAMES = ["facebook/esm2_t6_8M_UR50D",
                "facebook/esm2_t12_35M_UR50D",
                "facebook/esm2_t30_150M_UR50D",
                "facebook/esm2_t33_650M_UR50D",
                "facebook/esm2_t36_3B_UR50D",
                "facebook/esm2_t48_15B_UR50D"]
    MODEL_NAME = MODEL_NAMES[2]
    embedding_gen = EmbeddingGenerator(MODEL_NAME)
    embedding_df = embedding_gen.get_embeddings()
    pca_df, explained_variance_ratio, singular_values = pca(embedding_df)