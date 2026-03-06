# Explainable AI for Recombinant Spider Silk: Spinning Parameters and Sequence Effects

## About
Materials concerning the work "Explainable AI for Recombinant Spider Silk: Spinning Parameters and Sequence Effects". Supplementary material referenced in the report can be found in "supplementary.pdf".
The Spidrospin dataset containing the data for the artificial spider silk experiments and fibers is as of this moment confidential, and can unfortunately not be shared. The PLM vector embeddings however, are shared. Given the dataset (or a similar one tabular dataset recording spidroin processing parameters), all the experiments can be reproduced by utilizing a virtual environment and 
the requirenments file(s).

## Structure
* The code for the data processing and experiments/analysis can be found in ``/src``. 
* The generated figures can be found in ``/figures``. 
* The trained outer-fold models can be found in ``/models``.

### Code structure
* The ``data_processing`` module cleans the raw data found in multiple spreadsheets, makes it suitable for training, and saves it to another format (csv/hdf/xlsx). 
* The ``compare_properties`` module searches for matches for the protein sequences in the dataset using Smith-Waterman local alignment algorithm.
* The ``protein_sequences`` module runs code to derive vector embeddings from the corresponding protein sequences in the dataset. Utilizes the ``torch``and ``transformers`` modules.
* The ``dataset`` module wraps the cleaned dataset into a class. 
* The ``model_trainer`` module sets up the training scheme and model strucutre. 
* The ``evalute_models`` module conducts the main experiments, which compares the SpinML and SeqSpinML model across metrics, generates comparative plots, shows SHAP importance values across spinnig conditions, and conducts an ablation study. 
* The ``sequence_matching`` module compares the similarity between sequences in the Spidrospin dataset and from the Spider Silkome DB.

## Installation guide
If you want to play around with the code yourself, you can follow these steps on your terminal:
1. Clone the repository: 
```
git clone https://github.com/eriklidb/Explainable-AI-for-Recombinant-Spider-Silk-Review
2. Create a virtual environment: 
```
python -m venv spidro_env
```
3. Activate it. On Linux/MacOS: 
```
source spidro_env/bin/activate
``` 
On Windows 
```
spidro_env\Scripts\activate
```
4. Install core dependencies: 
```
python -m pip install --upgrade pip 
python -m pip install -r requirements.txt
```
5. (Optional) installing ``torch`` and ``transformers`` will enable you to run the ``protein_sequences`` module and reproduce the PLM-derived embeddings:
```
python -m pip install -r requirements-plm.txt
```
