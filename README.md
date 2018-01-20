# Cross-type Biomedical Name Tagger with Deep Multi-task Learning

This project provides a neural network based multi-task learning framework for biomedical named entity recognition (BioNER).

The implementation is based on the PyTorch library. Our model collectively trains different biomedical entity types to build a unified model that benefits the training of each single entity type and achieves a significantly better performance compared with the state-of-the-art BioNER systems.

## Quick Links

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Prediction](#prediction)

## Installation

For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.

### PyTorch

The code is based on PyTorch. You can find installation instructions [here](http://pytorch.org/). 

### Dependencies

The code is written in Python 3.6. Its dependencies are summarized in the file ```requirements.txt```. You can install these dependencies like this:
```
pip3 install -r requirements.txt
```

## Data

We use five biomedical corpora collected by Crichton et al. for biomedical NER. The dataset is publicly available and can be downloaded from [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). The details of each dataset are listed below:

|Dataset | Entity Type | Dataset Size |
| ------------- |-------------| -----|
| [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES) | Gene/Protein | 20,000 sentences |
| [BC4CHEMD](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC4CHEMD-IOBES) | Chemical | 10,000 abstracts |
| [BC5CDR](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-IOBES) | Chemical, Disease | 1,500 articles |
| [NCBI-disease](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/NCBI-disease-IOBES) | Disease | 793 abstracts |
| [JNLPBA](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/JNLPBA-IOBES) | Gene/Protein, DNA, Cell Type, Cell Line, RNA | 2,404 abstracts |

### Format

Users may want to use other datasets. We assume the corpus is formatted as same as the CoNLL 2003 NER dataset.

More specifically, **empty lines** are used as separators between sentences, and the separator between documents is a special line as below.
```
-DOCSTART- -X- -X- -X- O
```
Other lines contains words, labels and other fields. **Word** must be the **first** field, **label** mush be the **last**, and these fields are **separated by space**.
For example, the first several lines in the WSJ portion of the PTB POS tagging corpus should be like the following snippet.

```
-DOCSTART- -X- -X- -X- O

Selegiline	S-Chemical
-	O
induced	O
postural	B-Disease
hypotension	E-Disease
in	O
Parkinson	B-Disease
'	I-Disease
s	I-Disease
disease	E-Disease
:	O
a	O
longitudinal	O
study	O
on	O
the	O
effects	O
of	O
drug	O
withdrawal	O
.	O
```

### Embedding
We initialize the word embedding matrix with pre-trained word vectors from Pyysalo et al., 2013. These word vectors are
trained using the skip-gram model on the PubMed abstracts together with all the full-text articles
from PubMed Central (PMC) and a Wiikipedia dump. You can download the embedding files from [here](http://evexdb.org/pmresources/vec-space-models/). 

Please do not forget to [convert](https://github.com/anotheremily/bin2txt) the .bin file to a .txt file.

## Usage

```train_wc.py``` is the script for our multi-task LSTM-CRF model.
The usages of it can be accessed by the parameter ````-h````, i.e., 
```
python train_wc.py -h
```

The default running commands are:

```
python3 train_wc.py --train_file [training file 1] [training file 2] ... [training file N] \
                    --dev_file [developing file 1] [developing file 2] ... [developing file N] \
                    --test_file [testing file 1] [testing file 2] ... [testing file N] \
                    --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --word_dim 200
```

Users may incorporate an arbitrary number of corpora into the training process. In each epoch, our model randomly select one dataset i. We use training set i to learn the parameters and developing set i to evaluate the performance. If the current model achieves the best performance for dataset i on the developing set, we will then calculate the precision, recall and F1 on testing set i.

Users can also refer to ```run_lm-lstm-crf.sh``` (single-task model) and ```run_lm-lstm-crf5.sh``` (multi-task model for the 5 datasets mentioned above) for detailed usage.

## Benchmarks

Here we compare our model with recent state-of-the-art models on the five datasets mentioned above. We use F1 score as the evaluation metric.

|Model | [BC2GM](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOBES) | [BC4CHEMD](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC4CHEMD-IOBES) | [BC5CDR](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-IOBES) | [NCBI-disease](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/NCBI-disease-IOBES) | [JNLPBA](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/JNLPBA-IOBES) |
| ------------- |-------------| -----| -----| -----| ---- |
| Dataset Benchmark | 87.21 | 87.39 | 86.76 | 82.90 | 72.55 |
| [Crichton et al. 2016](https://github.com/cambridgeltl/MTL-Bioinformatics-2016) | 84.41 | 83.02 | 83.90 | 80.37 | 70.09 |
| [Lample et al. 2016](https://github.com/glample/tagger) | 86.53 | 86.62 | 86.61 | 84.64 | 73.48 |
| [Ma and Hovy 2016](https://github.com/XuezheMax/LasagneNLP) | 85.27 | 86.43 | 83.24 | 84.04 | 74.40 |
| [Liu et al. 2018](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF) | 87.82 | 87.01 | 85.18 | 85.10 | 74.69 |
| Our Model | **89.06** | **88.29** | **87.43** | **86.37** | **75.19** |


## Prediction
Our ```train_wc.py``` provides an option to directly output the annotation results during the training process by the parameter ````--output_annotation````, i.e.,
```
python3 train_wc.py --train_file [training file 1] [training file 2] ... [training file N] \
                    --dev_file [developing file 1] [developing file 2] ... [developing file N] \
                    --test_file [testing file 1] [testing file 2] ... [testing file N] \
                    --caseless --fine_tune --emb_file [embedding file] --shrink_embedding --output_annotation --word_dim 200 --gpu 0
```

If users do not use ````--output_annotation````, the best performing model during the training process will be saved in ```./checkpoint/```. Using the saved model, ```seq_wc.py``` can be applied to annotate raw text. Its usage can be accessed by command ````python seq_wc.py -h````, and a running command example is provided below:
```
python3 seq_wc.py --load_arg checkpoint/cwlm_lstm_crf.json --load_check_point checkpoint/cwlm_lstm_crf.model --input_file test.tsv --output_file annotate/output --gpu 0
```
The annotation results will be in ```./annotate/```.

The input format is similar to CoNLL, but each line is required to only contain one field, token. For example, an input file could be:

```
In
the
absence
of
shock
,
sepsis
,
or
other
identifiable
causes
of
lactic
acidosis
,
the
severe
anemia
(
hemoglobin
1
.
2
g
/
dl
)
appeared
to
be
the
primary
etiologic
factor
.
```
and the corresponding output is:

```
In O
the O
absence O
of O
shock O
, O
sepsis O
, O
or O
other O
identifiable O
causes O
of O
lactic O
acidosis O
, O
the O
severe O
anemia O
( O
hemoglobin B-GENE
1 I-GENE
. I-GENE
2 I-GENE
g I-GENE
/ I-GENE
dl E-GENE
) O
appeared O
to O
be O
the O
primary O
etiologic O
factor O
. O 
```
