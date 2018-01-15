# Cross-type Biomedical Name Tagger with Deep Multi-task Learning

This project provides a neural network based multi-task learning framework for biomedical named entity recognition (BioNER).

The implementation is based on the PyTorch library. Our model collectively trains different biomedical entity types to build a unified model that benefits the training of each single entity type and achieves a significantly better performance compared with the state-of-the-art BioNER systems.

## Quick Links

- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Benchmarks](#benchmarks)
- [Pretrained model](#pretrained-model)

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

## Usage

```train_wc.py``` and ```eval_wc.py``` are scripts for our multi-task LSTM-CRF model.
The usages of these scripts can be accessed by the parameter ````-h````, i.e., 
```
python train_wc.py -h
python eval_wc.py -h
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

Here we compare LM-LSTM-CRF with recent state-of-the-art models on the CoNLL 2000 Chunking dataset, the CoNLL 2003 NER dataset, and the WSJ portion of the PTB POS Tagging dataset. All experiments are conducted on a GTX 1080 GPU.

### NER

When models are only trained on the CoNLL 2003 English NER dataset, the results are summarized as below.

|Model | Max(F1) | Mean(F1) | Std(F1) | Reported(F1) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| [Lample et al. 2016](https://github.com/glample/tagger) | 91.14 | 90.76 | 0.08 | 90.94 | 46 |
| [Ma et al. 2016](https://github.com/XuezheMax/LasagneNLP) | 91.67 | 91.37 | 0.17 | 91.21 | 7 |
| LM-LSTM-CRF | **91.85** | **91.71** | 0.10 | | 6 |

### POS

When models are only trained on the WSJ portion of the PTB POS Tagging dataset, the results are summarized as below.

|Model | Max(Acc) | Mean(Acc) | Std(Acc) | Reported(Acc) | Time(h) |
| ------------- |-------------| -----| -----| -----| ---- |
| [Lample et al. 2016](https://github.com/glample/tagger) | 97.51 | 97.35 | 0.09 | N/A | 37 |
| [Ma et al. 2016](https://github.com/XuezheMax/LasagneNLP) | 97.46 | 97.42 | 0.04 | 97.55 | 21 |
| LM-LSTM-CRF | **97.59** | **97.53** | 0.03 | | 16 |

### Chunking

When models are only trained on the CoNLL 2000 Chunking dataset, the results are summarized as below.

|Model | Max(F1) | Mean(F1) | Std(F1) | Time(h) |
| ------------- |-------------| -----| -----| ----|
| [Lample et al. 2016](https://github.com/glample/tagger) | 94.49 | 94.37 | 0.07 | 26 |
| [Ma et al. 2016](https://github.com/XuezheMax/LasagneNLP) | 95.93 | 95.80 | 0.13 | 6|
| LM-LSTM-CRF | **96.13** | **95.96** | 0.08 | 5 |


## Pretrained Model

### Evaluation
We released pre-trained model on these three tasks. The checkpoint file can be downloaded at:

 
| CoNLL03 NER | WSJ-PTB POS Tagging | CoNLL00 Chunking|
| ------------ | -------------------| -----------------|
| [Args](https://drive.google.com/file/d/0B587SdKqutQmN1UwNjhHQkhUWEk/view?usp=sharing) | [Args](https://drive.google.com/a/illinois.edu/file/d/0B587SdKqutQmN1UwNjhHQkhUWEk/view?usp=sharing) | [Args](https://drive.google.com/file/d/0B587SdKqutQmYmpiNFp6b1hKWEE/view?usp=sharing) |
| [Model](https://drive.google.com/file/d/0B587SdKqutQmSDlJRGRNandhMGs/view?usp=sharing) | [Model](https://drive.google.com/a/illinois.edu/file/d/0B587SdKqutQmSDlJRGRNandhMGs/view?usp=sharing) | [Model](https://drive.google.com/file/d/0B587SdKqutQmNnR3Nnk1WHdIMG8/view?usp=sharing) |

Also, ```eval_wc.py``` is provided to load and run these checkpoints. Its usage can be accessed by command ````python eval_wc.py -h````, and a running command example is provided below:
```
python eval_wc.py --load_arg checkpoint/ner/ner_4_cwlm_lstm_crf.json --load_check_point checkpoint/ner_ner_4_cwlm_lstm_crf.model --gpu 0 --dev_file ./data/ner/testa.txt --test_file ./data/ner/testb.txt
```
### Prediction

To annotated raw text, ```seq_wc.py``` is provided to annotate un-annotated text. Its usage can be accessed by command ````python seq_wc.py -h````, and a running command example is provided below:
```
python seq_wc.py --load_arg checkpoint/ner/ner_4_cwlm_lstm_crf.json --load_check_point checkpoint/ner_ner_4_cwlm_lstm_crf.model --gpu 0 --input_file ./data/ner2003/test.txt --output_file output.txt
```

The input format is similar to CoNLL, but each line is required to only contain one field, token. For example, an input file could be:

```
-DOCSTART-

But
China
saw
their
luck
desert
them
in
the
second
match
of
the
group
,
crashing
to
a
surprise
2-0
defeat
to
newcomers
Uzbekistan
.
```
and the corresponding output is:

```
-DOCSTART- -DOCSTART- -DOCSTART-

But <LOC> China </LOC> saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers <LOC> Uzbekistan </LOC> . 

```

## Reference

```
@inproceedings{2017arXiv170904109L,
  title = "{Empower Sequence Labeling with Task-Aware Neural Language Model}", 
  author = {{Liu}, L. and {Shang}, J. and {Xu}, F. and {Ren}, X. and {Gui}, H. and {Peng}, J. and {Han}, J.}, 
  booktitle={AAAI},
  year = 2018, 
}
```
