python3 train_wc.py --train_file /shared/data/xuan/bioner/LSTM/data/MTL-Bioinformatics-2016/data/BC2GM-IOBES/train.tsv \
                    --dev_file /shared/data/xuan/bioner/LSTM/data/MTL-Bioinformatics-2016/data/BC2GM-IOBES/devel.tsv \
                    --test_file /shared/data/xuan/bioner/LSTM/data/MTL-Bioinformatics-2016/data/BC2GM-IOBES/test.tsv \
                    --caseless --fine_tune --emb_file /shared/data/xuan/bioner/LSTM/data/PubMed-and-PMC-w2v.txt --word_dim 200 --gpu 1 --shrink_embedding --patience 6 --epoch 200
