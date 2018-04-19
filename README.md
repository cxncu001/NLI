## Models for Nature Language Inference (NLI).

We are trying to reproduce some classical models in literal papers for Nature Language Inferece, and report performance on the Stanford Natural Language Inference data set ([SNLI](https://nlp.stanford.edu/projects/snli/)). 

## Models
- decompose: A Decomposable Attention Model for Natural Language Inference, [paper](http://www.aclweb.org/anthology/D16-1244)
- To be continued ...

## Environment
- TensorFlow 1.3 or higher
- Python 3.5
- Numpy
- Sklearn

## Data preparation
nliutils.py can be used for data preparation.
- build_vocab(): Build vocabulary according the training data.
- load_vocab(): Load vocabulary from file.
- convert_data(): Convert NLI data from 'JSON' format to the following 'TXT' format: gold_label ||| sentence1 ||| sentence2.
- process_file(): Prepare data for model, including converting words into indexes according the vocabulary, padding sentences into fix length, creating the corresponding mask arrays, and loading the classification labels of data into a 1-D array.
- batch_iter(): Generate a batch of data.
- convert_embeddings(): Convert embeddings from TXT (one word embedding per line) to a easy-to-use format in Python, which consists of a 2-d numpy array for embeddings and a dictionary for vocabulary.
- pre-trained word embeddings: You can download pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/), and use convert_embeddings() to get needed format in the code.

## Results
Model          | Acc reported in papers  | Our Acc
------------   | -------------           | -------------
decompose      | 86.3%                   | 86.19%


## Hyper-parameters
- decompose: 

Train model: python3 decompose/train.py --embeddings ../../res/embeddings/glove.840B.300d.we --train_em 0 -op adagrad -lr 0.05 --require_improvement 10000000 --vocab ../cdata/snli/vocab.txt -ep 500 --normalize 1 -l2 0.0 -bs 8 --report 8000 --save_per_batch 8000 -cl 100

Test model: python3 decompose/test.py -m modelfile -d testdata
