## Models for Nature Language Inference (NLI).

We try to reproduct some classical models in literal papers for Nature Language Inferece, and show performance on Stanford Natural Language Inference data set ([SNLI](https://nlp.stanford.edu/projects/snli/)). 

## Models
- decompose: A Decomposable Attention Model for Natural Language Inference, [paper](http://www.aclweb.org/anthology/D16-1244)
- To be continued ...

## Environment
- TensorFlow 1.3 or higher
- Python 3.5
- Numpy
- Sklearn

## Data preparation
nliutils.py can be used for data preparation
- build_vocab(): Build vocabulary according the training data
- load_vocab(): Load vocabulary from file
- process_file(): Prepare data for model, including converting words into indexes according the vocabulary, padding sentences into fix length, creating the corresponding mask arrays, and loading the classification labels of data into a 1-D array
- 

## Results
Model          | Acc reported in papers  | Our Acc
------------   | -------------           | -------------
decompose      | 86.3%                   | 86.19%


## Hyper-parameters
