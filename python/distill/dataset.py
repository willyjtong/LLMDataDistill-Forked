from functools import partial

import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator

ch = spacy.load('zh_core_web_sm')
def charTokenize(text):
    return [char for char in text]

def cnTokenize(text):
    return [tok.text for tok in ch(text)]

def extract_attributes(row):
    return row[1], row[2]

def get_tokens(data_iter, tokenizer):
    for _, text in data_iter:
            yield tokenizer(text)

def build_vocab(documents, tokenizer):
    data_pipe = dp.iter.IterableWrapper(documents)
    data_pipe = dp.iter.FileOpener(data_pipe, mode='r')
    data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
    data_pipe = data_pipe.map(extract_attributes)

    vocab = build_vocab_from_iterator(
        get_tokens(data_pipe, tokenizer),
        min_freq=2,
        specials=["<pad>", "<sos>", "<eos>", "<unk>"],
        special_first=True
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# build train pipe and valid pipe
def getTransform(vocab):
    text_transform = T.Sequential(
        T.VocabTransform(vocab),
        T.AddToken(1, begin=True),
        T.AddToken(2, begin=False)
    )
    return text_transform

def apply_transform(sample, vocab):
    text_transformer = getTransform(vocab)
    tokenized_text = charTokenize(sample[1])
    return text_transformer(tokenized_text), [1. if float(sample[0]) >= 30 else 0.]

def sortBucket(bucket):
    return sorted(bucket, key=lambda x: len(x[0]))

def separate_batch(batch):
    '''
    Inputs: [(text1, label1), (text2, label2), ...]
    Outputs: ([text1, text2, ...], [label1, label2, ...])
    '''
    texts, labels = zip(*batch)
    return texts, labels

def apply_padding(sample):
    return (T.ToTensor(0)(list(sample[0])), torch.tensor(list(sample[1])))

def build_data_pipe(data_path, vocab, batch_size=16):
    data_pipe = dp.iter.IterableWrapper([data_path])
    data_pipe = dp.iter.FileOpener(data_pipe, mode='r')
    data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
    data_pipe = data_pipe.map(extract_attributes)
    
    apply_transform_with_vocab = partial(apply_transform, vocab=vocab)
    data_pipe = data_pipe.map(apply_transform_with_vocab)
    print('Applied transform...')
    #for sample in data_pipe:
    #    print(sample)
    #    break

    data_pipe = data_pipe.bucketbatch(
        batch_size = batch_size, 
        batch_num=5,  # batch_num is the number of batches to keep in a bucket
        bucket_num=1, # bucket_num is the number of buckets
        use_in_batch_shuffle=False, 
        sort_key=sortBucket
    )

    print('Afte batch ...')
    #for batch in data_pipe:
    #    print(batch)
    #    print(len(batch))
    #    break

    data_pipe = data_pipe.map(separate_batch)
    print('After seperate batch ...')
    #for texts, labels in data_pipe:
    #    print(len(texts), texts)
    #    print(len(labels), labels)
    #    break


    data_pipe = data_pipe.map(apply_padding)
    print('After apply padding ...')
    return data_pipe

