import os
import lyric_config
import logging
from gensim.models import word2vec
from pipes import quote

import scipy.io.wavfile as wav
import numpy as np
from config import nn_config
import parse_files

import config.nn_config as nn_config

conf = lyric_config.lyric_conf()
def getdict(train = False):
		binfilename = conf['emb_binfile']

		if train:
				logging.info("Training Word Embeddings")
				txtfilename = ""
				cmd = 'third-party/word2vec -train {0} -output {1} -cbow 0 -size {3} -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1'.format(quote(txtfilename), quote(binfilename))
				os.system(cmd)

		return word2vec.Word2Vec.load_word2vec_format(binfilename, binary=True)


def word_to_vec(filename):
		f = open(filename)
		model = getdict()
		sequences = []
		#y_data = np.zeros(out_shape)
		for line in f.readlines():
				curSeq = []
				words = line.rstrip().split()
				if words == []: continue
				for word in words:
						try:
								cur_vect = model[word.lower()]								
						except:
								cur_vect = np.zeros(200)
						curSeq.append(cur_vect)

				sequences.append(curSeq)
		return sequences

def gen_batch(X, sequences, batch_size):
		batch_size = 5
		sequences = sorted(sequences, key = lambda cur_seq: len(cur_seq))
		num_batch = len(sequences) // batch_size
		for i in range(num_batch):
				cur_batch = sequences[i * batch_size: min((i + 1) * batch_size, len(sequences))]
				x_idx = np.random.choice(X.shape[0], len(cur_batch))
				x_batch = X[x_idx]
				yield x_batch, padding(cur_batch)

def padding(batch):
		max_len = len(batch[-1])
		shape = batch[-1][0].shape[0]
		batch_size = len(batch)
		result = np.zeros([batch_size, max_len, shape])
		for cur_seq in batch:
				while len(cur_seq) < max_len:
						cur_seq.append(np.zeros(shape))
		for i in range(batch_size):
				for j in range(max_len):
						result[i,j,] = batch[i][j]
		return result




if __name__ == "__main__":
		config = nn_config.get_neural_net_configuration()
		input_directory = config['dataset_directory']
		output_filename = config['model_file'] 

		freq = config['sampling_frequency'] #sample frequency in Hz
		clip_len = 5 		#length of clips for training. Defined in seconds
		block_size = freq / 4 #block sizes used for training - this defines the size of our input state
		max_seq_len = int(round((freq * clip_len) / block_size)) #Used later for zero-padding song sequences
		batch_size = 20
		#Step 1 - convert MP3s to WAVs
		new_directory = parse_files.convert_folder_to_wav(input_directory, freq)
		#Step 2 - convert WAVs to frequency domain with mean 0 and standard deviation of 1
		result = parse_files.convert_wav_files_to_nptensor(new_directory, block_size, max_seq_len, output_filename)
		#print result
		X, x_mean, x_std = result
		lyric_sequence = word_to_vec(quote("./lyrical-net/data/lyrics/taylorswift/backtodecember.txt"))
		for X, y in gen_batch(X, lyric_sequence, batch_size):
				print X.shape, y.shape










