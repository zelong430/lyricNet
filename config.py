#/usr/bin/env python

#music configuration
music_conf = {}
music_conf['sampling_frequency'] = 44100
#Number of hidden dimensions.
#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
music_conf['hidden_dimension_size'] = 1024
#The weights filename for saving/loading trained models
music_conf['model_basename'] = './YourMusicLibraryNPWeights'
#The model filename for the training data
music_conf['model_file'] = './datasets/YourMusicLibraryNP'
#The dataset directory
music_conf['dataset_directory'] = './datasets/YourMusicLibrary/'

#lyric configuration
def lyric_conf():
   conf = {}
   conf['emb_size'] = 128
   conf['lyric_directory'] = './data/lyrics'
   conf['emb_binfile'] = './pretrain/text8-vector.bin'
   return conf
