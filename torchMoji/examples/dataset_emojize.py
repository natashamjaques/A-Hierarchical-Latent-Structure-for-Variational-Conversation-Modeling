# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a dataset.
    Save the softmax outputs to a file.
    Print first 10 sentences and their top 5 emojis as example.
"""

from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import argparse

import numpy as np
import emoji
import pickle

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# Emoji map in emoji_overview.png
EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--filepath', type=str, default='/mas/u/asma_gh/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling/datasets/cornell/train/sentences.pkl', required=False, help="Dataset file path")
    argparser.add_argument('--debuglen', type=int, default=10, help="Number of the sentences to show the output for")
    argparser.add_argument('--step', type=int, default=100, help="Number of steps for saving output")
    argparser.add_argument('--maxlen', type=int, default=1000, help="Max length of input text")
    args = argparser.parse_args()
    sentence_probs = []
    output_path = args.filepath[:-4]+'_emojis.pkl'

    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, args.maxlen)

    # Loading model
    model = torchmoji_emojis(PRETRAINED_PATH)

    sentences = pickle.load(open(args.filepath, 'rb'))
    for idx, turn in enumerate(sentences):
        flat_list = [token for line in turn for token in line]
        sentence = " ".join(flat_list)
        # Running predictions
        tokenized, _, _ = st.tokenize_sentences([sentence])
        # Get sentence probability
        prob = model(tokenized)[0]
        sentence_probs += [list(prob)]
        if (idx < args.debuglen):
            print (max(prob))
            print (prob)
            # Top emoji id
            emoji_ids = top_elements(prob, 1)

            # map to emojis
            emojis = map(lambda x: EMOJIS[x], emoji_ids)

            print(emoji.emojize("{} {}".format(sentence,' '.join(emojis)), use_aliases=True))
        if (idx % args.step == 0):
            print ('Step: ', idx+1)
            pickle.dump(sentence_probs, open(output_path, 'wb'))
    print('Step: ', idx + 1)
    pickle.dump(sentence_probs, open(output_path, 'wb'))
