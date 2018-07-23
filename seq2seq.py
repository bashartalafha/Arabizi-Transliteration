from __future__ import print_function

import sys
import os
import pandas as pd
import numpy as np
import random

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional, Dropout
from keras.models import Model, load_model

from keras.layers.normalization import BatchNormalization
from models import encoding

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
np.random.seed(1)

from tensorflow import set_random_seed
set_random_seed(1)

def main(s):
    LEARNING_RATE=0.001
    BATCH_SIZE = 64
    HIDDEN_NODES = 256
    NUMBER_OF_LAYERS = 1
    EMB_SIZE= 50
    EPOCHS= 30
    LOSS="categorical_crossentropy"
    OPT = "adam"
    BI = "No"
    DROPOUT = 0.2
    SEED=s

    saver_folder = "logs/Model_check_seed_"+str(SEED)+"_lr_"+str(LEARNING_RATE)+"_batch_"+str(BATCH_SIZE)+"_node_"+str(HIDDEN_NODES)+"_l_"+str(NUMBER_OF_LAYERS)+"_emb_"+str(EMB_SIZE)+"_ep_"+str(EPOCHS)+"_loss_"+LOSS+"_opt_"+OPT+"_bi_"+BI+"_dropout_"+str(DROPOUT)
    if not os.path.exists(saver_folder):
            os.makedirs(saver_folder)

    data = pd.read_excel('Arabizi-Arabic Parallel corpora.xlsx', header=None, skiprows=1)
    data = data.sample(frac=1, random_state=0)

    data_input = [str(s).strip().lower() for s in data[0]]
    data_output = [str(s).strip() for s in data[1]]

    print("Shuffling ..")
    data_src_shuf=[]
    data_trg_shuf=[]

    random.seed(SEED)

    index_shuf = list(range(len(data_input)))
    random.shuffle(index_shuf)

    for i in index_shuf:
        data_src_shuf.append(data_input[i])
        data_trg_shuf.append(data_output[i])

    data_input = data_src_shuf
    data_output = data_trg_shuf

    print("Removing dublicate")
    data_input_uniq=[]
    data_output_uniq=[]
    dict = {}
    for i,o in zip(data_input, data_output):
        if i not in dict:
            dict[i]=1
            data_input_uniq.append(i)
            data_output_uniq.append(o)
    print("Data size with dublicate",len(data_input))
    print("Data size without dublicate",len(data_input_uniq))
    print(data_input_uniq[0], data_output_uniq[0])
    print(data_input_uniq[5], data_output_uniq[5])

    data_size = len(data_input_uniq)

    # We will use the first 0-60th %-tile (60%) of data for the training
    range_of_train = int(data_size*80/100)

    training_input  = data_input_uniq[0: range_of_train ]
    training_output = data_output_uniq[0: range_of_train ]

    # We will use the first 60-70th %-tile (10%) of data for the training
    range_of_train = int(data_size*80/100)
    range_of_val = int(data_size*90/100)

    validation_input = data_input_uniq[range_of_train:range_of_val]
    validation_output = data_output_uniq[range_of_train: range_of_val]

    print('training size', len(training_input))
    print('validation size', len(validation_input))

    ration =range_of_val + 100
    testing_input = data_input_uniq[range_of_val:ration]
    testing_output = data_output_uniq[range_of_val:ration]

    print('testing size', len(testing_input))

    INPUT_LENGTH=0
    OUTPUT_LENGTH=0
    for i in data_input_uniq:
        if len(i)>INPUT_LENGTH:
            INPUT_LENGTH=len(i)
    for i in data_output_uniq:
        if len(i)>OUTPUT_LENGTH:
            OUTPUT_LENGTH=len(i)
    print("max input len=", INPUT_LENGTH,"\nmax output len=",OUTPUT_LENGTH)

    input_encoding, input_decoding, input_dict_size = encoding.build_characters_encoding(data_input_uniq)
    output_encoding, output_decoding, output_dict_size = encoding.build_characters_encoding(data_output_uniq)

    print('English character dict size:', input_dict_size)
    print('Arabic character dict size:', output_dict_size)

    encoded_training_input = encoding.transform(
        input_encoding, training_input, vector_size=INPUT_LENGTH)
    encoded_training_output = encoding.transform(
        output_encoding, training_output, vector_size=OUTPUT_LENGTH)

    print('encoded_training_input', encoded_training_input.shape)
    print('encoded_training_output', encoded_training_output.shape)

    encoded_validation_input = encoding.transform(
        input_encoding, validation_input, vector_size=INPUT_LENGTH)
    encoded_validation_output = encoding.transform(
        output_encoding, validation_output, vector_size=OUTPUT_LENGTH)

    print('encoded_validation_input', encoded_validation_input.shape)
    print('encoded_validation_output', encoded_validation_output.shape,"\n")

    encoder_input = Input(shape=(INPUT_LENGTH,))
    decoder_input = Input(shape=(OUTPUT_LENGTH,))

    from keras.layers import SimpleRNN

    encoder = Embedding(input_dict_size, EMB_SIZE, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
    if BI == "Yes":
        print("Adding Bi-LSTM to Encoder")
        encoder = Bidirectional(LSTM(HIDDEN_NODES, return_sequences=True, unroll=True))(encoder)

    for i in range(NUMBER_OF_LAYERS):
        print("Adding LSTM #"+str(i)+" to Encoder")
        encoder = LSTM(HIDDEN_NODES, return_sequences=True, unroll=True)(encoder)
        #encoder = BatchNormalization()(encoder)
        encoder = Dropout(DROPOUT)(encoder)
    encoder_last = encoder[:,-1,:]

    print('encoder', encoder)
    print('encoder_last', encoder_last,"\n")

    decoder = Embedding(output_dict_size, EMB_SIZE, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
    for i in range(NUMBER_OF_LAYERS):
        print("Adding LSTM #"+str(i)+" to Decoder")
        decoder = LSTM(HIDDEN_NODES, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])
        #decoder= BatchNormalization()(decoder)
        decoder = Dropout(DROPOUT)(decoder)

    print('decoder', decoder, "\n")

    from keras.layers import Activation, dot, concatenate

    # Equation (7) with 'dot' score from Section 3.1 in the paper.
    # Note that we reuse Softmax-activation layer instead of writing tensor calculation
    attention = dot([decoder, encoder], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)
    print('attention', attention)

    context = dot([attention, encoder], axes=[2,1])
    print('context', context)

    decoder_combined_context = concatenate([context, decoder])
    print('decoder_combined_context', decoder_combined_context)

    # Has another weight + tanh layer as described in equation (5) of the paper
    output = TimeDistributed(Dense(HIDDEN_NODES, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)
    print('output', output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])

    from keras import optimizers
    opt = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    training_encoder_input = encoded_training_input
    import pdb; pdb.set_trace()
    training_decoder_input = np.zeros_like(encoded_training_output)
    training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
    training_decoder_input[:, 0] = encoding.CHAR_CODE_START
    training_decoder_output = np.eye(output_dict_size)[encoded_training_output.astype('int')]

    validation_encoder_input = encoded_validation_input
    validation_decoder_input = np.zeros_like(encoded_validation_output)
    validation_decoder_input[:, 1:] = encoded_validation_output[:,:-1]
    validation_decoder_input[:, 0] = encoding.CHAR_CODE_START
    validation_decoder_output = np.eye(output_dict_size)[encoded_validation_output.astype('int')]

    history = None
    if os.path.isfile(saver_folder+'/model.h5'):
        model = load_model(saver_folder+'/model.h5')
    else:
        history = model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
            validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
            verbose=2, batch_size=BATCH_SIZE, epochs=EPOCHS)

    #model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
    #         validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
    #        verbose=2, batch_size=BATCH_SIZE, epochs=5)

    #model.save(saver_folder+'/model.h5')

    def generate(text):
        encoder_input = encoding.transform(input_encoding, [text.lower()], INPUT_LENGTH)
        decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
        decoder_input[:,0] = encoding.CHAR_CODE_START
        for i in range(1, OUTPUT_LENGTH):
            output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
            decoder_input[:,i] = output[:,i]
        return decoder_input[:,1:]

    def decode(decoding, sequence):
        text = ''
        for i in sequence:
            if i == 0:
                break
            text += output_decoding[i]
        return text

    def to_arabic(text):
        decoder_output = generate(text)
        return decode(output_decoding, decoder_output[0])


    common_american_names = ['zalameh', 'm7mad', 'a5bark', '7elo', 'kefak', 'tamam']
    for name in common_american_names:
        print(name, to_arabic(name))


    print('testing size', len(testing_input))
    ref = open(saver_folder+"/ref.txt", "w")
    pred = open(saver_folder+"/pred.txt", "w")

    import sys
    sum_of_trues=0

    i=0
    for s,t in zip(testing_input,testing_output):
        sys.stdout.write(str(i)+' ')
        ref.write(" ".join(t)+"\n")
        i+=1
        p = to_arabic(s)
        pred.write(" ".join(p)+"\n")
    # print(s,"=>",to_arabic(s), (to_arabic(s) == t))
        if p == t:
            sum_of_trues+=1

    res_file = open(saver_folder+"/results.txt", "w")
    res_file.write("Accuracy= "+str(sum_of_trues/(len(testing_input)) ))
    print("\nAccuracy=",sum_of_trues/(len(testing_input)))
    ref.close()
    pred.close()

    from bleu import moses_multi_bleu
    ref_arr = [i.strip() for i in open(saver_folder+"/ref.txt").readlines()]
    pred_arr = [i.strip() for i in open(saver_folder+"/pred.txt").readlines()]

    res_file.write("\nBleu= "+ str(moses_multi_bleu(ref_arr, pred_arr)))
    res_file.close()

i=51
while (i<1000):
    print("\n\nSeed",i)
    main(i)
    i+=2
