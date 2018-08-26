"""
  Date Generator
    This code creates data for our date translation model

  References:
    https://github.com/rasmusbergpalm/normalization/blob/master/babel_data.py
    https://github.com/joke2k/faker
    https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

  Contact:
    zaf@datalogue.io (@zafarali)
"""
import random
import json
import os

DATA_FOLDER = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))

# from faker import Faker
# import babel
# from babel.dates import format_date

#fake = Faker()
#fake.seed(230517)
#random.seed(230517)

#FORMATS = ['short',
#            'medium',
#            'long',
#            'full',
#            'd MMM YYY',
#            'd MMMM YYY',
#            'dd MMM YYY',
#            'd MMM, YYY',
#            'd MMMM, YYY',
#            'dd, MMM YYY',
#            'd MM YY',
#            'd MMMM YYY',
#            'MMMM d YYY',
#            'MMMM d, YYY',
#            'dd.MM.YY',
#            ]

# change this if you want it to work with only a single language
# LOCALES = ['en_US']
# LOCALES = babel.localedata.locale_identifiers()


def create_dataset(dataset_name, vocabulary=False):
    """
        Creates a csv dataset with n_examples and optional vocabulary
        :param dataset_name: name of the file to save as
        :n_examples: the number of examples to generate
        :vocabulary: if true, will also save the vocabulary
    """
    human_vocab = set()
    machine_vocab = set()

    data = open(dataset_name, 'r').readlines()
    for i in data:
            h, m = i.strip().split(",")
            if h is not None:
                human_vocab.update(tuple(h))
                machine_vocab.update(tuple(m))

    if vocabulary:
        int2human = dict(enumerate(human_vocab))
        int2human.update({len(int2human): '<unk>',
                          len(int2human)+1: '<eot>'})
        int2machine = dict(enumerate(machine_vocab))
        int2machine.update({len(int2machine):'<unk>',
                            len(int2machine)+1:'<eot>'})

        human2int = {v: k for k, v in int2human.items()}
        machine2int = {v: k for k, v in int2machine.items()}

        with open('human_vocab.json', 'w') as f:
            json.dump(human2int, f)
        with open('machine_vocab.json', 'w') as f:
            json.dump(machine2int, f)

if __name__ == '__main__':
    print('creating dataset')
    create_dataset('train.csv', vocabulary=True)
    create_dataset('test.csv', vocabulary=True)
    print('dataset created.')
