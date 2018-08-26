import numpy as np

# EXAMPLES = ['mar7aba', 'kefak', 'sho', 'zalameh']
EXAMPLES_tmp = open("utils/test.csv").readlines()
EXAMPLES = [i.strip() for i in EXAMPLES_tmp]

REF_tmp = open("utils/test_ref.csv").readlines()
REF = [i.strip() for i in REF_tmp]

def run_example(model, input_vocabulary, output_vocabulary, text):
    encoded = input_vocabulary.string_to_int(text)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)

def run_examples(model, input_vocabulary, output_vocabulary, examples=EXAMPLES):
    refs = REF
    predicted = []
    true_count = 0
    count = 0
    for example, ref in zip(examples, refs):
        predicted.append(''.join(run_example(model, input_vocabulary, output_vocabulary, example)))
        answer = predicted[-1][0:predicted[-1].index("<")]
        if count <30:
            print('input:',example)
            print('output:',answer)

        if answer == ref:
            true_count +=1
        count+=1
    acc = float(true_count)/count
    print("Accuracy =", str(acc))
    return predicted
