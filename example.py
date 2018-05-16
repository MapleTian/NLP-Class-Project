from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
import time


def execute_demo(language):
    data = Dataset(language)


    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['target_word'])#sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset,data.testset)

    predictions = baseline.test(data.trainset,data.testset)

    gold_labels = [sent['gold_label'] for sent in data.testset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    start = time.time()
    execute_demo('english')
    execute_demo('spanish')
    end = time.time()
    print("The running time is :",end - start)


