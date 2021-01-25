import jsonlines
import numpy as np
import pandas as pd
from base import BaseSolver
from utils import RSG_MorphAnalyzer
from MuSeRC import MuSeRC_metrics

class MuSeRCSolver(BaseSolver):

    def __init__(self, path: str, path_valid=None):
        self.passages = []
        self.y_true = []
        self.qa = []
        self.morph = RSG_MorphAnalyzer() # PyMorphy + cashing
        self.c = 0
        self.c_true = 0
        super(MuSeRCSolver, self).__init__(path, path_valid)


    def preprocess(self, path):
        """ preprocess sentences to apply heuristics"""
        self.reshape_dataset(path)
        self.passages = pd.DataFrame(self.passages, columns=['passage'])
        self.passages['lemmas'] = self.morph.lemmantize_sentences(
            self.passages.passage.to_list())


    def get_heuristics(self, answer_len,
                       intersection_len, lemmas_len,
                       heuristic) -> dict:
        """ all heuristics at once or one of them """

        heuristics = {
            "0": {
                "short answer":  answer_len < 4,
                "no overlap": intersection_len == 0,
                "little overlap": intersection_len == 1},

            "1": {
                "long answer": answer_len > 11,
                "much overlap": intersection_len > 6,
                "total overlap": intersection_len == lemmas_len}
                }

        if heuristic != None:
            # return a single heuristic only
            key = list(heuristic.keys())[0]
            value = heuristic[key]

            return({
                key: { # key = "0" or "1"
                      value: heuristics[key][value] # "heuristic name": Boolean
                      }
                    })

        return heuristics


    def heuristics(self, heuristic = None):
        """
            apply heuristics to a dataset
            To check on a single heursitic, pass
                        heuristic = {"label": "heuristic name"}
            to this function
        """
        # generate a 2D array of random labels:
        # total number of questions x maximum number of answers per question
        random_labels = self.generate_matrix(len(self.y_true))

        y_pred = []

        for question_id, batch in enumerate(self.qa):
            # a batch consists of rows with the same question but differen asnwer
            batch = self.preprocess_batch(batch)

            y_pred.append(self.iterate_over_batch(batch, random_labels,
                                                  question_id, heuristic))

        print(f'Heuristics appears for {self.c} samples, {self.c_true} of them correct')
        print(MuSeRC_metrics(y_pred, self.y_true))

        self.reset_counters()
        return


    def reshape_dataset(self, path):
        """ a function from MuSeRC.py modified to solve tasks with heursitics """

        with jsonlines.open(path) as reader:
            lines = list(reader)

        for passage_id, row in enumerate(lines):
            passage, qa, labels = self.reshape_dataset_row(row, passage_id)
            self.passages.append(passage)
            self.qa.extend(qa)
            self.y_true.extend(labels)


    def reshape_dataset_row(self, row, passage_id):
        """ a function from MuSeRC.py modified to solve tasks with heursitics """

        # split passages, labels, and questions+answers
        passage = row["passage"]["text"]
        labels = []
        qa = []

        for line in row["passage"]["questions"]:
            line_answers = []
            line_labels = []

            for answ in line["answers"]:
                line_labels.append(answ.get("label", 0))
                answ = f"{line['question']} {answ['text']}"
                line_answers.append([passage_id, answ])

            labels.append(line_labels)
            qa.append(line_answers)

        return passage, qa, labels


    def measure_intersection(self, lemmas, passage_id) -> int:
        intersection = set(
            lemmas
            ).intersection(
                set(self.passages.iloc[passage_id].lemmas))

        return len(intersection)


    def update_passage_id(self, passage_id: str) -> int:
        return int(re.sub('?', '', passage_id))


    def generate_matrix(self, size):
        # 7 is a maximum number of answer per question in Validate dataset.
        np.random.seed(self.seed)
        random_labels=np.random.choice([0,1], size=size * 7)

        return random_labels.reshape((-1,7))


    def preprocess_batch(self, batch):
        """ prepare a batch of QandA to be interated over """
        batch = pd.DataFrame(batch, columns=['passage_id', 'QA'])
        batch[['question',
                'answer',
                ]] =  batch[ # splits questions and answer
                            'QA'
                            ].str.split(r"(?<=\?)\s",
                                        expand=True)


        batch['answer_tokens'] = batch['answer'].str.split()
        batch['answer_len'] = batch['answer_tokens'].apply(len)
        batch['lemmas'] = self.morph.lemmantize_sentences(
            batch.answer.to_list())

        return batch


    def iterate_over_batch(self, batch, random_labels, question_id, heuristic) -> list:
        line_pred = [] # prediction for all the answers in a batch

        for answer_id, row in batch.iterrows():

            intersection_len = self.measure_intersection(row.lemmas,
                                                            row.passage_id)

            lemmas_len = len(row.lemmas)

            heuristics = self.get_heuristics(row.answer_len,
                                                intersection_len,
                                                lemmas_len,
                                                heuristic)

            if ('1' in heuristics.keys() and
                (True in list(heuristics['1'].values()))):
                self.c += 1
                line_pred.append(1)

                # compares a predicted label with a correct one
                if self.y_true[question_id][answer_id] == 1:
                    self.c_true += 1

            elif ('0' in heuristics.keys() and
                (True in list(heuristics["0"].values()))):
                self.c += 1
                line_pred.append(0) # inserts an opposite label

                # compares a predicted label with a correct one
                if self.y_true[question_id][answer_id] == 0:
                    self.c_true += 1
            else:
                # insert random value from a pre-generated set
                line_pred.append(random_labels[question_id][answer_id])

        return line_pred


    def reset_counters(self):
        self.c = 0
        self.c_true = 0
