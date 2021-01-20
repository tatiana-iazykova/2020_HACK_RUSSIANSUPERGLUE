import re
from base import BaseSolver
from utils import RSG_MorphAnalyzer
from scipy import stats
from sklearn.metrics import matthews_corrcoef

class LiDiRusSolver(BaseSolver):

    def __init__(self, path: str, path_valid=None):
        self.e_words = {"чтобы", 'будет', "от", "он"} # -> entailment
        self.ne_words = {'и', "не", "никогда", "вовсе", 'что', "это"} # -> not_eintailment
        self.morph = RSG_MorphAnalyzer() # PyMorphy + cashing
        super(LiDiRusSolver, self).__init__(path, path_valid)

    def preprocess(self):
        self.cashe = {} # create a dictionary for lemmas
        """ preprocess sentences to apply heuristics"""
        self.train["sentence1_words"] = self.train['sentence1'].str.split()
        self.train["sentence2_words"] = self.train['sentence2'].str.split()
        self.train["sentence1_lemmas"] = self.morph.lemmantize_sentences(
            self.train.sentence1.to_list())
        self.train["sentence2_lemmas"] = self.morph.lemmantize_sentences(
            self.train.sentence2.to_list())


    def get_heuristics(self, non_intersect, intersect, non_intersect_lemmas, heuristic) -> dict:
        """ all heuristics at once or one of them """

        heuristics = {
            "not_entailment": {
                "little overlap": len(non_intersect) > 10,
                "some overlap": len(intersect) < 6,

                # catches if there is an extra clause inside
                "extra clause": len(re.findall(r",", " ".join(non_intersect))) > 1,

                "keyword": len(non_intersect) == 2,

                # negated word, e.g: необычный, незапланированно
                "negated words": re.match(r"не\w+", " ".join(non_intersect)) != None ,

                # has one of the words from the list
                "wordlist": len(self.ne_words.intersection(non_intersect)) > 0},

            "entailment": {
                "all lemmas overlap": len(non_intersect_lemmas) == 0,

                "wordlist": len(self.e_words.intersection(intersect)) > 0}}

        if heuristic != None:
            # return a single heuristic only
            key = list(heuristic.keys())[0]
            value = heuristic[key]

            return({
                key: { # key = "entailment" or "not_entailment"
                      value: heuristics[key][value] # "heuristic name": Boolean
                      }
                    })

        return(heuristics)


    def heuristics(self, final_decision = None, heuristic = None):
        """
            apply heuristics to a dataset

            To check on a single heursitic, pass
                        heuristic = {"label": "heuristic name"}
            to this function
        """

        test_size = len(self.train.label)
        random_labels = final_decision(test_size=test_size)

        y_true = self.train.label
        y_pred = []

        c = 0 # heuristics counter
        c_true = 0 # valid heuristics counter

        for i, row in self.train.iterrows():

            sentence1 = row['sentence1_words']
            sentence2 = row['sentence2_words']

            non_intersect = set(sentence1) ^ set(sentence2)
            intersect = set(sentence1).intersection(sentence2)
            lemmas_non_intersect = set(row.sentence1_lemmas) ^ set(row.sentence2_lemmas)

            heuristics = self.get_heuristics(non_intersect,
                                             intersect,
                                             lemmas_non_intersect,
                                             heuristic)


            if ('entailment' in heuristics.keys() and
                (True in list(heuristics['entailment'].values()))):
                    c += 1
                    y_pred.append('entailment')

                    if row.label == 'entailment': # compares a predicted label with a correct one
                        c_true += 1
            elif ('not_entailment' in heuristics.keys() and
                (True in list(heuristics['not_entailment'].values()))):
                c += 1
                y_pred.append('not_entailment') # inserts an opposite label

                if row.label == 'not_entailment': # compares a predicted label with a correct one
                    c_true += 1
            else:
                y_pred.append(random_labels[i]) # insert random

        print(f'Heuristics appears for {c} samples, {c_true} of them correct')
        print(self.show_mc(y_true, y_pred))

    def show_mc(self, y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)
