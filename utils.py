from string import punctuation
from pymorphy2 import MorphAnalyzer
from razdel import tokenize as razdel_tokenize

class RSG_MorphAnalyzer():

    def __init__(self):
        self.morpho = MorphAnalyzer()
        self.cashe = {}

    def lemmantize_sentences(self, sentences):
        """
            receives a list of tokens by sentence
            returns list of lemmas by sentence
        """
        res = []
        for sentence in sentences:
            res.append(self.lemmantize(sentence))

        return(res)

    def lemmantize(self, txt) -> list:
        """
            returns only lemmas
        """

        words = self.tokenize(txt)

        res=[]

        for w in words:
            if w in self.cashe:
                res.append(self.cashe[w])
            else:
                r=self.morpho.parse(w)[0].normal_form
                res.append(r)
                self.cashe[w]=r

        return(res)

    def tokenize(self, txt) -> list:
        """
            tokenizes and removes punctuation from a string
        """
        punkt = punctuation + '«»—…–“”'
        tokens = []

        for word in list(razdel_tokenize(txt)):
            token = word.text.strip(punkt).lower() # remove punctuation
            if token == "": # skip empty elements if any
                continue
            tokens.append(token)

        return(tokens)
