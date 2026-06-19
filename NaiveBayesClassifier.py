from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict, Counter
import pandas as pd
from MachineLearning import split_data

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_word = re.findall("[a-z0-9]+", text)
    return set(all_word)

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] +=1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)
        return (p_token_spam, p_token_ham)
    
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
            else: 
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
        
        log_pro_diff = log_prob_if_spam - log_prob_if_ham
        return 1.0 / (1.0 + math.exp(-log_pro_diff))

messages = [Message("spam rules", True),
            Message("ham rules", False),
            Message("hello ham", False)]
model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "rules", "ham", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam" : 1, "rules" : 1}
assert model.token_ham_counts == {"ham" : 2, "rules" : 1, "hello" : 1}


df = pd.read_csv("res/spam_ham_dataset.csv")

messages = [Message(row.text, True if row.label == "spam" else False) 
            for row in df.itertuples()]
messages_train, messages_test = split_data(messages, 0.7)
model = NaiveBayesClassifier(k=0.5)
model.train(messages_train)
predictions = [(message, model.predict(message.text)) for message in messages_test]
confusion_matrix = Counter((message.is_spam, predic_spam > 0.8) 
                           for message, predic_spam in predictions)


