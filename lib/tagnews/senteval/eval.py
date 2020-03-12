from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

from tagnews.senteval.police_words import police_words_list, bins


# def process_google_result(text):
#     document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
#     sentiment = client.analyze_entity_sentiment(document=document)
#
#     for entity in sentiment.entities:
#         clean_entity = "".join(filter(str.isalpha, entity)).lower()
#
#         if clean_entity in police_words_list:
#
#             for mention in entity.mentions:
#                 return mention.sentiment.score


class SentimentGoogler:
    def __init__(self):
        self.client = self.connect_to_client()
        self.police_words = police_words_list
        self.bins = bins[::-1] # reversed because we start with lower numbered bins
        self.num_bins = len(bins)

    def run(self, doc_text):
        sentiment_ = self.call_api(doc_text)
        for entity in sentiment_.entities:
            police_entity = self.is_police_entity(entity)
            if police_entity:
                return self.sentiment_from_entity(police_entity)

    def connect_to_client(self):
        return language.LanguageServiceClient()

    def sentiment_from_entity(self, entity):
        return entity.sentiment.score

    def call_api(self, doc_text):
        """
        Parameters
        ----------
        doc_text : str
            article text

        Returns
        -------
        sentiment : json
            google response call
        """
        document = types.Document(content=doc_text, type=enums.Document.Type.PLAIN_TEXT)
        sentiment = self.client.analyze_entity_sentiment(document=document)

        return sentiment

    def is_police_entity(self, entity):
        if entity in self.police_words:
            return entity
        for mention in entity.mentions:
            if pre_process_text(mention.text.content) in self.police_words:
                return entity
            return False

    def extract_google_priority_bin(self, article:str, cpd_model_val=1, cpd_val=1):
        cop_word_counts = sum([article.count(substr) for substr in self.police_words])
        score = 0.5 * cpd_val + 0.25 * cpd_model_val + 0.25 * min(cop_word_counts / (2 * len(self.police_words)), 1.)
        bin = [bin for bin, bin_max_val in enumerate(self.bins) if bin_max_val >= score][-1]
        return bin


def pre_process_text(html_text):
    """
    Parameters
    ----------
    html_text : str
        Article text.

    Returns
    -------
    words: str
        lower case, just letters
    """
    words = "".join(filter(str.isalpha, html_text)).lower()
    return words
