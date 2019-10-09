from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


def process_google_result(text):
    document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_entity_sentiment(document=document)

    for entity in sentiment.entities:
        clean_entity = "".join(filter(str.isalpha, entity)).lower()

        if clean_entity in ["police", "officer", "cop", "officers", "pigs"]:

            for mention in entity.mentions:
                return mention.sentiment.score


def clean_html_text(html_text):
    return "".join(filter(str.isalpha, html_text)).lower()


class SentimentGoogler:
    def __init__(self):
        self.client = self.connect_to_client()

    def connect_to_client(self):
        return language.LanguageServiceClient()

    @staticmethod
    def pre_process(html_text):
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
        cleaned_doc_text = self.pre_process(doc_text)
        document = types.Document(
            content=cleaned_doc_text, type=enums.Document.Type.PLAIN_TEXT
        )
        sentiment = self.client.analyze_entity_sentiment(document=document)

        return sentiment

    @staticmethod
    def is_police_entity(sentiment_response):
        possible_responses = [
            "police",
            "officer",
            "cop",
            "officers",
            "pigs",
            "policeofficer",
        ]
        for entity in sentiment_response.entities:
            if clean_html_text(clean_entity) in possible_responses:
                return entity
            for mention in entity.mentions:
                if clean_html_text(mention.text.content) in possible_responses:
                    return entity
                return False
