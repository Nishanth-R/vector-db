from traceback import format_exception

from database import Database
from errors import FetchFailure

import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from helper import filter_stopwords_encoded, filter_stopwords_in_text



class AppFlow:
    def __init__(self):
        self.database = Database()
        self.vocabulary = self.database.load_bow()

    @staticmethod
    def fetch_from_url(url):
        try:
            headers = {'Accept-Encoding': 'gzip, deflate'}
            response = requests.get(url, verify=False, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            article_elements = soup.find_all("article")
            if not article_elements:
                article_elements = soup.find_all("section",
                                                 attrs={"class": "post-content"})
            if not article_elements:
                article_elements = soup.find_all("div", attrs={"class": "article-content"})
            if not article_elements:
                article_elements = soup.find_all("div", attrs={"itemprop": "articleBody"})

            if article_elements:
                article_content = ""
                for element in article_elements:
                    for p in element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):  # Add more tags as needed
                        article_content += p.get_text().strip() + "\n"
                return article_content.strip()
            else:
                all_p_tags = soup.find_all("p")
                if all_p_tags:
                    all_p_content = ""
                    for p in all_p_tags:
                        all_p_content += p.get_text().strip() + "\n"
                    return all_p_content.strip()
                else:
                    print("Could not find article elements on the page.  Inspect website HTML.")
                    return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            raise FetchFailure(f' Error occured during fetching from the Internet - {e}')

    def get_text_from_row(self,row):
        row = row.split(self.database.internal_delimiter)
        text = row[-1].split()
        return list(map(int, text ))

    @staticmethod
    def pad_arrays(vector1,vector2):
        len1 = vector1.shape[1]
        len2 = vector2.shape[1]
        if len1 < len2:
            padding = np.zeros((1, len2 - len1))
            vector1 = np.concatenate([vector1, padding], axis=1)
        elif len2 < len1:
            padding = np.zeros((1, len1 - len2))
            vector2 = np.concatenate([vector2, padding], axis=1)
        return vector1, vector2

    def find_closest_articles_by_text(self, text):
        text_list = self.database.query_all_document()
        text_filtered = filter_stopwords_in_text(text)
        if not text_filtered:
            return None
        similarities = []
        for document in text_list:
            other_text = self.get_text_from_row(document)
            other_text_filtered = filter_stopwords_encoded(other_text)
            if not other_text_filtered:
                similarity = 0
            else:
                try:
                    text_vector = np.array(self.database.encode_text(text_filtered)).reshape(1, -1)
                    other_text_vector = np.array(other_text_filtered).reshape(1, -1)
                    if text_vector.size == 0:
                        text_vector = np.zeros((1, 1))
                    if other_text_vector.size == 0:
                        other_text_vector = np.zeros((1, 1))

                    text_vector, other_text_vector = self.pad_arrays(text_vector,other_text_vector)

                    similarity = cosine_similarity(text_vector, other_text_vector)[0][0]
                except ValueError as err:
                    print(format_exception(err))
                    print(err.__str__())
                    return -1
            similarities.append(similarity)
        if not similarities:
            return None
        closest_index = np.argmax(similarities)
        return text_list[closest_index]

    def write_to_db_from_url(self, url):
        text = self.fetch_from_url(url)
        self.database.interface_insert(text)

