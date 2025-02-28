"""
Pure python implementation of a vector database, will include next to no libraries.
Simple and easy to understand code.
Class definitions -
Database - Implementation of a simple database that will load data to and from disk

Assumptions -
    id - is the only primary key
"""
import traceback
import uuid

from errors import InsertIntoException, LoadingException, InvalidRowError, EncodingError
from traceback import format_exception
import pickle
import os

class Database:
    def __init__(self):
        self.insert = False
        self.filename = os.getcwd()+'/vectors.pickle'
        self.delimiter = '|@|'
        self.internal_delimiter = '||'
        self.bow_filename = os.getcwd()+'/bow.pickle'
        self.current_idx = self.get_current_idx()

    def insert_into(self, data):
        try:
            loaded_list = self.query_all_document()
            loaded_list.append(data)
            with open(self.filename, "wb+") as file:
                pickle.dump(loaded_list, file)
        except Exception as err:
            print(err.__str__())
            raise InsertIntoException('Exception occurred while executing "INSERT INTO" ' +
                                      format_exception(err))

    def query_all_document(self):
        try:
            with open(self.filename, 'rb') as file:
                response = pickle.load(file)
            return response
        except EOFError:
            with open(self.filename, 'wb') as file:
                pickle.dump([], file)
                return []
        except Exception as err:
            print(err.__str__())
            raise LoadingException('Exception occurred while executing SELECT *  statement '
                                   + format_exception(err))

    def query_by_id(self, idx):
        try:
            all_documents = self.query_all_document()
            for document in all_documents.split(self.delimiter):
                for subdoc in document.split(self.internal_delimiter):
                    if subdoc[0] == idx:
                        return document
        except Exception as err:
            print(err.__str__())
            raise LoadingException('Exception occurred while executing "SELECT by id" statement '
                                   + format_exception(exc = err))

    def validate_row(self,data):
        all_documents = self.query_all_document()
        all_ids = []
        for document in all_documents.split(self.delimiter):
            for subdoc in document.split(self.internal_delimiter):
                all_ids.append(subdoc[0])
        row = data.split(self.internal_delimiter)
        if row[0] in all_ids:
            raise InvalidRowError('ID '+str(row[0])+' is already in use.')

    @staticmethod
    def tokenise_row(text):
        if text:
            return text.split(' ')
        return None

    def load_bow(self):
        try:
            with open(self.bow_filename, 'rb') as file:
                return pickle.load(file=file)
        except EOFError:
            with open(self.bow_filename, 'wb') as file:
                pickle.dump({}, file)
                return {}

    def dump_bow(self, bow):
        with open(self.bow_filename, 'wb') as file:
            pickle.dump(bow, file)

    @staticmethod
    def get_word_id(word, bow):
        try:
            return bow[word]
        except KeyError:
            return None

    def encode_text(self, text):
        try:
            if isinstance(text,str):
                words = self.tokenise_row(text)
            elif isinstance(text, list):
                words = text
            else:
                raise ValueError('Invalid datatype sent for encoding.')
            if not words:
                raise ValueError('No text available to encode.')
            tokens = []
            bow = self.load_bow()
            bow_change_flag = False

            for word in words:
                word_id = self.get_word_id(word.lower(), bow)
                if word_id:
                    tokens.append(word_id)
                else:
                    new_word_id = len(bow) + 1
                    tokens.append(new_word_id)
                    bow.update({word.lower().strip():new_word_id})
                    bow_change_flag = True

            #Wow, an optimisation
            if bow_change_flag:
                self.dump_bow(bow)

            return tokens

        except Exception as err:
            print(err.__str__())
            raise EncodingError('Error during encoding process '+
                                traceback.format_exception(err))

    def get_current_idx(self):
        return len(self.query_all_document()) + 1

    def interface_insert(self,text):
        encoded_text = self.encode_text(text)
        encoded_text = ' '.join(str(_) for _ in encoded_text)
        current_idx = self.get_current_idx()
        insert_data = (str(current_idx)+ self.internal_delimiter + str(uuid.uuid4())
                       + self.internal_delimiter + encoded_text)
        self.insert_into(insert_data)

    def convert_vector_to_text(self,vector):
        bow = self.load_bow()
        reverse_bow = {value: key for key, value in bow.items()}
        if isinstance(vector,list): #Extend in the future
            appender = []
            for word in vector:
                appender.append(reverse_bow[word])
            return ' '.join(word for word in appender)
