## Introduction 

This is an implementation of the world's simplest vector database. 
It is a simple key-value store that stores vectors of integers. 
The database supports the following operations:
- Insert 
- Encode 
- Load BOW 
- Dump BOW 

The inside is basically a pickle file which stores the vectors. 
The data is stored as a string and is delimited using an internal delimiter. 

## Data structure 
Data is stored in a list where each element is a string with the following structure,
 DB-id||generated-uuid||Encoded data

## Encoding 
Simple DB as the name suggests uses a simple encoding scheme, it uses a Bag of Words (BOW) encoding - https://en.wikipedia.org/wiki/Bag-of-words_model. 
The encoding is done using the following steps:
- Tokenize the input string - Currently, the tokenizer is a simple split on space
- Create a dictionary of the tokens - Gets better when it sees more data
- Encode the tokens using the dictionary
- Return the encoded string

## Appplication 
The application is a simple command line interface that allows the user to interact with the database. It wraps aroune the database logic and 
provides a simple interface to the user. It allows the user to find the closest document to entered text. Application also allows insert based on text from a file or from an url. 


## Usage 
&nbsp;

`closest`  - Fetch the closest document indexed to the text entered. It currently supports only top 1 document. Top n documents will be another  
             command. 
            *param* text - Text to compare the documents against. :return: Text present in the closest document.
&nbsp;

`insert-file`  Insert text contents of the entered Filename into the database. Please note that the filename should also contain the file          
             extensions. Currently, there is support only for .txt and pickle files ending in .pkl or .pickle *param* filename: Filename to write 
             to database <Extensions are mandatory> 
&nbsp;

`insert-url`   Insert text contents of the entered url into the database. :param url: URL which should be inserted into the "Database" 