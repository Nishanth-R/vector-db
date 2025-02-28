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
