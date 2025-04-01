import typer
from app import AppFlow
import pickle

app = typer.Typer()
app_flow = AppFlow()

@app.command()
def insert_url(url:str):
    """
    Insert text contents of the entered url into the database.
    :param url: URL which should be inserted into the "Database"
    :return: None
    """
    app_flow.write_to_db_from_url(url)

@app.command()
def insert_file(filename:str):
    """
    Insert text contents of the entered Filename into the database.
    Please note that the filename should also contain the file extensions.
    Currently, there is support only for .txt and pickle files ending in .pkl or .pickle
    :param filename: Filename to write to database <Extensions are mandatory>
    :return:
    """
    if filename.endswith('.txt'):
        with open(filename, 'r') as file:
            text = file.read()
    if filename.endswith('.pkl') or filename.endswith('.pickle'):
        with open(filename, 'rb') as file:
            text = pickle.load(file)

    app_flow.database.interface_insert(text)

@app.command()
def closest(text: str):
    """
    Fetch the closest document indexed to the text entered. It currently supports only
    top 1 document. Top n documents will be another command.
    :param text: Text to compare the documents against.
    :return: Text present in the closest document.
    """
    article = app_flow.find_closest_articles_by_text(text)
    return app_flow.database.convert_vector_to_text(app_flow.get_text_from_row(article))

if __name__ == '__main__':
    app()