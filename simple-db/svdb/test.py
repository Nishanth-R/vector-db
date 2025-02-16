from app import AppFlow

app = AppFlow()
app.write_to_db_from_url('https://www.geeksforgeeks.org/python-typer-module/')
app.write_to_db_from_url('https://www.geeksforgeeks.org/top-10-system-design-interview-questions-and-answers/')
app.write_to_db_from_url('https://www.geeksforgeeks.org/how-to-design-a-rate-limiter-api-learn-system-design/')
app.write_to_db_from_url('https://www.geeksforgeeks.org/system-design-url-shortening-service/')
article = app.find_closest_articles_by_text('lmao yeet')
print('Closest article')
print('------------------------------------+++++++++++++++++++++++++++++++++++----------------')
print(app.database.convert_vector_to_text(app.get_text_from_row(article)))