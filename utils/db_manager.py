import os 
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

DB_NAME=os.getenv("DB_NAME")
POSTGRES_LIB_USER=os.getenv("POSTGRES_LIB_USER")
POSTGRES_LIB_PASS=os.getenv("POSTGRES_LIB_PASS")
POSTGRES_LIB_HOST=os.getenv("POSTGRES_LIB_HOST")
POSTGRES_LIB_PORT=os.getenv("POSTGRES_LIB_PORT")


class DBManager:
    def __init__(self, connection):
        self.conn = connection
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def fetch_reviews(self):
        query = """
        SELECT id_acc, response, checkoutdate, checkindate, language, title, context, review_date, id, rev_id
        FROM public.reviews_new LIMIT 100;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def read(self, table, criteria: dict):
        pass

    def update(self, table, record_id, updates: dict):
        pass

    def delete(self, table, record_id):
        pass

if __name__ == "__main__":
    

    connection = psycopg2.connect(
        dbname=DB_NAME,
        user=POSTGRES_LIB_USER,
        password=POSTGRES_LIB_PASS,
        host=POSTGRES_LIB_HOST,
        port=POSTGRES_LIB_PORT
    )

    db_manager = DBManager(connection = connection)

    reviews = db_manager.fetch_reviews()
    for review in reviews:
        print(review['id_acc'], review['response'], review['checkoutdate'], review['checkindate'], review['language'], review['title'], review['context'], review['review_date'], review['id'], review['rev_id'])
    
    connection.close()