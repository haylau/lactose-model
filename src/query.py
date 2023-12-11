from dotenv import load_dotenv
import os
import psycopg2

# PSQL Connection
DB_CONFIG = psycopg2.connect(
    dbname="food_item_db",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=5432
)

# # Create a cursor object using the connection
# cursor = conn.cursor()


# try:
#     # Execute the query
#     cursor.execute(sql_query)

#     # Fetch all the results
#     rows = cursor.fetchall()

#     # Print the results
#     for row in rows:
#         print(row)

# except (Exception, psycopg2.Error) as error:
#     print("Error fetching data:", error)

# # Close the cursor and connection
# cursor.close()
# conn.close()
