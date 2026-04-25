import pandas as pd

# Load CSV file
df = pd.read_csv("chatbot_data.csv")

def get_response(user_query):

    user_query = user_query.lower()

    for index, row in df.iterrows():
        if row["Query"].lower() == user_query:
            return row["Response"]

    return "Sorry, I didn't understand your query."
