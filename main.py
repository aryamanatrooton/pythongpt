from flask import Flask, request, jsonify, render_template, session
import openai
import os
from dotenv import load_dotenv
from flask_session import Session  # for managing sessions
import requests
from flask_caching import Cache  # for caching responses
import pymongo
import pandas as pd
import json
from bson import ObjectId
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.urandom(24)  # Set a secret key for session encryption
app.config['SESSION_TYPE'] = 'filesystem'  # Session type filesystem
Session(app)  # Initialize the session
cache = Cache(app) # Initializing the cache


# load_dotenv()
project_folder = os.path.expanduser('/home/aryamansingh786/myapp/')  # adjust as appropriate
load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")

context = """
You are an AI developed by OpenAI, powered by GPT-4. Your core function is to engage with users primarily interested in pursuing their studies in Canada and exploring the possibility of applying for Permanent Residency (PR) there. Through a series of strategic questions, you will extract valuable information about their current immigration status and their aspirations towards Canadian PR. Your questions should follow a seamless, back-to-back format that is tailored to the user's specific context.

Based on the user's responses, you will analyze the provided information and generate an estimated percentage of their success in obtaining Canadian PR. It's important to remember that these percentages are not guaranteed outcomes but rather educated predictions based on the data given.

Finally, if the user is a student, you will suggest a selection of courses and colleges in Canada with Level Of Education that align with their academic interests and could potentially enhance their chances of securing PR. These recommendations should be presented at the end of the interaction, following a comprehensive understanding of the user's profile and aspirations, and also always provide a html button in the same message with id="getrecommendation" class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded" to trigger the recommendation.
"""

@app.route('/', methods=['GET', 'POST'])
def chat():
    # referrer = request.headers.get("Referer")
    # if not referrer or 'canadian-course-compass.onrender.com' not in referrer:
    #     return "Forbidden", 403
    if 'conversation' not in session:
        # If conversation doesn't exist in session, create a new one
        session['conversation'] = [
            {"role": "system", "content": context},
        ]

    if request.method == 'POST':
        user_message = request.form['usercontext']
        session['conversation'].append({"role": "user", "content": user_message})
        session.modified = True  # Tell the session it was modified
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=session['conversation'],
            max_tokens=3000
        )
        message = response['choices'][0]['message']['content']
        print(message)
        session['conversation'].append({"role": "system", "content": message})
        session.modified = True  # Tell the session it was modified
        return jsonify({'message': message})
    return render_template('index.html')

@app.route('/get_conversation_history', methods=['GET'])
def get_conversation_history():
    if 'conversation' in session:
        conversation_history = session['conversation']
        return (conversation_history)
    else:
        return jsonify([])
@app.route('/clear_session', methods=['GET'])
def clear_session():
    session.clear()
    return jsonify({'message': 'Session cleared'})



# @app.route('/recommendation',methods=['GET', 'POST'])
# def recommendation():
#     email = request.json.get('email')
#         # Handle the email here
#         # ...
#     return jsonify({'message': 'Email received and processed', 'email': email})
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

@app.route('/recommendation', methods=['GET', 'POST'])
def thingsgettingstarted():
    if request.method == 'POST':
        email = request.json.get('email')
        convo = request.json.get('convo')
        print(type(convo))
        print("email from rooton to .ca --->",email)
        # Handle the email here
        # ...
        courses = fetch_courses_from_database()
        # print(courses)
        specificuser = [fetch_user_details("test","userdetails",email)]
        temp = pd.DataFrame(specificuser)

# print temp and user just for debug
        # print("temp ",temp)

# Convert the DataFrame to a dictionary
        user = temp.to_dict('records')[0]
        # print(type(specificuser))
        # user = pd.DataFrame(specificuser)


        messages = [{
        "role": "system", 
        "content": "Understand this conversation json "+ str(convo) +
        " this is user's detail in json "+ str(specificuser) +
        " now do analysis and provide a list of courses through which this particular user can get PR in Canada. in list additionally include if any courses are recommended in conversation json"
        },
        {
        "role": "user", 
        "content": "List of top 18 courses by analyzing the Admission requirements and success in obtaining Canadian PR may vary based on factors such as your language proficiency test scores (IELTS or TOEFL), research experience, work experience, and other application criteria from the given data through which I can get PR in Canada"
        },]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",  # or use "text-davinci-003"
            messages=messages,

        )
        message = response['choices'][0]['message']['content']
        # print(message)
        selected_fos = extract_courses(message)
        # print(selected_fos)
        # Add the model's response to the conversation
        messages.append({"role": "assistant", "content": message})

        # selected_fos = message
        # print("selected_fos ---< ",selected_fos)
        selected_level = user['AchieveLevel']
        input_words = get_input_words(selected_fos)
        college_df = preprocess_dataframe(courses)
        college_df["combined"] = college_df["FieldOfStudy"] + " " + college_df["Title"] + " " + college_df["Level"]

        vectorizer = TfidfVectorizer(stop_words='english')
        recommended_courses = recommendation(college_df, ' '.join(input_words), vectorizer)
        for index, row in recommended_courses.iterrows():
            # Access the 'Intake' column for the current row and remove '_id' from each dictionary
            for intake_entry in row['Intake']:
                if '_id' in intake_entry:
                    del intake_entry['_id']

        # Assuming the JSON data is stored in a list called 'data'
        intakes = []

        # Create lists to hold the values for each column
        seasons = []
        statuses = []
        deadlines = []

        # Iterate over each dictionary in the data list
        for d in recommended_courses["Intake"]:
            # Extract the intake information from the dictionary
            intake = d

            # Create lists to hold the values for this dictionary's intake information
            d_seasons = []
            d_statuses = []
            d_deadlines = []

            # Iterate over each intake entry
            for i in intake:
                # Add the season, status, and deadline values to the corresponding lists
                d_seasons.append(i["season"])
                d_statuses.append(i["status"])
                d_deadlines.append(i["deadline"])

            # Join the season, status, and deadline lists for this dictionary into strings
            seasons.append(", ".join(d_seasons))
            statuses.append(", ".join(d_statuses))
            deadlines.append(", ".join(d_deadlines))

        # Create a DataFrame with the columns "season", "status", and "deadline"
        intake_df = pd.DataFrame(
            {"Seasons": seasons, "Status": statuses, "Deadline": deadlines})
        # print(intake_df)
        recommended_courses.reset_index(drop=True, inplace=True)
        intake_df.reset_index(drop=True, inplace=True)
        recommended_courses = pd.concat([recommended_courses.drop('Intake', axis=1), intake_df], axis=1)
        eligible, not_eligible = calibre_checker(recommended_courses, user)
        not_eligible = not_eligible[not_eligible["Level"] == selected_level]
        dictionary = {'Title': input_words, 'Level': selected_level.lower()}
        eligible = priority(eligible,dictionary,input_words)
        eligible['Length'] = eligible['Length'].apply(
                lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months" if x >= 12 else f"{x} Months")
        not_eligible['Length'] = not_eligible['Length'].apply(
                lambda x: f"{x // 12} Year{'s' if x // 12 > 1 else ''} " + f"{x % 12} Months" if x >= 12 else f"{x} Months")
        eligible["PR Chances"] = eligible["PR"]
        eligible = eligible.reindex(columns=['CreatedOn', 'FieldOfStudy', 'InstituteName', 'Title', 'Level', 'Length', 'ApplicationFee', 'FeeText','Seasons', 'Status', 'Deadline',
                                        'Percentage', 'Backlog', 'Gap', 'Campus', 'IeltsOverall', "PteOverall", 'TOEFLOverall', "DuolingoOverall", 'GRE', 'GMAT', 'City', 'Province', 'Visa Chances', 'PR', 'PR Chances', 'Notes'])
        not_eligible = not_eligible.reindex(columns=['CreatedOn', 'FieldOfStudy', 'InstituteName', 'Title', 'Level', 'Length', 'ApplicationFee', 'FeeText', 'Seasons', 'Status',
                                              'Deadline', 'Percentage', 'Backlog', 'Gap', 'Campus', 'IeltsOverall', "PteOverall", 'TOEFLOverall', "DuolingoOverall", 'GRE', 'GMAT', 'City', 'Province', 'Notes'])
        eligible.fillna("N/A", inplace=True)   
        not_eligible.fillna("N/A", inplace=True)   

        results = {
            'eligible_courses': eligible.to_dict(orient='records'),
            'not_eligible_courses': not_eligible.to_dict(orient='records')
        }
        # data = recommended_courses.to_dict("records")
        # # json_data = json.dumps(data)
        # json_data = JSONEncoder().encode(results)

        # print("data ---<",json_data)

        return jsonify({'message': 'Email received and processed', 'email': email, 'history': convo,'reco': results})
    else:
        return jsonify({'message': 'No email received'})


def is_eligible(row, user):
        tests = ['IELTS', 'TOEFL', 'PTE', 'Duolingo', 'GRE', 'GMAT']
        for test in tests:
            if test in user['Scores']:
                if test == 'IELTS' and user['Scores'][test]['Overall'] < row['IeltsOverall']:
                    return False
                elif test == 'TOEFL' and user['Scores'][test]['Overall'] < row['TOEFLOverall']:
                    return False
                elif test == 'PTE' and user['Scores'][test]['Overall'] < row['PteOverall']:
                    return False
                elif test == 'Duolingo' and user['Scores'][test]['Overall'] < row['DuolingoOverall']:
                    return False
                elif test == 'GRE' and user['Scores'][test]['Total'] < row['GRE']:
                    return False
                elif test == 'GMAT' and user['Scores'][test]['Total'] < row['GMAT']:
                    return False

        # Checking for Percentage, Backlog and Gap
        if 'Undergraduate' in user['previousEducation']:
            for education in user['previousEducation']['Undergraduate']:
                if education['Percentage'] < row['Percentage']:
                    return False
                if education['Backlogs'] > row['Backlog']:
                    return False

        if 'Gap' in user and user['Gap'] > row['Gap']:
            return False

        return True

def calibre_checker(df, user):
    # Creates a new DataFrame column 'Eligible', where the is_eligible function is applied to every row in the DataFrame
    df['Eligible'] = df.apply(lambda row: is_eligible(row, user), axis=1)
    eligible = df[df['Eligible'] == True]
    not_eligible = df[df['Eligible'] == False]
    return eligible, not_eligible

def get_input_words(list_of_fos):
        title = ' '.join(list_of_fos)
        input_words = set(re.findall(r'[a-zA-Z]+', title.lower()))
        joining_words = {'and', 'or', 'for', 'in', 'the', 'of', 'on', 'to', 'a', 'an'}
        return input_words - joining_words

def preprocess_dataframe(df):
    df["IeltsOverall"] = df["IeltsOverall"].astype(float).round(1)
    df['Length'] = df['Length'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
    df["FeeText"] = df['Fee']
    df["Fee"] = df.Fee.str.extract('(\d+)')
    df['Fee'] = df['Fee'].astype(int)
    return df

def recommendation(df, query, vectorizer):
        """
        Returns a Pandas DataFrame containing the documents in df that are most similar to the query, based on cosine similarity.

        Args:
        - df (Pandas DataFrame): a DataFrame containing text documents
        - query (str): the text query to find similar documents for
        - vectorizer (sklearn CountVectorizer): a CountVectorizer object used to transform the documents in df and the query into vectors

        Returns:
        - Pandas DataFrame: a DataFrame containing the most similar documents to the query, sorted by decreasing similarity
        """
        X = vectorizer.fit_transform(df["combined"])
        input_transform = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(input_transform, X).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-300:-1]
        return df.iloc[related_docs_indices]

@cache.cached(timeout=36000, key_prefix='courses')
def fetch_courses_from_database():
    courses = pd.DataFrame(fetch_all_data("test", "courses"))
    return courses

def fetch_all_data(database, collection):
    MONGODB_URI = os.getenv('MONGODB_URI')
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[database]
    collection = db[collection]
    cursor = collection.find({})
    documents = list(cursor)
    client.close()
    return documents

def priority(dataframe, dictionary,input_words):
        # print(input_words)
        # title=dictionary['Title']
        # x = title.replace(",", " ")
        # input_words = x.split()
        # input_words = set(input_words)
        # # Remove digits and special characters
        # input_words = set(re.findall(r'[a-zA-Z]+', ' '.join(input_words)))

        # # Remove repeating words
        # input_words = set([w.lower() for w in input_words])

        # # Remove joining words
        # joining_words = {'and', 'or', 'for', 'in', 'the', 'of', 'on', 'to', 'a', 'an'}
        # input_words = input_words - joining_words
        # print("Line 66---->",input_words)
        # dictionary['Title']=input_words
        # print("Line 61----> ",dictionary)
        results = []
        df_copy = dataframe.copy()
        for i in range(len(dictionary), 0, -1):
            df_copy = dataframe.copy()
            # get a copy of the dictionary with the last i key-value pairs removed
            sub_dict = dict(list(dictionary.items())[0:i])
            if len(sub_dict) > 0:
                comp_str = ""
                for i, key in enumerate(sub_dict):
                    # print("Hello  --> ",i,key)
                    if i == len(sub_dict)-1:
                        if i == 0:
                            comp_str += "(df_copy['{0}'].str.contains('|'.join({1}), case=False))".format(
                                key, sub_dict[key])
                        else:
                            if isinstance(sub_dict[key], (int, float)):
                                comp_str += "(df_copy['{0}']<={1})".format(key,
                                                                           sub_dict[key])
                            else:
                                comp_str += "(df_copy['{0}'].str.lower()=='{1}')".format(
                                    key, sub_dict[key])
                    elif i <= len(sub_dict)-2:
                        if i == 0:
                            comp_str += "(df_copy['{0}'].str.contains('|'.join({1}), case=False)) & ".format(
                                key, sub_dict[key])
                        else:
                            if isinstance(sub_dict[key], (int, float)):
                                comp_str += "(df_copy['{0}']<={1}) & ".format(
                                    key, sub_dict[key])
                            else:
                                comp_str += "(df_copy['{0}'].str.lower()=='{1}') & ".format(
                                    key, sub_dict[key])
                    else:
                        if isinstance(sub_dict[key], (int, float)):
                            comp_str += "(df_copy['{0}']<={1})".format(key,
                                                                       sub_dict[key])
                        else:
                            comp_str += "(df_copy['{0}'].str.lower()=='{1}')".format(
                                key, sub_dict[key])
                # print(comp_str)
                df = df_copy[eval(comp_str)]

                # df['title_matches'] = df['Title'].str.contains.count('|'.join(sub_dict['Title']), case=False)
                # df = df.sort_values(by='title_matches', ascending=False)
                # filter the DataFrame based on the conditions and sort by the new column
                # counts = df['Title'].str.count(
                #     '|'.join(input_words), flags=re.IGNORECASE)
                # print(counts)
    # compute a score for each row based on the number of matches
    # print(scores)
    # sort the DataFrame based on the scores in descending order
                # df1 = df.assign(score=counts).sort_values(
                #     'score', ascending=False).drop('score', axis=1)
                results.append(df)
                # print(result)

        final_result = pd.concat(results, ignore_index=True)
        final_finalResult = pd.concat([final_result, dataframe], ignore_index=True)
        # print(final_finalResult)
        final_finalResult.drop_duplicates(subset='_id', keep='first', inplace=True)
        lo_bhai = final_finalResult.filter(['FieldOfStudy', 'Province', 'Institute', 'Length', 'IeltsOverall',
                                           "DuolingoOverall", "PteOverall", "Intake", 'City', 'Campus', 'Title', 'Level', 'Fee'], axis=1)
        # lo_bhai.to_csv("final_result.csv",index=False)
        # remove any duplicate rows from the final result
        return final_finalResult
    
def extract_courses(text):
    # The regular expression pattern to match the universities and courses
    pattern = r"\d+\.\s(.*?)- (.*?)[\n|$]"
    matches = re.findall(pattern, text)
    return [" - ".join(match) for match in matches]




# def convert_to_serializable(document):
#     # Convert specific columns to strings
#     columns_to_convert = ['email', 'Gender', 'fname', 'lname', 'AchieveLevel', 'CountryOfOrigin', 'AppliedForVisa']
#     for column in columns_to_convert:
#         document[column] = str(document[column])

#     # Remove _id and __v fields if they exist
    

#     return document

def fetch_user_details(database, collection,email):
    MONGODB_URI = os.getenv('MONGODB_URI')
    client = pymongo.MongoClient(MONGODB_URI)
    db = client[database]
    collection = db[collection]
    document = collection.find_one({'email':email})
    # documents = list(cursor)
    del document['_id']
    del document['__v']
    client.close()
    return document

def should_end_conversation(messages):
    # Count the number of system messages
    system_messages = [msg for msg in messages if msg["role"] == "system"]

    # If there are two or more system messages, end the conversation
    if len(system_messages) >= 2:
        return True
    else:
        return False


if __name__ == '__main__':
    app.run(debug=True)
