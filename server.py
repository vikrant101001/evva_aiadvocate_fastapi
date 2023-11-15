from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain import LLMChain
from langchain.llms import OpenAIChat
from langchain.prompts import Prompt
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import json
import traceback
import random
import openai
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from collections import defaultdict
import threading
import time

from fastapi import FastAPI, HTTPException, Depends, Header
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os


#packages to Note
#pip install pydantic==1.10.12
#pip install python-multipart(-0.0.6)
#pip install uvicorn(-0.23.1)


app = FastAPI()
#no dotenv here
#new files added

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Faiss index and model
index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
  store = pickle.load(f)

store.index = index

with open("training/master.txt", "r") as f:
  promptTemplate = f.read()

prompt = Prompt(template=promptTemplate,
                input_variables=["history", "context", "question"])

llmChain = LLMChain(prompt=prompt,
                    llm=OpenAIChat(
                        temperature=0.5,
                        model_name="gpt-3.5-turbo-16k",
                        openai_api_key=os.environ["OPENAI_API_KEY"]))

history = []





# Access the API key
openai.api_key = os.environ["OPENAI_API_KEY"]
API_SECRET = os.environ["API_SECRET"]



#***************CHECKIN MODULE***********************************
# Load existing data from the JSON file, if any
json_file_path = 'user_responses.json'
try:
  with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
except FileNotFoundError:
  data = {
    "check_ins": {
        "careteam_id1": {
            "Week 1": {
                "current_question_index": 0
            }
        },
        "careteam_id2": {
            "Week 1": {
                "current_question_index": 0
            }
        }
    }
  }
  # Log the error for debugging

json_file_path2 = 'questionnaire_data.json'
try:
  with open(json_file_path2, 'r') as json_file:
    qdata = json.load(json_file)
except FileNotFoundError:
  qdata = {
      "assessments": {
          "careteam_id1": {
              "Week 1": {
                  "current_question_index": 0
              }
          },
          "careteam_id2": {
              "Week 1": {
                  "current_question_index": 0
              }
          }
      }
  }

# Define the questions and gather user information
questions = {
    'Q1': {
        'question': "How is the patient's mood today?",
        'options':
        ["Agitated", "Angry", "Frustrated", "Lost", "Neutral", "Happy"]
    },
    'Q2': {
        'question':
        "Has the patient experienced any of the following in the past week?",
        'options': ["Increasing irritability", "Wandering", "Delusions"]
    },
    'Q3': {
        'question':
        "How was the patient's sleep yesterday?",
        'options': [
            "Well rested", "Woke up once", "Woke up 2 or 3 times",
            "Woke up 4 or more times", "Disrupted"
        ]
    },
    'Q4': {
        'question':
        "Has the patient experienced any vomiting in the past week? (yes/no)",
        'options': ["yes", "no"]
    },
    'Q5.1': {
        'question':
        "How many times in the past week?",
        'options':
        ["Once only", "2 to 3 times", "4 to 5 times", "More than 5 times"]
    },
    'Q5.2': {
        'question':
        "For how long did the episode last?",
        'options': [
            "Less than 30 minutes", "30 mins to an hour", "1 to 4 hours",
            "Full day"
        ]
    },
    'Q5.3': {
        'question':
        "How did it impact the patient's daily activities?",
        'options': [
            "Difficulty eating and drinking", "Difficulty walking", "Fatigue",
            "Severe dehydration"
        ]
    }
}

#questionairre = 'questionairre_questions.json'
#with open(questionairre, 'r') as json_file:
#  questionaire_questions_new = json.load(json_file)
questionaire_questions_new = {
    'Q1': {
        "title":
        "Daily Activities",
        "question":
        "Can the patient feed herself without assistance or does she need help during meal times?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q2': {
        "title":
        "Daily Activities",
        "question":
        "I see. Moving on, when it comes to dressing, can the patient put on her clothes without any assistance? Or does she struggle with certain aspects like buttons, zippers, or shoe laces?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q3': {
        "title":
        "Mobility",
        "question":
        "Got it. How about transferring? Is the patient able to move in and out of her bed or chair on her own? Or does she need some help or an assistive device for that?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q4': {
        "title":
        "Mobility",
        "question":
        "How would you rate the patient’s ability to walk a block or climb a flight of stairs or more?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q5': {
        "title":
        "Cognition",
        "question":
        "Has Martha experienced any difficulties with memory, attention, or problem-solving that affected daily life?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q6': {
        "title":
        "Cognition",
        "question":
        "Thank you. Has the patient exhibited difficulties in recognizing their environment or wandered away from home lately?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q7': {
        "title":
        "Mind",
        "question":
        "Has the patient felt agitated, anxious, irritable, or depressed?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q8': {
        "title":
        "Mind",
        "question":
        "Thank you for answering. From late afternoon till night, has the patient shown increased confusion, disorientation, restlessness, or difficulty sleeping?",
        "options": [{
            "option": "Rarely or never",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Frequently",
            "score": 1
        }, {
            "option": "Constantly",
            "score": 0
        }]
    },
    'Q9': {
        "title":
        "Independence (IADL)",
        "question":
        "Thanks for that. Now, let's move on to some instrumental activities. Can the patient prepare and cook meals by herself? Or does she require help with certain dishes or prefer pre-prepared meals?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q10': {
        "title":
        "Independence (IADL)",
        "question":
        "Understood. When it comes to her medications, can the patient manage and take them correctly without supervision? Or does she sometimes forget and need reminders?",
        "options": [{
            "option": "Manages independently",
            "score": 3
        }, {
            "option": "Requires occasional assistance",
            "score": 2
        }, {
            "option": "Needs frequent assistance",
            "score": 1
        }, {
            "option": "Requires full assistance",
            "score": 0
        }]
    },
    'Q11': {
        "title":
        "Social",
        "question":
        "Now, thinking about the patient’s social activity. Has Martha had any difficulty interacting with relatives or friends or participating in community or social activities?",
        "options": [{
            "option": "No difficulty, socializes independently",
            "score": 3
        }, {
            "option": "Some difficulty",
            "score": 2
        }, {
            "option": "Much difficulty",
            "score": 1
        }, {
            "option": "Cannot socialize without assistance",
            "score": 0
        }]
    },
    'Q12': {
        "title":
        "Social",
        "question":
        "How frequently does the patient engage in social activities or maintain relationships?",
        "options": [{
            "option": "Regularly",
            "score": 3
        }, {
            "option": "Occasionally",
            "score": 2
        }, {
            "option": "Rarely",
            "score": 1
        }, {
            "option": "Almost never or never",
            "score": 0
        }]
    }
}

scoring = {
    "Manages independently": 3,
    "Requires occasional assistance": 2,
    "Needs frequent assistance": 1,
    "Requires full assistance": 0,
    "Rarely or never": 3,
    "Occasionally": 2,
    "Frequently": 1,
    "Constantly": 0,
    "No difficulty, socializes independently": 3,
    "Some difficulty": 2,
    "Much difficulty": 1,
    "Cannot socialize without assistance": 0,
    "Regularly": 3,
    "Occasionally": 2,
    "Rarely": 1,
    "Almost never or never": 0
}
# Create a defaultdict with a default value of 0
scoring_dict = defaultdict(lambda: 0)

# Update the default_scoring dictionary with your scoring_dict
scoring_dict.update(scoring)
# Store the scores for each title
#title_scores = {
#    question["title"]: 0
#    for question in questionaire_questions_new
#}

# Database configuration
username = "aiassistantevvaadmin"
password = "EvvaAi10$"
hostname = "aiassistantdatabase.postgres.database.azure.com"
database_name = "aidatabasecombined"

# Construct the connection URL
db_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

# SQLAlchemy model
Base = declarative_base()


class AdvocateHistory(Base):
  __tablename__ = 'aiadvocatehistory'

  id = Column(Integer, primary_key=True, autoincrement=True)
  user_question = Column(String)
  bot_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)
  timestamp = Column(DateTime, default=datetime.utcnow)


# Function to insert conversation into the database
def insert_conversation(user_question: str, bot_answer: str, careteam_id: str,
                        caregiver_id: str):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create an AdvocateHistory object
    conversation = AdvocateHistory(user_question=user_question,
                                   bot_answer=bot_answer,
                                   careteam_id=careteam_id,
                                   caregiver_id=caregiver_id)

    # Add the AdvocateHistory object to the session and commit the transaction
    session.add(conversation)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting conversation: {e}")


class CheckinHistory(Base):
  __tablename__ = 'acmcheckinhistory'

  id = Column(Integer, primary_key=True, autoincrement=True)
  checkin_question = Column(String)
  user_answer = Column(String)
  caregiver_id = Column(String)
  careteam_id = Column(String)
  timestamp = Column(DateTime, default=datetime.utcnow)


def insert_checkin(checkin_question: str, user_answer: str, caregiver_id: str,
                   careteam_id: str):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a CheckinHistory object
    checkin = CheckinHistory(checkin_question=checkin_question,
                             user_answer=user_answer,
                             caregiver_id=caregiver_id,
                             careteam_id=careteam_id)

    # Add the CheckinHistory object to the session and commit the transaction
    session.add(checkin)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting checkin: {e}")


class FAHistory(Base):
  __tablename__ = 'acmfahistory'

  id = Column(Integer, primary_key=True, autoincrement=True)
  fa_question = Column(String)
  fa_answer = Column(String)
  fa_title = Column(String)
  fa_score = Column(Integer)
  caregiver_id = Column(String)
  careteam_id = Column(String)
  timestamp = Column(DateTime, default=datetime.utcnow)


def insert_fa(fa_question: str, fa_answer: str, fa_title: str, fa_score: int,
              caregiver_id: str, careteam_id: str):
  try:
    # Create a SQLAlchemy engine and session
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create an FAHistory object
    fa_history = FAHistory(fa_question=fa_question,
                           fa_answer=fa_answer,
                           fa_title=fa_title,
                           fa_score=fa_score,
                           caregiver_id=caregiver_id,
                           careteam_id=careteam_id)

    # Add the FAHistory object to the session and commit the transaction
    session.add(fa_history)
    session.commit()

    # Close the session
    session.close()

  except Exception as e:
    # Handle exceptions (e.g., database errors)
    print(f"Error inserting functional assessment data: {e}")


from fastapi import File, UploadFile, HTTPException


@app.post('/train_ai_advocate')
async def train_ai(file: UploadFile = File(...)):
  try:
    # Process the uploaded file and get the data
    result = process_uploaded_file(file)

    # Convert and save the data to a file
    if result is not None:
      convert_and_save_to_file(result)

    # Get the training data from the specified folder
    training_data_folder = Path("training/facts/")
    training_data = list(training_data_folder.glob("**/*.*"))

    # Check if there is data in the trainingData folder
    if len(training_data) < 1:
      raise HTTPException(
          status_code=400,
          detail=
          "The folder training/facts should be populated with at least one .txt or .md file."
      )

    # Read data from training files
    data = []
    for training_file in training_data:
      with open(training_file) as f:
        print(f"Add {f.name} to dataset")
        data.append(f.read())

    # Split the text into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=0)
    docs = [text_splitter.split_text(chunk) for chunk in data]

    # Initialize OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    # Create FAISS index from the texts
    store = FAISS.from_texts(docs, embeddings)

    # Write the index to the file
    faiss.write_index(store.index, "training.index")
    store.index = None

    # Save the store object to a pickle file
    with open("faiss.pkl", "wb") as f:
      pickle.dump(store, f)

    return {"message": "Action performed successfully!"}

  except Exception as e:
    raise HTTPException(status_code=500,
                        detail=f"Internal Server Error: {str(e)}")


@app.get("/")
def index():
  return "API Online"


def declare():
  username = "aiassistantevvaadmin"
  password = "EvvaAi10$"
  hostname = "aiassistantdatabase.postgres.database.azure.com"
  database_name = "aidatabasecombined"

  # Construct the connection URL
  db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"


last_api_call_time = time.time()


def reset_history():
  global history
  history = []


# Assuming you have already defined the app, last_api_call_time, history, geocode, store, llmChain, splitter, and other necessary components


@app.post("/")
async def ask(request: Request,
        careteam_id: str = Header("not implied"),
        caregiver_id: str = Header("not implied")):
  global last_api_call_time, history

  username = "aiassistantevvaadmin"
  password = "EvvaAi10$"
  hostname = "aiassistantdatabase.postgres.database.azure.com"
  database_name = "aidatabasecombined"

  # Construct the connection URL
  db_connection_url = f"postgresql://{username}:{password}@{hostname}/{database_name}"

  #api_secret_from_frontend = request.headers.get('X-API-SECRET')
  #if api_secret_from_frontend != API_SECRET:
  #  raise HTTPException(status_code=401, detail="Unauthorized access")

  careteam_id = request.headers.get('careteam_id', "not implied")
  caregiver_id = request.headers.get('caregiver_id', "not implied")
  
  if careteam_id == "not implied" or caregiver_id == "not implied":
    return JSONResponse(
        content={'message': "Caregiver or careteam id not implied"})

  try:
    req_data = await request.json()
    blanker = 1
    if blanker == 1:
      user_question = req_data['question']

      # Check if it's been more than 2 minutes since the last API call
      current_time = time.time()
      if current_time - last_api_call_time > 120:
        reset_history()
      last_api_call_time = current_time  # Update the last API call time

      # Process user location only if needed
      if "mapbox" in user_question.lower(
      ) or "mapboxapi" in user_question.lower():
        location = user_question
        latitude, longitude = geocode(location, os.environ["MAP_KEY"])
        answer = "Please provide your complete location so that we can find the nearest required professional for you: "
      else:
        docs = store.similarity_search(user_question)
        contexts = [
            f"Context {i}:\n{doc.page_content}" for i, doc in enumerate(docs)
        ]
        answer = llmChain.predict(question=user_question,
                                  context="\n\n".join(contexts),
                                  history=history)

      #bot_answer = splitter(answer)

      history.append(f"Human: {user_question}")
      history.append(f"Bot: {answer}")

      insert_conversation(user_question, answer, careteam_id, caregiver_id)

      return JSONResponse(content={"answer": answer, "success": True})
    else:
      return JSONResponse(content={
          "answer": None,
          "success": False,
          "message": "Unauthorised"
      })
  except Exception as e:
    return JSONResponse(content={
        "answer": None,
        "success": False,
        "message": str(e)
    },
                        status_code=400)


@app.get('/get_first_question')
def get_question(request: Request,careteam_id: str = Header("not implied"),
                 caregiver_id: str = Header("not implied")):
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  #if api_secret_from_frontend != API_SECRET:
    #raise HTTPException(status_code=401, detail="Unauthorized access")
  careteam_id = request.headers.get('careteam_id', "not implied")
  caregiver_id = request.headers.get('caregiver_id', "not implied")
  
  if careteam_id == "not implied" or caregiver_id == "not implied":
    return JSONResponse(
        content={'message': "Caregiver or careteam id not implied"})

  current_week = len(data['check_ins'])
  week = "Week " + str(current_week)

  # Access the data for the specific careteam_id and week
  careteam_data = data['check_ins'].setdefault(careteam_id,
                                               {}).setdefault(week, {})
  current_question_index = careteam_data.get('current_question_index', 0)

  if current_question_index <= len(questions):
    # Update the data structure for the specific careteam_id
    careteam_data.clear()
    careteam_data["current_question_index"] = 0

    # Save the updated data in 'user_responses.json' file
    with open('user_responses.json', 'w') as file:
      json.dump(data, file, indent=4)

    current_question_index = 0
    current_question_key = list(questions.keys())[current_question_index]
    current_question = questions[current_question_key]

    if isinstance(current_question, dict) and 'options' in current_question:
      options = current_question['options']
      formatted_options = [f"{option}" for i, option in enumerate(options)]
      question_text = {
          'question': current_question['question'],
          'options': formatted_options
      }

      return JSONResponse(
          content={
              'message': "First Question",
              'question': current_question['question'],
              'options': formatted_options
          })
    else:
      question_text = {'question': current_question}

      return JSONResponse(content={
          'message': "First Question",
          'question': current_question['question']
      })


@app.post('/submit_answer')
async def submit_answer(request: Request,
                  careteam_id: str = Header("not implied"),
                  caregiver_id: str = Header("not implied")):
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  #if api_secret_from_frontend != API_SECRET:
  #  raise HTTPException(status_code=401, detail="Unauthorized access")

  careteam_id = request.headers.get('careteam_id', "not implied")
  caregiver_id = request.headers.get('caregiver_id', "not implied")
  
  if careteam_id == "not implied" or caregiver_id == "not implied":
    return JSONResponse(
        content={'message': "Caregiver or careteam id not implied"})

  try:
    req_data = await request.json()
    user_response = req_data['answer']
    if not user_response:
      return JSONResponse(content={'error': 'Invalid request'},
                          status_code=400)

    current_week = len(data['check_ins'])

    # Check if this week's entry exists for the specific careteam_id, if not, create a new entry
    week = f"Week {current_week}"
    careteam_data = data['check_ins'].setdefault(careteam_id,
                                                 {}).setdefault(week, {})
    current_question_index = careteam_data.get('current_question_index', 0)

    # Get the list of questions and their keys
    question_keys = list(questions.keys())
    question_count = len(question_keys)

    # Check if all questions have been answered for this week
    if current_question_index >= question_count:
      return JSONResponse(
          content={
              'message': 'All questions for this week have been answered!'
          })

    # Get the current question and options (if applicable)
    current_question_key = question_keys[current_question_index]
    current_question = questions[current_question_key]

    # Insert into the database (if needed)
    insert_checkin(str(current_question), user_response, caregiver_id,
                   careteam_id)

    # Update the data with the user's response
    careteam_data[f"Q{current_question_index + 1}"] = {
        'question': current_question,
        'response': user_response
    }

    # Increment the current question index for the next iteration
    careteam_data['current_question_index'] = current_question_index + 1

    # If all questions are answered, move to the next week
    if careteam_data['current_question_index'] >= question_count:
      next_week = f"Week {current_week + 1}"
      data['check_ins'][careteam_id][next_week] = {'current_question_index': 0}

    # Save the data in the JSON file
    with open(json_file_path, 'w') as json_file:
      json.dump(data, json_file, indent=4)

    if (current_question_index + 1) >= question_count:
      return JSONResponse(
          content={
              'message':
              'Response saved successfully! You have completed the check-in'
          })
    else:
      next_question_key = question_keys[current_question_index + 1]
      next_question = questions[next_question_key]
      if 'options' in next_question:
        return JSONResponse(
            content={
                'message': 'Response saved successfully!',
                'question': next_question['question'],
                'options': next_question['options']
            })
      else:
        return JSONResponse(
            content={
                'message': 'Response saved successfully!',
                'question': next_question['question']
            })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return JSONResponse(content={'error': 'Internal Server Error'},
                        status_code=500)


@app.get('/get_questionnaire_question')
def get_questionnaire_question(request: Request,
                               careteam_id: str = Header("not implied"),
                               caregiver_id: str = Header("not implied")):
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  #if api_secret_from_frontend != API_SECRET:
  #  raise HTTPException(status_code=401, detail="Unauthorized access")


  careteam_id = request.headers.get('careteam_id', "not implied")
  caregiver_id = request.headers.get('caregiver_id', "not implied")
  
  if careteam_id == "not implied" or caregiver_id == "not implied":
    return JSONResponse(
        content={'message': "Caregiver or careteam id not implied"})

  qcurrent_week = len(qdata['assessments'])

  week = f"Week {qcurrent_week}"
  qcurrent_question_index = qdata['assessments'].get(careteam_id, {}).get(
      week, {}).get('current_question_index', 0)

  if qcurrent_question_index <= len(questionaire_questions_new):
    # Update the data structure for the specific careteam_id
    week_data = qdata['assessments'].setdefault(careteam_id,
                                                {}).setdefault(week, {})
    week_data.clear()
    week_data["current_question_index"] = 0

    # Save the updated data in 'questionnaire_data.json' file
    with open('questionnaire_data.json', 'w') as file:
      json.dump(qdata, file, indent=4)

    qcurrent_question_index = 0
    qcurrent_question_key = list(
        questionaire_questions_new.keys())[qcurrent_question_index]
    qcurrent_question = questionaire_questions_new[qcurrent_question_key]

    if isinstance(qcurrent_question, dict) and 'options' in qcurrent_question:
      options = [option["option"] for option in qcurrent_question["options"]]

      return JSONResponse(
          content={
              'message': "First Question",
              'question': qcurrent_question['question'],
              'options': options
          })
    else:
      question_text = {'question': qcurrent_question}

      return JSONResponse(content={
          'message': "First Question",
          'question': qcurrent_question['question']
      })


@app.post('/submit_questionnaire_answer')
async def submit_questionnaire_answer(request: Request,
                                careteam_id: str = Header("not implied"),
                                caregiver_id: str = Header("not implied")):
  # Check if the API_SECRET from the frontend matches the one stored in the environment
  #if api_secret_from_frontend != API_SECRET:
  #  raise HTTPException(status_code=401, detail="Unauthorized access")

  careteam_id = request.headers.get('careteam_id', "not implied")
  caregiver_id = request.headers.get('caregiver_id', "not implied")
  
  if careteam_id == "not implied" or caregiver_id == "not implied":
    return JSONResponse(
        content={'message': "Caregiver or careteam id not implied"})

  req_data = await request.json()
  user_response = req_data['answer']
  
  try:
    if not user_response:
      return JSONResponse(content={'error': 'Invalid request'},
                          status_code=400)

    score = scoring_dict.get(user_response, None)

    if score is None:
      return JSONResponse(content={'error': 'Invalid answer'}, status_code=400)

    qcurrent_week = len(qdata['assessments'])

    # Check if this week's entry exists for the specific careteam_id, if not, create a new entry
    week = f"Week {qcurrent_week}"
    careteam_data = qdata['assessments'].setdefault(careteam_id,
                                                    {}).setdefault(week, {})
    current_question_index = careteam_data.get('current_question_index', 0)

    if current_question_index < 2:
      title = "Daily Activities"
    elif current_question_index < 4:
      title = "Mobility"
    elif current_question_index < 6:
      title = "Cognition"
    elif current_question_index < 8:
      title = "Mind"
    else:
      title = "Independence (IADL)"

    # Get the list of questions and their keys
    question_keys = list(questionaire_questions_new.keys())
    question_count = len(question_keys)

    # Check if all questions have been answered for this week
    if current_question_index >= question_count:
      return JSONResponse(
          content={
              'message': 'All questions for this week have been answered!'
          })

    # Get the current question and options (if applicable)
    current_question_key = question_keys[current_question_index]
    current_question = questionaire_questions_new[current_question_key]

    # Insert into the database (if needed)
    insert_fa(str(current_question), user_response, title, score, caregiver_id,
              careteam_id)

    # Update the data with the user's response
    careteam_data[f"Q{current_question_index + 1}"] = {
        'question': current_question['question'],
        'response': user_response,
        'title': title,
        'score': score
    }

    # Increment the current question index for the next iteration
    careteam_data['current_question_index'] = current_question_index + 1

    # If all questions are answered, move to the next week
    if careteam_data['current_question_index'] >= question_count:
      next_week = f"Week {qcurrent_week + 1}"
      qdata['assessments'][careteam_id][next_week] = {
          'current_question_index': 0
      }

    # Save the data in the JSON file
    with open(json_file_path2, 'w') as json_file:
      json.dump(qdata, json_file, indent=4)

    if (current_question_index + 1) >= question_count:
      return JSONResponse(
          content={
              'message':
              'Response saved successfully! You have completed the check-in'
          })
    else:
      next_question_key = question_keys[current_question_index + 1]
      next_question = questionaire_questions_new[next_question_key]

      if 'options' in next_question:
        options = [option["option"] for option in next_question["options"]]
        return JSONResponse(
            content={
                'message': 'Response saved successfully!',
                'question': next_question['question'],
                'options': options
            })
      else:
        return JSONResponse(
            content={
                'message': 'Response saved successfully!',
                'question': next_question['question']
            })

  except Exception as e:
    # Manually print the traceback to the console
    traceback.print_exc()

    return JSONResponse(content={'error': 'Internal Server Error'},
                        status_code=500)


if __name__ == '__main__':
  import uvicorn

  uvicorn.run(app, host='0.0.0.0', port=3000)
