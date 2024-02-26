from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import json
import pandas as pd
import datetime
import os
import subprocess
from docx import Document

import os


import json
import base64

import requests
from docx import Document
from io import BytesIO
import random
import string


import boto3

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Retrieve AWS credentials from environment variables
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.environ.get('AWS_DEFAULT_REGION')

# Set up AWS session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_default_region
)

# Now you can use the session to interact with AWS services, e.g.:
s3 = session.client('s3')


def generate_fhir_document_reference(json_data):
    # Convert JSON to string
    json_string = json.dumps(json_data)

    # Encode the string to Base64
    base64_encoded_data = base64.b64encode(json_string.encode()).decode()

    fhir_document_reference = {
        "resourceType": "DocumentReference",
        "identifier": [
            {
                "system": "http://example.org/documents",
                "value": json_data["Consultant"]  # Assuming Consultant is unique identifier
            }
        ],
        "version": "1",  # Assuming version 1
        "status": "current",  # Assuming current status
        "type": {
            "text": "Consultation Document"
        },
        "subject": {
            "reference": f"Patient/{json_data['MRN']}"  # Assuming MRN is the patient identifier
        },
        "date": "2024-02-19T00:00:00Z",  # Assuming fixed date format
        "author": [
            {
                "reference": f"Practitioner/{json_data['Consultant']}"  # Assuming Consultant is the author
            }
        ],
        "content": [
            {
                "attachment": {
                    "contentType": "application/json",
                    "data": base64_encoded_data,
                    "title": "Consultation Document"
                }
            }
        ]
    }

    return fhir_document_reference



# Function to fetch data from the database based on patient_id
def fetch_data_from_db(patient_id):
    try:
        # Database connection details
        # Replace with your actual database credentials
        conn = psycopg2.connect(host="vaserverevva.postgres.database.azure.com",
                                dbname="vadatabaseevva",
                                user="vaadminevva",
                                password="Avvenimdaav10$$")

        # Create a cursor object
        cur = conn.cursor()

        # Fetch data based on patient_id
        cur.execute("SELECT prev FROM userdata WHERE \"visitId\" = %s", (patient_id,))
        data = cur.fetchone()

        # Close cursor and connection
        cur.close()
        conn.close()

        return data[0] if data else None
    except Exception as e:
        print("Error:", e)
        return None

# Function to process JSON data and generate CSV file
def process_and_generate_csv(data, api_call_counter):
    try:
        print("Data received for processing:", data)  # Debug statement
        print("Type of data received:", type(data))  # Debug statement

        # Ensure data is a JSON string
        if isinstance(data, list):
            print("Converting data to JSON string...")
            data = json.dumps(data)

        # Load JSON data
        print("Loading JSON data...")  # Debug statement
        json_data = json.loads(data)

        # Extract required information and create a list of dictionaries
        word_data = []
        for entry in json_data:
            speaker_id = entry['privSpeakerId']
            priv_json = json.loads(entry['privJson'])
            display_text = priv_json['DisplayText']  # Get the display text from the first version
            nbest = priv_json['NBest'][0]  # Get the first version of NBest
            words = nbest.get('Words', [])  # Check if 'Words' key is present, otherwise use an empty list
            for word_info in words:
                word_data.append({'Word': word_info['Word'], 'Confidence': word_info['Confidence'], 'SpeakerId': speaker_id, 'DisplayText': display_text})

        # Create DataFrame
        print("Creating DataFrame...")  # Debug statement
        df = pd.DataFrame(word_data, columns=['Word', 'Confidence', 'SpeakerId', 'DisplayText'])

        # Add an index column
        df['Index'] = range(1, len(df) + 1)

        # Reorder the columns
        df = df[['Index', 'Word', 'Confidence', 'SpeakerId', 'DisplayText']]

        # Get unique values in 'SpeakerId' column
        unique_speakers = df['SpeakerId'].unique()

        # Create new columns for each unique value in 'SpeakerId'
        for speaker in unique_speakers:
            df[speaker] = df['SpeakerId'] == speaker

        # Delete 'SpeakerId' and 'DisplayText' columns
        df.drop(columns=['SpeakerId', 'DisplayText'], inplace=True)

        # Get today's date
        today_date = datetime.date.today().strftime("%m-%d-%Y")

        # Constructing the file name
        file_name = f"T1_Gate2_speakermapping{api_call_counter}.csv"

        # Export DataFrame to CSV
        print("Exporting DataFrame to CSV...")  # Debug statement
        df.to_csv(file_name, index=False)

        return file_name
    except Exception as e:
        print("Error:", e)
        return None

# Function to fetch summary from the veteran_data table based on patient_id
def fetch_summary_from_db(patient_id):
    try:
        # Database connection details
        # Replace with your actual database credentials
        conn = psycopg2.connect(host="visitcompaniontest.postgres.database.azure.com",
                                port="5432",
                                dbname="postgres",
                                user="visitcompanion",
                                password="Test@1234")

        # Create a cursor object
        cur = conn.cursor()

        # Fetch data based on patient_id
        cur.execute("SELECT result FROM veteran_data WHERE patient_id = %s", (patient_id,))
        data = cur.fetchone()

        # Close cursor and connection
        cur.close()
        conn.close()

        return data[0] if data else None
    except Exception as e:
        print("Error:", e)
        return None

def extract_summary(json_data):

    try:
        

        subjective = json_data.get('Subjective')
        print("subjective is",subjective)
        objective = json_data.get('Objective')
        assessment = json_data.get('Assessment')
        plans = json_data.get('Plan')
        

        summary = {
            'Subjective': subjective,
            'Objective': objective,
            'Assessment': assessment,
            'Plans': plans
        }
        return summary
    except Exception as e:
        print("Error extracting detailed summary:", e)
        return None


# Function to fetch the 'details' column from the veteran_data table based on patient_id
def fetch_details_from_db(patient_id):
    try:
        # Database connection details
        # Replace with your actual database credentials
        conn = psycopg2.connect(host="vaserverevva.postgres.database.azure.com",
                                dbname="vadatabaseevva",
                                user="vaadminevva",
                                password="Avvenimdaav10$$")

        # Create a cursor object
        cur = conn.cursor()

        # Fetch 'details' column based on patient_id
        cur.execute("SELECT prev FROM userdata WHERE \"visitId\" = %s", (patient_id,))
        details = cur.fetchone()

        # Close cursor and connection
        cur.close()
        conn.close()

        return details[0] if details else None
    except Exception as e:
        print("Error:", e)
        return None





# Function to upload a file to S3 bucket
def upload_to_s3(file_path, s3_bucket, s3_key):
    command = f"aws s3 cp {file_path} s3://{s3_bucket}/{s3_key}"
    subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@app.route('/')
def hello_world():
  return 'api online'

# Define a global variable to store the counter
api_call_counter = -1

@app.route('/uploads3', methods=['POST'])
def upload_file_to_s3():
    global api_call_counter  # Access the global variable
    try:
        # Increment the counter for each API call
        api_call_counter += 1
        
        req_data = request.get_json()
        patient_id = req_data.get('patient_id')

        if not patient_id:
           return jsonify({'error': 'Patient ID not provided'})

        # Fetch data from the database
        data = fetch_data_from_db(patient_id)

        if not data:
          return jsonify({'error': 'No data found for the given patient ID'})

        # Process data and generate CSV
        file_name = process_and_generate_csv(data, api_call_counter)

        if not file_name:
            return jsonify({'error': 'Error occurred while processing data'})

        # Fetch summary from the database
        summary_data = fetch_summary_from_db(patient_id)

        if not summary_data:
            return jsonify({'error': 'No summary found for the given patient ID'})

        # Extract summary from the summary data
        summary = extract_summary(summary_data)

        if not summary:
            return jsonify({'error': 'Error extracting summary'})

          # Fetch details from the database
        details_data = fetch_details_from_db(patient_id)

        if not details_data:
            return jsonify({'error': 'No details found for the given patient ID'})

          # Get today's date
        today_date = datetime.date.today().strftime("%m-%d-%Y")

          # Upload CSV file to S3
        #csv_s3_key = f"01-028/VATest-Evva_Health_{today_date}_{patient_id}_speakermapping.csv"
        csv_s3_key = f"01-028/T1_Gate2_speakermapping{api_call_counter}.csv"
        upload_to_s3(file_name, "naii-01-dir", csv_s3_key)

          # Upload summary JSON to S3
        #summary_file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_FHIRnotes.json"
        summary_file_name = f"T1_Gate2_FHIR{api_call_counter}.json"
          # Create JSON file with patient ID and summary
        
        soap = json.dumps({'Facility': 'VA Tech Sprint', 'Consultant': 'VATechSprint001', 'Date': today_date, 'MRN': patient_id, 'detailed_results': summary})
        soap_dict = json.loads(soap)

        fhir_data = generate_fhir_document_reference(soap_dict)
        print(json.dumps(fhir_data, indent=2))
        
 

         
        with open(summary_file_name, 'w') as json_file:
           json.dump(fhir_data, json_file, indent=2)  # Use indent=2 for pretty formatting

        summary_s3_key = f"01-028/{summary_file_name}"
        upload_to_s3(summary_file_name, "naii-01-dir", summary_s3_key)



        #decoded_file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_decodednotes.json"
        decoded_file_name = f"T1_Gate2_decodednotes{api_call_counter}.json"
        with open(decoded_file_name, 'w') as json_file:
           json.dump(soap, json_file, indent=2)  # Use indent=2 for pretty formatting

        decoded_s3_key = f"01-028/{decoded_file_name}"
        upload_to_s3(decoded_file_name, "naii-01-dir", decoded_s3_key)


 
          # Upload details JSON to S3
        #details_file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_transcriptions.json"
        details_file_name = f"T1_Gate2_transcriptions{api_call_counter}.json"
        with open(details_file_name, 'w') as json_file:
             json.dump({'patient_id': patient_id, 'details': details_data}, json_file)
        details_s3_key = f"01-028/{details_file_name}"
        upload_to_s3(details_file_name, "naii-01-dir", details_s3_key)

        return jsonify({"message": "File uploaded successfully to S3!"}), 200
    except ValueError as value_error:
        return jsonify({"message": str(value_error)}), 400
    except Exception as e:
        print("error:", e)
        return jsonify({"message": "Internal Server Error","error":str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
