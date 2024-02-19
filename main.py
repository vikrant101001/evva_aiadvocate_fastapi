from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import json
import pandas as pd
import datetime
import os
import subprocess
from docx import Document

import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def fhir(soap):
  prompt = soap
  response = openai.ChatCompletion.create(
      model="gpt-4-1106-preview",
      messages=[
          {
              "role":
              "system",
              "content":
              """
              You are a helpful assistant that converts/maps my json data to a specific FHIR format written below. Note: always return in a json format which can be dumped into a json file and also do not add anything before and after the json response even if it is an important note. i.e. the start of the response should be with the opening  '{' of a json file and the end of the response should be the closing '}' of the json and i dont want anything in starting like '```json\n' just start with how a json file starts because i am going to dump the results directly in a json file . THis is the FHIR format:
              {
              "resourceType" : "Encounter",
              // from Resource: id, meta, implicitRules, and language
              // from DomainResource: text, contained, extension, and modifierExtension
              "identifier" : [{ Identifier }], // Identifier(s) by which this encounter is known
              "status" : "<code>", // R!  planned | in-progress | on-hold | discharged | completed | cancelled | discontinued | entered-in-error | unknown
              "class" : [{ CodeableConcept }], // Classification of patient encounter context - e.g. Inpatient, outpatient icon
              "priority" : { CodeableConcept }, // Indicates the urgency of the encounter icon
              "type" : [{ CodeableConcept }], // Specific type of encounter (e.g. e-mail consultation, surgical day-care, ...)
              "serviceType" : [{ CodeableReference(HealthcareService) }], // Specific type of service
              "subject" : { Reference(Group|Patient) }, // The patient or group related to this encounter
              "subjectStatus" : { CodeableConcept }, // The current status of the subject in relation to the Encounter
              "episodeOfCare" : [{ Reference(EpisodeOfCare) }], // Episode(s) of care that this encounter should be recorded against
              "basedOn" : [{ Reference(CarePlan|DeviceRequest|MedicationRequest|
               ServiceRequest) }], // The request that initiated this encounter
              "careTeam" : [{ Reference(CareTeam) }], // The group(s) that are allocated to participate in this encounter
              "partOf" : { Reference(Encounter) }, // Another Encounter this encounter is part of
              "serviceProvider" : { Reference(Organization) }, // The organization (facility) responsible for this encounter
              "participant" : [{ // List of participants involved in the encounter
                "type" : [{ CodeableConcept }], // I Role of participant in encounter
                "period" : { Period }, // Period of time during the encounter that the participant participated
                "actor" : { Reference(Device|Group|HealthcareService|Patient|Practitioner|
                PractitionerRole|RelatedPerson) } // I The individual, device, or service participating in the encounter
              }],
              "appointment" : [{ Reference(Appointment) }], // The appointment that scheduled this encounter
              "virtualService" : [{ VirtualServiceDetail }], // Connection details of a virtual service (e.g. conference call)
              "actualPeriod" : { Period }, // The actual start and end time of the encounter
              "plannedStartDate" : "<dateTime>", // The planned start date/time (or admission date) of the encounter
              "plannedEndDate" : "<dateTime>", // The planned end date/time (or discharge date) of the encounter
              "length" : { Duration }, // Actual quantity of time the encounter lasted (less time absent)
              "reason" : [{ // The list of medical reasons that are expected to be addressed during the episode of care
                "use" : [{ CodeableConcept }], // What the reason value should be used for/as
                "value" : [{ CodeableReference(Condition|DiagnosticReport|
                ImmunizationRecommendation|Observation|Procedure) }] // Reason the encounter takes place (core or reference)
              }],
              "diagnosis" : [{ // The list of diagnosis relevant to this encounter
                "condition" : [{ CodeableReference(Condition) }], // The diagnosis relevant to the encounter
                "use" : [{ CodeableConcept }] // Role that this diagnosis has within the encounter (e.g. admission, billing, discharge …)
              }],
              "account" : [{ Reference(Account) }], // The set of accounts that may be used for billing for this Encounter
              "dietPreference" : [{ CodeableConcept }], // Diet preferences reported by the patient
              "specialArrangement" : [{ CodeableConcept }], // Wheelchair, translator, stretcher, etc
              "specialCourtesy" : [{ CodeableConcept }], // Special courtesies (VIP, board member)
              "admission" : { // Details about the admission to a healthcare service
                "preAdmissionIdentifier" : { Identifier }, // Pre-admission identifier
                "origin" : { Reference(Location|Organization) }, // The location/organization from which the patient came before admission
                "admitSource" : { CodeableConcept }, // From where patient was admitted (physician referral, transfer)
                "reAdmission" : { CodeableConcept }, // Indicates that the patient is being re-admitted icon
                "destination" : { Reference(Location|Organization) }, // Location/organization to which the patient is discharged
                "dischargeDisposition" : { CodeableConcept } // Category or kind of location after discharge
              },
              "location" : [{ // List of locations where the patient has been
                "location" : { Reference(Location) }, // R!  Location the encounter takes place
                "status" : "<code>", // planned | active | reserved | completed
                "form" : { CodeableConcept }, // The physical type of the location (usually the level in the location hierarchy - bed, room, ward, virtual etc.)
                "period" : { Period } // Time period during which the patient was present at the location
              }]

              And this is a sample response from you:
              {
                "resourceType": "Encounter",
                "status": "unknown",  // Your JSON data doesn't specify encounter status, setting to "unknown"
                "class": [],  // Your JSON data doesn't specify encounter class, setting to an empty array
                "priority": null,  // Not specified in your JSON data
                "type": [],  // Not specified in your JSON data
                "serviceType": [],  // Not specified in your JSON data
                "subject": {
                  "reference": "Patient/Ybbc4109"  // Assuming MRN is the patient ID
                },
                "subjectStatus": null,  // Not specified in your JSON data
                "episodeOfCare": [],  // Not specified in your JSON data
                "basedOn": [],  // Not specified in your JSON data
                "careTeam": [],  // Not specified in your JSON data
                "partOf": null,  // Not specified in your JSON data
                "serviceProvider": null,  // Not specified in your JSON data
                "participant": [],  // Not specified in your JSON data
                "appointment": [],  // Not specified in your JSON data
                "virtualService": [],  // Not specified in your JSON data
                "actualPeriod": null,  // Not specified in your JSON data
                "plannedStartDate": "2024-02-12T00:00:00Z",  // Assuming Date from your JSON data is the planned start date/time
                "plannedEndDate": "2024-02-12T23:59:59Z",  // Assuming Date from your JSON data is the planned end date/time
                "length": null,  // Not specified in your JSON data
                "reason": [  // Mapping reasons from your JSON data
                  {
                    "valueReference": {
                      "reference": "Condition/lower-back-pain"  // Assuming this is the FHIR reference to the condition
                    }
                  }
                ],
                "diagnosis": [],  // Not specified in your JSON data
                "account": [],  // Not specified in your JSON data
                "dietPreference": [],  // Not specified in your JSON data
                "specialArrangement": [],  // Not specified in your JSON data
                "specialCourtesy": [],  // Not specified in your JSON data
                "admission": {  // Not specified in your JSON data
                  "preAdmissionIdentifier": null,
                  "origin": null,
                  "admitSource": null,
                  "reAdmission": null,
                  "destination": null,
                  "dischargeDisposition": null
                },
                "location": []  // Not specified in your JSON data
              }

""",
          },
          {
              "role": "user",
              "content": prompt
          },
      ],
  )
  result = response['choices'][0]['message']['content']
  return result





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
def process_and_generate_csv(data, patient_id):
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
        file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_speakermapping.csv"

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

@app.route('/uploads3', methods=['POST'])
def upload_file_to_s3():
    try:
        req_data = request.get_json()
        patient_id = req_data.get('patient_id')

        if not patient_id:
           return jsonify({'error': 'Patient ID not provided'})

        # Fetch data from the database
        data = fetch_data_from_db(patient_id)

        if not data:
          return jsonify({'error': 'No data found for the given patient ID'})

        # Process data and generate CSV
        file_name = process_and_generate_csv(data, patient_id)

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
        csv_s3_key = f"01-028/VATest-Evva_Health_{today_date}_{patient_id}_speakermapping.csv"
        upload_to_s3(file_name, "naii-01-dir", csv_s3_key)

          # Upload summary JSON to S3
        summary_file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_summary.json"
          # Create JSON file with patient ID and summary
        
        soap = json.dumps({'Facility': 'VA Tech Sprint', 'Consultant': 'VATechSprint001', 'Date': today_date, 'MRN': patient_id, 'detailed_results': summary})
        
 

        result = fhir(soap)

         # Convert the result string to a dictionary
        result_dict = json.loads(result)

         
        with open(summary_file_name, 'w') as json_file:
           json.dump(result_dict, json_file,
                indent=2)  # Use indent=2 for pretty formatting

        summary_s3_key = f"01-028/{summary_file_name}"
        upload_to_s3(summary_file_name, "naii-01-dir", summary_s3_key)
 
          # Upload details JSON to S3
        details_file_name = f"VATest-Evva_Health_{today_date}_{patient_id}_transcriptions.json"
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
