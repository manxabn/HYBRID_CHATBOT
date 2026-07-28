import sqlite3
import pandas as pd

conn = sqlite3.connect("knowledge_base.db")
cursor = conn.cursor()

# Step 5: Query Functions
# Function to query Prerequisites table
def query_prerequisites(course_name):
    query = '''
        SELECT * FROM Prerequisites WHERE Course = ?
    '''
    results = cursor.execute(query, (course_name,)).fetchall()
    if results:
        for result in results:
            print(result)
    else:
        print(f"No prerequisite information found for course: {course_name}")


def query_course_details(course_name):
    query = '''
        SELECT * FROM CourseDetails WHERE Course = ?
    '''
    results = cursor.execute(query, (course_name,)).fetchall()
    for result in results:
        print(result)




# Function to query English QA table
def query_english_qa(question):
    query = '''
        SELECT * FROM EnglishQA WHERE Question = ?
    '''
    results = cursor.execute(query, (question,)).fetchall()
    if results:
        for result in results:
            print(result)
    else:
        print(f"No entries found for question: {question}")

# Function to query Faculty List table
def query_faculty_details(faculty_initial):
    query = '''
        SELECT * FROM FacultyList WHERE Initial = ?
    '''
    results = cursor.execute(query, (faculty_initial,)).fetchall()
    if results:
        for result in results:
            print(result)
    else:
        print(f"No entries found for faculty with Initial: {faculty_initial}")

# Function to query Coordinator table
def query_coordinator_details(course_name):
    query = '''
        SELECT * FROM Coordinator WHERE Course = ?
    '''
    results = cursor.execute(query, (course_name,)).fetchall()
    if results:
        for result in results:
            print(result)
    else:
        print(f"No coordinator details found for course: {course_name}")



# Example Queries

print("\nDetails for course 'CSE101':")
query_course_details("CSE101-02")  # Replace with actual course name


print("\nDetails for English QA:")
query_english_qa("Can you give me the understanding of the overall advising process?")  # Replace with actual English question

print("\nDetails for Faculty with Initial 'AA':")
query_faculty_details("ACH")  # Replace with actual faculty initial

print("\nDetails for Coordinator for course 'CSE101':")
query_coordinator_details("CSE251")  # Replace with actual course name

print("\nDetails for prerequisite for course 'CSE101':")
query_prerequisites("CSE350")


