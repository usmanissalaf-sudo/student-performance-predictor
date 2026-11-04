
import gradio as gr
import pandas as pd
import joblib

model = joblib.load("student_performance_model.pkl")
preprocessor = joblib.load("student_preprocessor.pkl")


def predict_student(
    school, sex, age, address, famsize, Pstatus,
    Medu, Fedu, Mjob, Fjob, reason, guardian,
    traveltime, studytime, failures, schoolsup,
    famsup, paid, activities, nursery, higher,
    internet, romantic, famrel, freetime, goout,
    Dalc, Walc, health, absences
):
    student = pd.DataFrame([{
        "school": school, "sex": sex, "age": age,
        "address": address, "famsize": famsize,
        "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
        "guardian": guardian, "traveltime": traveltime,
        "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup,
        "paid": paid, "activities": activities,
        "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic,
        "famrel": famrel, "freetime": freetime,
        "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences
    }])

    encoded = preprocessor.transform(student)
    prediction = model.predict(encoded)[0]

    if prediction == 1:
        return "PASS: Student is predicted to pass."

    return "FAIL: Student is predicted to fail."


inputs = [
    gr.Dropdown(["GP", "MS"], label="School"),
    gr.Dropdown(["M", "F"], label="Sex"),
    gr.Slider(15, 22, step=1, value=16, label="Age"),
    gr.Dropdown(["U", "R"], label="Address"),
    gr.Dropdown(["GT3", "LE3"], label="Family Size"),
    gr.Dropdown(["T", "A"], label="Parent Status"),
    gr.Slider(0, 4, step=1, value=2, label="Mother's Education"),
    gr.Slider(0, 4, step=1, value=2, label="Father's Education"),
    gr.Dropdown(["teacher", "health", "services", "at_home", "other"], label="Mother's Job"),
    gr.Dropdown(["teacher", "health", "services", "at_home", "other"], label="Father's Job"),
    gr.Dropdown(["course", "reputation", "other", "home"], label="School Choice Reason"),
    gr.Dropdown(["mother", "father", "other"], label="Guardian"),
    gr.Slider(1, 4, step=1, value=1, label="Travel Time"),
    gr.Slider(1, 4, step=1, value=2, label="Study Time"),
    gr.Slider(0, 3, step=1, value=0, label="Past Failures"),
    gr.Dropdown(["yes", "no"], label="School Support"),
    gr.Dropdown(["yes", "no"], label="Family Support"),
    gr.Dropdown(["yes", "no"], label="Paid Classes"),
    gr.Dropdown(["yes", "no"], label="Activities"),
    gr.Dropdown(["yes", "no"], label="Nursery"),
    gr.Dropdown(["yes", "no"], label="Higher Education"),
    gr.Dropdown(["yes", "no"], label="Internet"),
    gr.Dropdown(["yes", "no"], label="Romantic Relationship"),
    gr.Slider(1, 5, step=1, value=4, label="Family Relationship"),
    gr.Slider(1, 5, step=1, value=3, label="Free Time"),
    gr.Slider(1, 5, step=1, value=3, label="Going Out"),
    gr.Slider(1, 5, step=1, value=1, label="Workday Alcohol"),
    gr.Slider(1, 5, step=1, value=1, label="Weekend Alcohol"),
    gr.Slider(1, 5, step=1, value=3, label="Health"),
    gr.Number(value=0, precision=0, label="Absences")
]

app = gr.Interface(
    fn=predict_student,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="Student Performance Prediction System",
    description="Enter student information to predict academic performance."
)

app.launch()
