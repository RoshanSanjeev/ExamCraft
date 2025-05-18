import os
from openai import OpenAI

class StudentAgent:
    def __init__(self, profile):
        self.profile = profile
        self.weak_areas = {}
        self.log = []
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_answer(self, question):
        """
        Generates a student-like answer to the teacher's question using the OpenAI LLM.
        """
        messages = [
            {"role": "system", "content": f"You are a student learning {self.profile['subject']} at {self.profile['difficulty']} level. Your goal is: {self.profile['goal']}."},
            {"role": "user", "content": f"Teacher asks: {question}"}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        self.log.append({"question": question, "student_answer": answer})
        return answer

    def evaluate_teacher_effectiveness(self, question, explanation, student_answer):
        """
        Evaluates how effective the teacher's explanation was based on the student's response and learning goal.
        Returns a score between 0.0 and 1.0.
        """
        messages = [
            {"role": "user", "content": f"""
Teacher asked: {question}
Teacher explained: {explanation}
Student answered: {student_answer}
Student's learning goal: {self.profile['goal']}

Rate the teacher's effectiveness in helping the student learn the concept.
Respond with a number between 0.0 (not helpful) and 1.0 (very effective). Only output the number.
"""}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return float(response.choices[0].message.content.strip())

    def update_weak_areas(self, question, student_answer):
        """
        Tracks weak topic areas by comparing the question and student's answer.
        """
        for topic in ["loops", "recursion", "lists", "variables", "functions"]:
            if topic in question.lower() and topic not in student_answer.lower():
                self.weak_areas[topic] = self.weak_areas.get(topic, 0) + 1
        return self.weak_areas