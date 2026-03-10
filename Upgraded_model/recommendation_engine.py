import os
import random
from dotenv import load_dotenv

try:
    from groq import Groq
except:
    Groq=None

load_dotenv()

class RecommendationEngine:

    def __init__(self):

        key=os.getenv("GROQ_API_KEY")

        self.client=Groq(api_key=key) if (Groq and key) else None

        self.thresholds={
        "attendance":80,
        "endurance":60,
        "strength":60,
        "flexibility":55,
        "participation":70,
        "skill_speed":60,
        "physical_progress":60,
        "motivation":6,
        "stress_level":6,
        "sleep_quality":6,
        "nutrition":6,
        "focus":6
        }

        self.static_tips={
        "endurance":"Jog or cycle 20 minutes three times weekly.",
        "strength":"Practice push-ups and squats twice weekly.",
        "flexibility":"Follow a 10-minute stretching routine daily.",
        "stress_level":"Practice breathing or meditation for stress.",
        "nutrition":"Include protein, fruits and vegetables daily.",
        "sleep_quality":"Maintain 7-9 hours sleep each night."
        }

    def analyze(self,inputs):

        strengths=[]
        weak=[]

        for k,v in inputs.items():

            th=self.thresholds.get(k,60)

            if k=="stress_level":
                condition=v>=th
            else:
                condition=v<th

            if condition:
                weak.append(k)
            else:
                strengths.append(k)

        return strengths[:3],weak

    def ai_recommend(self,weak,score):

        if not self.client or not weak:
            return [],"Groq unavailable or no weaknesses detected"

        try:

            prompt=f"Student PE score {score}%. Weak areas {weak}. Give two short coaching tips."

            chat=self.client.chat.completions.create(

            messages=[
            {"role":"system","content":"You are a PE coach"},
            {"role":"user","content":prompt}
            ],

            model="llama-3.3-70b-versatile",
            temperature=0.5)

            lines=chat.choices[0].message.content.split("\n")

            tips=[l.strip("-•1234567890. ") for l in lines if len(l)>10]

            return tips[:2],"Groq AI recommendations fetched successfully"

        except:

            return [],"Groq request failed — using fallback tips"

    def generate(self,inputs,score):

        strengths,weak=self.analyze(inputs)

        ai_tips,status=self.ai_recommend(weak,score)

        static=[self.static_tips[k] for k in weak if k in self.static_tips]

        random.shuffle(static)

        recs=list(dict.fromkeys(ai_tips+static[:3]))

        if not recs:
            recs=["Maintain current training and gradually increase intensity."]

        return {
        "strengths":[s.replace("_"," ").title() for s in strengths],
        "weak_areas":[w.replace("_"," ").title() for w in weak],
        "recommendations":recs
        },status


def generate_recommendations(inputs,score=0):

    engine=RecommendationEngine()

    return engine.generate(inputs,score)