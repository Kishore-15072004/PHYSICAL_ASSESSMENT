import os
import os
import random
from dotenv import load_dotenv

try:
    from groq import Groq
except Exception:
    Groq = None

load_dotenv()

class RecommendationEngine:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=groq_key) if (Groq is not None and groq_key) else None
        self.thresholds = {
            "attendance": 80, "endurance": 60, "strength": 60, "flexibility": 55,
            "participation": 70, "skill_speed": 60, "physical_progress": 60,
            "motivation": 6, "stress_level": 6, "sleep_quality": 6, "nutrition": 6
        }
        self.STATIC_DATA = {
            "endurance": "Perform 20 minutes of light jogging 3x per week.",
            "strength": "Complete 3 sets of bodyweight squats and push-ups 2x per week.",
            "flexibility": "Follow a 10-minute dynamic stretching routine every morning.",
            "stress_level": "Practice 'Box Breathing' (4s inhale, 4s hold, 4s exhale) during study breaks.",
            "attendance": "Use a digital planner to set reminders 15 minutes before PE sessions.",
            "nutrition": "Prioritize a high-protein breakfast to sustain energy."
        }

    def analyze_profile(self, user_inputs):
        strengths, all_weak = [], []
        for feature, val in user_inputs.items():
            threshold = self.thresholds.get(feature, 60)
            is_weak = val >= threshold if feature == "stress_level" else val < threshold
            if is_weak:
                all_weak.append(feature)
            else:
                strengths.append(feature)
        return strengths[:2], all_weak

    def generate(self, user_inputs, predicted_score):
        strengths, weak_keys = self.analyze_profile(user_inputs)

        # 1. Get AI Tips (optional)
        ai_tips = []
        if self.client and weak_keys:
            try:
                prompt = f"PE Score: {predicted_score}%. Weaknesses: {weak_keys}. Give 2 FITT tips. No intro, no numbers, one per line."
                chat = self.client.chat.completions.create(
                    messages=[{"role": "system", "content": "You are a PE coach."}, {"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.5,
                )
                raw_lines = chat.choices[0].message.content.strip().split('\n')
                ai_tips = [line.lstrip("1234567890. -*").strip() for line in raw_lines if len(line) > 10]
            except Exception:
                ai_tips = []  # Fallback to static only

        # 2. Get Static Tips
        static_tips = [self.STATIC_DATA[k] for k in weak_keys if k in self.STATIC_DATA]
        random.shuffle(static_tips)

        # 3. Merge top tips
        final_list = ai_tips[:2] + static_tips[:2]

        return {
            "strengths": [s.replace('_', ' ').title() for s in strengths],
            "weak_areas": [w.replace('_', ' ').title() for w in weak_keys],
            "recommendations": final_list
        }


def generate_recommendations(user_inputs, predicted_score=0):
    return RecommendationEngine().generate(user_inputs, predicted_score)