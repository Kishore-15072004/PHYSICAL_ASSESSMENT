from datetime import datetime
import numpy as np
import os
import pickle
from recommendation_engine import generate_recommendations

# --------------------------------------------------
# Helper for Relative Paths
# --------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

def rel_path(*parts):
    return os.path.join(base_dir, *parts)

# --------------------------------------------------
# Load Ensemble Models
# --------------------------------------------------

print("\n[System] Loading ensemble models...")

models_ensemble = {}
ensemble_info = {}

try:

    # BPNN
    bpnn_model = np.load(rel_path("saved_model","bpnn_model.npz"), allow_pickle=True)

    W1_bpnn, b1_bpnn = bpnn_model["W1"], bpnn_model["b1"]
    W2_bpnn, b2_bpnn = bpnn_model["W2"], bpnn_model["b2"]

    # Random Forest
    with open(rel_path("saved_model","ensemble","random_forest.pkl"),"rb") as f:
        models_ensemble["rf"] = pickle.load(f)

    # XGBoost
    with open(rel_path("saved_model","ensemble","xgboost.pkl"),"rb") as f:
        models_ensemble["xgb"] = pickle.load(f)

    # Gradient Boost
    with open(rel_path("saved_model","ensemble","gradient_boosting.pkl"),"rb") as f:
        models_ensemble["gb"] = pickle.load(f)

    # SVR
    with open(rel_path("saved_model","ensemble","svr.pkl"),"rb") as f:
        models_ensemble["svr"] = pickle.load(f)

    # Linear Regression
    with open(rel_path("saved_model","ensemble","linear_regression.pkl"),"rb") as f:
        models_ensemble["lr"] = pickle.load(f)

    with open(rel_path("saved_model","ensemble","ensemble_info.pkl"),"rb") as f:
        ensemble_info = pickle.load(f)

    print("✓ Ensemble models loaded successfully")

except FileNotFoundError as e:
    print("❌ Model loading failed:", e)
    exit()

# --------------------------------------------------
# Activation Function
# --------------------------------------------------

def relu(x):
    return np.maximum(0,x)

# --------------------------------------------------
# Feature Configuration
# --------------------------------------------------

compulsory_features = [

("attendance",50,100),
("endurance",30,95),
("strength",35,95),
("flexibility",30,90),
("participation",55,100),
("skill_speed",30,95),
("physical_progress",40,95)

]

optional_features = [

("motivation",4,9),
("stress_level",2,8),
("self_confidence",4,9),
("focus",3,9),
("teamwork",4,9),
("peer_support",5,9),
("communication",4,9),
("sleep_quality",4,9),
("nutrition",4,9),
("screen_time",2,8)

]

optional_defaults = {f[0]:6 for f in optional_features}

# --------------------------------------------------
# Feature Names for PFI
# --------------------------------------------------

feature_names = [f[0] for f in compulsory_features] + [f[0] for f in optional_features]

# --------------------------------------------------
# Descriptive Mapping
# --------------------------------------------------

descriptive_options = {

"motivation":[
("Low (Reluctant / Needs encouragement)",4),
("Moderate (Participates when guided)",6),
("High (Self-driven / Enthusiastic)",9)
],

"stress_level":[
("Low (Calm / Relaxed)",2),
("Moderate (Manageable)",5),
("High (Overwhelmed / Anxious)",8)
],

"self_confidence":[
("Low (Hesitant / Doubtful)",4),
("Moderate (Occasionally confident)",6),
("High (Self-assured / Bold)",9)
],

"focus":[
("Distracted (Difficulty staying on task)",4),
("Steady (Functional concentration)",6),
("Sharp (High flow state)",9)
],

"teamwork":[
("Developing (Needs guidance in groups)",4),
("Cooperative (Works well with peers)",6),
("Leadership (Guides and motivates team)",9)
],

"peer_support":[
("Limited (Rare encouragement)",5),
("Supportive (Encourages teammates)",7),
("Highly Supportive (Actively uplifts others)",9)
],

"communication":[
("Passive (Minimal interaction)",4),
("Clear (Effective exchanges)",6),
("Proactive (Leadership / High engagement)",9)
],

"sleep_quality":[
("Poor (Less than 5 hours / Restless)",4),
("Fair (5–7 hours / Average)",6),
("Restorative (7–9+ hours / Deep)",9)
],

"nutrition":[
("Poor (Processed foods / Inconsistent)",4),
("Balanced (Healthy mix / Regular)",6),
("Optimal (Nutrient-dense / High fuel)",9)
],

"screen_time":[
("Mindful (Under 2 hours daily)",3),
("Moderate (3–5 hours daily)",5),
("Excessive (6+ hours daily)",8)

]

}

# --------------------------------------------------
# Collect Inputs
# --------------------------------------------------

inputs = []
user_inputs_dict = {}
user_inputs_display = {}
selected_optional_attributes = []
selected_labels = {}

print("\n--- ENTER PHYSICAL ATTRIBUTES ---")

for name,min_v,max_v in compulsory_features:

    while True:

        try:

            val = float(input(f"{name.replace('_',' ').title()} ({min_v}-{max_v}): "))

            if min_v <= val <= max_v:

                inputs.append(val)
                user_inputs_dict[name] = val
                user_inputs_display[name] = val
                break

            print("❌ Out of range")

        except:
            print("❌ Invalid input")

# --------------------------------------------------
# Optional Inputs
# --------------------------------------------------

print("\n--- OPTIONAL ATTRIBUTES ---")

for i,(name,_,_) in enumerate(optional_features,1):
    print(f"{i}. {name.replace('_',' ').title()}")

choice = input("\nSelect numbers (e.g.,1,3,5) or press Enter: ").strip()

selected = [int(i) for i in choice.split(",") if i.strip().isdigit()] if choice else []

for idx,(name,_,_) in enumerate(optional_features,1):

    if idx in selected:

        selected_optional_attributes.append(name)

        options = descriptive_options[name]

        print(f"\n{name.replace('_',' ').title()}:")

        for i,(label,val) in enumerate(options,1):
            print(f"{i}. {label}")

        while True:

            try:

                pick = int(input("Select option: "))

                if 1 <= pick <= len(options):

                    label,val = options[pick-1]

                    selected_labels[name] = label

                    inputs.append(val)
                    user_inputs_dict[name] = val
                    user_inputs_display[name] = label

                    break

                print("Invalid selection")

            except:
                print("Enter a number")

    else:

        val = optional_defaults[name]
        inputs.append(val)
        user_inputs_dict[name] = val

# --------------------------------------------------
# Normalize Input
# --------------------------------------------------

X = np.array(inputs)
X_norm = X.copy()

X_norm[:7] /= 100
X_norm[7:] /= 10

X_norm = X_norm.reshape(1,-1)

# --------------------------------------------------
# BPNN Prediction
# --------------------------------------------------

def model_predict(data):

    hidden = relu(np.dot(data,W1_bpnn) + b1_bpnn)
    output = (np.dot(hidden,W2_bpnn) + b2_bpnn) * 100

    return output.flatten()

# --------------------------------------------------
# Hybrid Ensemble Prediction (used for PFI)
# --------------------------------------------------

def hybrid_predict(data):

    y_bpnn = model_predict(data)[0]
    y_rf = models_ensemble["rf"].predict(data)[0] * 100
    y_xgb = models_ensemble["xgb"].predict(data)[0] * 100
    y_gb = models_ensemble["gb"].predict(data)[0] * 100
    y_svr = models_ensemble["svr"].predict(data)[0] * 100
    y_lr = models_ensemble["lr"].predict(data)[0] * 100

    preds = np.array([y_bpnn,y_rf,y_xgb,y_gb,y_svr,y_lr])

    weights = ensemble_info.get(
        "weights",
        np.array([0.22,0.20,0.18,0.16,0.12,0.12])
    )

    return np.dot(weights,preds)

# --------------------------------------------------
# Hybrid Permutation Feature Importance
# --------------------------------------------------

def compute_hybrid_pfi(X, feature_names):

    baseline = hybrid_predict(X)

    importances = []

    for i in range(X.shape[1]):

        X_perm = X.copy()

        np.random.shuffle(X_perm[:, i])

        perm_score = hybrid_predict(X_perm)

        importance = abs(baseline - perm_score)

        importances.append(importance)

    ranking = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    return ranking

# --------------------------------------------------
# Ensemble Predictions
# --------------------------------------------------

print("\n[System] Generating ensemble predictions...")

y_bpnn = model_predict(X_norm)[0]
y_rf = models_ensemble["rf"].predict(X_norm)[0] * 100
y_xgb = models_ensemble["xgb"].predict(X_norm)[0] * 100
y_gb = models_ensemble["gb"].predict(X_norm)[0] * 100
y_svr = models_ensemble["svr"].predict(X_norm)[0] * 100
y_lr = models_ensemble["lr"].predict(X_norm)[0] * 100

predictions = np.array([y_bpnn,y_rf,y_xgb,y_gb,y_svr,y_lr])

weights = ensemble_info.get(
    "weights",
    np.array([0.22,0.20,0.18,0.16,0.12,0.12])
)

predicted_score = np.clip(np.dot(weights,predictions),0,100)

# --------------------------------------------------
# Hybrid PFI Analysis
# --------------------------------------------------

pfi_ranking = compute_hybrid_pfi(X_norm,feature_names)

top_positive = pfi_ranking[:3]
top_negative = pfi_ranking[-3:]

primary_positive_influencer = top_positive[0][0].replace("_"," ").title()
major_negative_factor = top_negative[-1][0].replace("_"," ").title()

# --------------------------------------------------
# Recommendations
# --------------------------------------------------

print("[System] Generating coaching recommendations...")

recs,groq_status = generate_recommendations(user_inputs_dict,predicted_score)

print(f"[Groq Status] {groq_status}")

# --------------------------------------------------
# Console Report
# --------------------------------------------------

print("\n"+"="*55)
print("STUDENT PERFORMANCE REPORT")
print("="*55)

print("Final Score:",round(predicted_score,2),"%")

print("\nTop Influencing Factors (Hybrid PFI):")

for name,_ in top_positive:
    print("•",name.replace("_"," ").title())

print("\nStrengths:",", ".join(recs["strengths"]))
print("Weak Areas:",", ".join(recs["weak_areas"]))

print("\nRecommendations:")

for tip in recs.get("recommendations", []):

    tip = tip.replace("**","")
    parts = tip.split(". ")

    first = True

    for p in parts:

        p = p.strip()

        if not p:
            continue

        if first:
            print("•", p + ".")
            first = False
        else:
            print(" ", p + ".")

# --------------------------------------------------
# Save Professional Report
# --------------------------------------------------

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
filename = f"PE_Diagnostic_Report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

try:

    with open(filename,"w",encoding="utf-8") as f:

        f.write("="*60+"\n")
        f.write("      PHYSICAL EDUCATION DIAGNOSTIC ASSESSMENT REPORT\n")
        f.write(f"      Generated on: {timestamp}\n")
        f.write("="*60+"\n\n")

        f.write("--- CORE METRICS & DIAGNOSTICS ---\n")

        f.write(f"{'FINAL SCORE':<25}: {predicted_score:.2f}%\n")
        f.write(f"{'Primary Positive Influencer':<25}: {primary_positive_influencer}\n")
        f.write(f"{'Major Negative Factor':<25}: {major_negative_factor}\n")

        f.write("\nTop Influencing Factors (Hybrid PFI):\n")

        for name,_ in top_positive:
            f.write(f"• {name.replace('_',' ').title()}\n")

        f.write("-"*60+"\n\n")

        f.write("--- USER INPUT DATA ---\n")

        for name,_,_ in compulsory_features:
            val = user_inputs_dict[name]
            f.write(f"{name.replace('_',' ').title():<25}: {val}\n")

        for name in selected_optional_attributes:
            val = user_inputs_dict[name]
            label = selected_labels.get(name,"")
            f.write(f"{name.replace('_',' ').title():<25}: {val} ({label})\n")

        f.write("\n"+"-"*60+"\n")

        if recs.get("strengths"):
            f.write(f"TOP STRENGTHS: {', '.join(recs['strengths'])}\n")

        if recs.get("weak_areas"):
            f.write(f"FOCUS AREAS  : {', '.join(recs['weak_areas'])}\n")

        f.write("\nRECOMMENDED ACTION PLAN:\n")

        for tip in recs.get("recommendations",[]):

            tip = tip.replace("**","")
            parts = tip.split(". ")

            first = True

            for p in parts:

                p = p.strip()

                if not p:
                    continue

                if first:
                    f.write(f"• {p}.\n")
                    first = False
                else:
                    f.write(f"  {p}.\n")

            f.write("\n")

        f.write("\n"+"="*60+"\n")
        f.write("END OF REPORT\n")

    print(f"\n✅ Report saved as: {filename}")

except Exception as e:

    print(f"\n❌ Error saving report: {e}")