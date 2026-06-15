
import numpy as np

def classify_palm_personality(features):
    """
    Rule-based personality prediction using:
    - Line dominance (length × curve)
    - Curvature strength
    - Ratio between lines
    - Intersections (indicative of mixed/complex personality)
    """

    # Extract values safely
   
    life_len = features.get("life_length", 0)
    head_len = features.get("head_length", 0)
    heart_len = features.get("heart_length", 0)

    life_curve = features.get("life_curve", 0)
    head_curve = features.get("head_curve", 0)
    heart_curve = features.get("heart_curve", 0)

    life_int = features.get("life_intersections", 0)
    head_int = features.get("head_intersections", 0)
    heart_int = features.get("heart_intersections", 0)

    # Compute dominance score
 
    scores = {
        "life": life_len * (1 + life_curve),
        "head": head_len * (1 + head_curve),
        "heart": heart_len * (1 + heart_curve),
    }

    dominant_line = max(scores, key=scores.get)

    # RULE SET
   

    # 1️⃣ Emotional Type → strong heart line  
    if dominant_line == "heart" and heart_len > 0.35:
        return "emotional"

    # 2️⃣ Analytic Type → strong head line  
    if dominant_line == "head" and head_len > life_len * 0.8:
        return "analytic"

    # 3️⃣ Creative Type → high curvature (especially heart/head)
    if max(life_curve, head_curve, heart_curve) > 0.35:
        return "creative"

    # 4️⃣ Balanced Type → similar lengths & low curvature
    lens = np.array([life_len, head_len, heart_len])
    if np.std(lens) < 0.25 and max(life_curve, head_curve, heart_curve) < 0.2:
        return "balanced"

    # 5️⃣ Complex personality → many intersections
    if (life_int + head_int + heart_int) >= 6:
        return "complex"

    # fallback
    return "balanced"
