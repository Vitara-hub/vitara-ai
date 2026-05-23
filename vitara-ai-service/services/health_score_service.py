import math
from schemas.health_score import HealthScoreRequest, HealthScoreResponse, HealthScoreBreakdown

def round_half_up(n: float) -> int:
    """
    Rounds a float to the nearest integer, with ties (.5) rounding up.
    Includes a small epsilon to handle float precision issues.
    """
    return math.floor(n + 0.5 + 1e-9)

class HealthScoreService:
    @staticmethod
    def calculate_mood_score(emotion: str) -> int:
        """
        Calculates the mood score (0-100) based on the detected emotion.
        """
        emotion_lower = emotion.lower().strip()
        
        # Mapping rules for emotion to score
        if emotion_lower in ["happy", "joy", "excited", "love", "cheerful"]:
            return 90
        elif emotion_lower in ["neutral", "calm", "relaxed"]:
            return 70
        elif emotion_lower in ["sad", "anxious", "fear", "stressed", "lonely", "worried"]:
            return 40
        elif emotion_lower in ["angry", "frustrated", "annoyed", "irritated"]:
            return 30
        else:
            # Fallback for unexpected or default emotions
            return 60

    @staticmethod
    def calculate_nutrition_score(estimated_calories: int) -> int:
        """
        Calculates the nutrition score (0-100) based on meal calories.
        - Ideal meal size is between 400 and 700 kcal.
        """
        cals = estimated_calories
        
        if 400 <= cals <= 700:
            return 90
        elif 200 <= cals < 400:
            return 70
        elif cals < 200:
            return 50
        elif 700 < cals <= 1000:
            return 60
        else:  # cals > 1000
            return 40

    @classmethod
    def calculate_health_score(cls, request: HealthScoreRequest) -> HealthScoreResponse:
        """
        Calculates sub-scores and overall health score using rule-based formulas.
        Supports partial data with dynamic weighting.
        """
        scores = {}
        weights = {}

        # 1. Stress Score (0-100) - Higher is less stressed (100 = tidak stres)
        # Bounded by NLP stress_level and typing stress_score if available
        stress_elements = []
        if request.nlp_result is not None:
            stress_elements.append(request.nlp_result.stress_level)
        if request.typing_result is not None:
            stress_elements.append(request.typing_result.stress_score)

        if stress_elements:
            avg_stress_level = sum(stress_elements) / len(stress_elements)
            stress_score = max(0, min(100, round_half_up((1.0 - avg_stress_level) * 100)))
            scores['stress'] = stress_score
            weights['stress'] = 0.30
        else:
            stress_score = None

        # 2. Sleep Score (0-100)
        if request.sleep_result is not None:
            sleep_score = max(0, min(100, request.sleep_result.quality_score))
            scores['sleep'] = sleep_score
            weights['sleep'] = 0.30
        else:
            sleep_score = None

        # 3. Mood Score (0-100)
        if request.nlp_result is not None:
            mood_score = cls.calculate_mood_score(request.nlp_result.emotion)
            scores['mood'] = mood_score
            weights['mood'] = 0.20
        else:
            mood_score = None

        # 4. Nutrition Score (0-100)
        if request.food_result is not None:
            nutrition_score = cls.calculate_nutrition_score(request.food_result.estimated_calories)
            scores['nutrition'] = nutrition_score
            weights['nutrition'] = 0.20
        else:
            nutrition_score = None

        # 5. Overall Health Score (0-100)
        if scores:
            total_weight = sum(weights.values())
            raw_overall = sum(scores[key] * (weights[key] / total_weight) for key in scores)
            overall_score = max(0, min(100, round_half_up(raw_overall)))
        else:
            # Fallback jika tidak ada data sama sekali (Default Netral/60)
            overall_score = 60

        breakdown = HealthScoreBreakdown(
            mood=mood_score,
            nutrition=nutrition_score,
            stress=stress_score,
            sleep=sleep_score
        )

        return HealthScoreResponse(
            health_score=overall_score,
            breakdown=breakdown
        )
