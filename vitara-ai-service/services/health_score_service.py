from schemas.health_score import HealthScoreRequest, HealthScoreResponse, HealthScoreBreakdown

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
        """
        # 1. Stress Score (0-100) - Higher is less stressed (100 = tidak stres)
        avg_stress_level = (request.nlp_result.stress_level + request.typing_result.stress_score) / 2.0
        stress_score = max(0, min(100, int(round((1.0 - avg_stress_level) * 100))))
        
        # 2. Sleep Score (0-100) - Direct from sleep model quality score
        sleep_score = max(0, min(100, request.sleep_result.quality_score))
        
        # 3. Mood Score (0-100) - Based on emotional classification mapping
        mood_score = cls.calculate_mood_score(request.nlp_result.emotion)
        
        # 4. Nutrition Score (0-100) - Evaluated from estimated meal calories
        nutrition_score = cls.calculate_nutrition_score(request.food_result.estimated_calories)
        
        # 5. Overall Health Score (0-100) - Weighted average
        # Stress: 30%, Sleep: 30%, Mood: 20%, Nutrition: 20%
        raw_overall = (stress_score * 0.3) + (sleep_score * 0.3) + (mood_score * 0.2) + (nutrition_score * 0.2)
        overall_score = max(0, min(100, int(round(raw_overall))))
        
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
