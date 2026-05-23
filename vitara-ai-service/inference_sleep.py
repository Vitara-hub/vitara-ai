import os
import sys
import json

class SleepPredictor:
    """
    Predictor class for Sleep Quality Score.
    Currently uses the calibrated rule-based logic from the router.
    """
    def predict(self, duration_hours, interruptions, sleep_debt_hours=None):
        score = 100.0 - (interruptions * 10)
        
        if duration_hours < 7.0:
            score -= (7.0 - duration_hours) * 10.0
            
        if sleep_debt_hours is not None:
            score -= sleep_debt_hours * 5.0
            
        quality_score = max(0, min(100, int(round(score))))
        return quality_score

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference_sleep.py '<json_input>'", file=sys.stderr)
        print("Example: python inference_sleep.py '{\"duration_hours\": 6.5, \"interruptions\": 2, \"sleep_debt_hours\": 1.0}'", file=sys.stderr)
        sys.exit(1)

    json_input_str = sys.argv[1]

    try:
        data = json.loads(json_input_str)
        duration_hours = float(data.get("duration_hours", 7.0))
        interruptions = int(data.get("interruptions", 0))
        sleep_debt_hours = data.get("sleep_debt_hours")
        if sleep_debt_hours is not None:
            sleep_debt_hours = float(sleep_debt_hours)

        predictor = SleepPredictor()
        quality_score = predictor.predict(
            duration_hours=duration_hours,
            interruptions=interruptions,
            sleep_debt_hours=sleep_debt_hours
        )

        output = {
            "quality_score": quality_score
        }
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
