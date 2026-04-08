import pytest
from graders import grade_classification, grade_schedule_extraction, grade_multi_intent


class TestClassificationGrader:
    def test_correct_classification(self):
        action = {"category": "spam"}
        ground_truth = {"expected_category": "spam"}
        
        score = grade_classification(action, ground_truth)
        assert score >= 0.95
    
    def test_incorrect_classification(self):
        action = {"category": "normal"}
        ground_truth = {"expected_category": "urgent"}
        
        score = grade_classification(action, ground_truth)
        assert 0.0 <= score < 0.7
    
    def test_confidence_bonus(self):
        action_high = {"category": "spam", "confidence": 0.95}
        action_low = {"category": "spam", "confidence": 0.6}
        ground_truth = {"expected_category": "spam"}
        
        score_high = grade_classification(action_high, ground_truth)
        score_low = grade_classification(action_low, ground_truth)
        
        assert score_high >= score_low
    
    def test_reasoning_bonus(self):
        action_reasoning = {
            "category": "spam",
            "reasoning": "This is spam because of suspicious links and promotional language"
        }
        action_no_reasoning = {"category": "spam"}
        ground_truth = {"expected_category": "spam"}
        
        score_with = grade_classification(action_reasoning, ground_truth)
        score_without = grade_classification(action_no_reasoning, ground_truth)
        
        assert score_with >= score_without
    
    def test_empty_category(self):
        action = {"category": ""}
        ground_truth = {"expected_category": "urgent"}
        
        score = grade_classification(action, ground_truth)
        assert score == 0.0
    
    def test_normal_vs_urgent_confusion(self):
        action = {"category": "normal"}
        ground_truth = {"expected_category": "urgent"}
        
        score = grade_classification(action, ground_truth)
        assert 0.3 <= score <= 0.6
    
    def test_case_insensitive(self):
        action = {"category": "SPAM"}
        ground_truth = {"expected_category": "spam"}
        
        score = grade_classification(action, ground_truth)
        assert score >= 0.95


class TestScheduleExtractionGrader:
    def test_single_meeting_match(self):
        action = {
            "meetings": [
                {"date": "Nov 14", "time": "10:00 AM", "duration": "1.5 hours"}
            ]
        }
        ground_truth = {
            "scheduling_intent": True,
            "meetings": [
                {"date": "November 14", "time": "10:00 AM", "duration": "1.5 hours"}
            ]
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert score >= 0.7
    
    def test_no_scheduling_intent(self):
        action = {"meetings": []}
        ground_truth = {
            "scheduling_intent": False,
            "meetings": []
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert score == 1.0
    
    def test_partial_match(self):
        action = {
            "meetings": [
                {"date": "Thursday", "time": "2pm"}
            ]
        }
        ground_truth = {
            "scheduling_intent": True,
            "meetings": [
                {"date": "Thursday", "time": "2:00 PM", "duration": "45 min"},
                {"date": "Friday", "time": "afternoon"}
            ]
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert 0.3 <= score <= 0.7
    
    def test_multiple_options(self):
        action = {
            "meetings": [
                {"date": "Mon Nov 18", "time": "10 AM"},
                {"date": "Wed Nov 20", "time": "2 PM"}
            ]
        }
        ground_truth = {
            "scheduling_intent": True,
            "meetings": [
                {"date": "Monday Nov 18", "time": "10:00 AM"},
                {"date": "Wednesday Nov 20", "time": "2:00 PM"},
                {"date": "Friday Nov 22", "time": "11:00 AM"}
            ]
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert 0.5 <= score <= 0.8
    
    def test_empty_meetings_when_expected(self):
        action = {"meetings": []}
        ground_truth = {
            "scheduling_intent": True,
            "meetings": [{"date": "Nov 15", "time": "3 PM"}]
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert score == 0.0
    
    def test_time_tolerance(self):
        action = {
            "meetings": [{"date": "Nov 14", "time": "10:05 AM"}]
        }
        ground_truth = {
            "scheduling_intent": True,
            "meetings": [{"date": "November 14", "time": "10:00 AM"}]
        }
        
        score = grade_schedule_extraction(action, ground_truth)
        assert score >= 0.6


class TestMultiIntentGrader:
    def test_all_intents_correct(self):
        action = {
            "intents": ["schedule_meeting", "request_documentation", "out_of_office_notification"]
        }
        ground_truth = {
            "expected_intents": ["schedule_meeting", "request_documentation", "out_of_office_notification"]
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert score >= 0.9
    
    def test_partial_intents(self):
        action = {
            "intents": ["schedule_meeting", "request_documentation"]
        }
        ground_truth = {
            "expected_intents": ["schedule_meeting", "request_documentation", "out_of_office_notification"]
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert 0.4 <= score <= 0.9
    
    def test_primary_intent_bonus(self):
        action = {
            "intents": ["request_refund", "escalate_issue"],
            "primary_intent": "request_refund"
        }
        ground_truth = {
            "expected_intents": ["request_refund", "escalate_issue", "complaint"],
            "primary_intent": "request_refund"
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert 0.5 <= score <= 1.0
    
    def test_sentiment_bonus(self):
        action = {
            "intents": ["complaint"],
            "sentiment": "negative"
        }
        ground_truth = {
            "expected_intents": ["complaint", "request_refund"],
            "sentiment": "negative"
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert 0.3 <= score <= 0.9
    
    def test_reply_bonus(self):
        action_no_reply = {
            "intents": ["schedule_meeting"]
        }
        action_with_reply = {
            "intents": ["schedule_meeting"],
            "suggested_reply": "I can meet on Thursday at 2pm. Please confirm if that works for you."
        }
        ground_truth = {
            "expected_intents": ["schedule_meeting"]
        }
        
        score_no_reply = grade_multi_intent(action_no_reply, ground_truth)
        score_with_reply = grade_multi_intent(action_with_reply, ground_truth)
        
        assert score_with_reply >= score_no_reply
    
    def test_empty_intents(self):
        action = {"intents": []}
        ground_truth = {
            "expected_intents": ["request_information"]
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert score == 0.0
    
    def test_intent_normalization(self):
        action = {
            "intents": ["Request Information", "schedule-meeting"]
        }
        ground_truth = {
            "expected_intents": ["request_information", "schedule meeting"]
        }
        
        score = grade_multi_intent(action, ground_truth)
        assert score >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])