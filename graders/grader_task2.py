import re
from typing import List, Dict, Any


def grade_schedule_extraction(action: dict, ground_truth: dict) -> float:
    if not ground_truth.get("scheduling_intent", False):
        if not action.get("meetings") or len(action.get("meetings", [])) == 0:
            return 1.0
        else:
            return 0.5
    
    expected_meetings = ground_truth.get("meetings", [])
    actual_meetings = action.get("meetings", [])
    
    if not expected_meetings and not actual_meetings:
        return 1.0
    
    if not expected_meetings and actual_meetings:
        return 0.6
    
    if expected_meetings and not actual_meetings:
        return 0.0
    
    total_expected = len(expected_meetings)
    matched = 0
    
    for expected in expected_meetings:
        for actual in actual_meetings:
            score = _score_single_meeting(expected, actual)
            if score >= 0.5:
                matched += 1
                break
    
    recall = matched / total_expected if total_expected > 0 else 0.0
    
    precision_bonus = 0.0
    if len(actual_meetings) > 0:
        precision = matched / len(actual_meetings)
        precision_bonus = precision * 0.2
    
    base_score = 0.8 * recall + precision_bonus
    
    base_score = min(1.0, base_score)
    
    return max(0.0, min(1.0, base_score))


def _score_single_meeting(expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
    score = 0.0
    
    if _compare_dates(expected.get("date", ""), actual.get("date", "")):
        score += 0.35
    
    if _compare_times(expected.get("time", ""), actual.get("time", "")):
        score += 0.3
    
    if expected.get("duration") and actual.get("duration"):
        if _compare_durations(expected["duration"], actual["duration"]):
            score += 0.2
    
    if expected.get("purpose") and actual.get("purpose"):
        exp_purpose = str(expected["purpose"]).lower()
        act_purpose = str(actual["purpose"]).lower()
        if exp_purpose in act_purpose or act_purpose in exp_purpose:
            score += 0.15
        elif any(word in act_purpose for word in exp_purpose.split()):
            score += 0.08
    
    return min(1.0, score)


def _compare_dates(date1: str, date2: str) -> bool:
    d1 = _normalize_date(date1)
    d2 = _normalize_date(date2)
    
    if not d1 or not d2:
        return date1.lower().strip() == date2.lower().strip()
    
    return d1 == d2


def _normalize_date(date_str: str) -> str:
    date_str = str(date_str).lower().strip()
    
    months = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
        'monday': '', 'tuesday': '', 'wednesday': '',
        'thursday': '', 'friday': '', 'saturday': '', 'sunday': ''
    }
    
    for name, num in months.items():
        if name in date_str:
            date_str = date_str.replace(name, num)
    
    nums = re.findall(r'\d+', date_str)
    if len(nums) >= 2:
        return ''.join(nums[:3]).zfill(6)
    
    return date_str


def _compare_times(time1: str, time2: str) -> bool:
    t1 = _normalize_time(time1)
    t2 = _normalize_time(time2)
    
    if not t1 or not t2:
        return time1.lower().strip() == time2.lower().strip()
    
    time_diff = abs(t1 - t2)
    return time_diff <= 1


def _normalize_time(time_str: str) -> float:
    time_str = str(time_str).lower().strip()
    
    time_str = time_str.replace('am', '').replace('pm', '')
    time_str = time_str.replace('a.m.', '').replace('p.m.', '')
    time_str = time_str.replace('est', '').replace('pst', '').replace('edt', '')
    time_str = time_str.strip()
    
    nums = re.findall(r'\d+', time_str)
    
    if len(nums) >= 2:
        hours = int(nums[0])
        minutes = int(nums[1])
    elif len(nums) == 1:
        hours = int(nums[0])
        minutes = 0
    else:
        return 0
    
    if time_str.isdigit():
        val = int(time_str)
        if val > 12:
            hours = val // 100
            minutes = val % 100
    
    if 'pm' in str(time_str).lower() and hours < 12:
        hours += 12
    
    return hours + minutes / 60


def _compare_durations(duration1: str, duration2: str) -> bool:
    d1 = _normalize_duration(duration1)
    d2 = _normalize_duration(duration2)
    
    if d1 == 0 or d2 == 0:
        return str(duration1).lower().strip() == str(duration2).lower().strip()
    
    ratio = min(d1, d2) / max(d1, d2)
    return ratio >= 0.7


def _normalize_duration(duration_str: str) -> float:
    duration_str = str(duration_str).lower().strip()
    
    nums = re.findall(r'[\d.]+', duration_str)
    
    if not nums:
        return 0
    
    value = float(nums[0])
    
    if 'hour' in duration_str or 'hr' in duration_str:
        return value
    elif 'min' in duration_str:
        return value / 60
    elif value >= 10:
        return value / 60
    
    return value