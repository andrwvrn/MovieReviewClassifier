from django.http import JsonResponse

from .scripts.text_processing import process_text
from .scripts.prediction import classify_review

def get_prediction(request):
    if request.method == 'GET':
        text = request.GET.get('text')
        proc_text = process_text([text])
        rating, sent = classify_review(proc_text)
        prediction = {'Rating': int(rating), 'Class': 'Positive' if sent == 1 else 'Negative'}
        return JsonResponse({'prediction': prediction})
