from src.retrieval.search import search_exercise_by_text

results = search_exercise_by_text('shoulder press chest')
for r in results:
    print(r)
    