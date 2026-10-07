from src.evaluation import calculate_metrics, average_precision

def test_metrics():
    r=calculate_metrics(["TICK-001","TICK-002","TICK-003"],["TICK-001","TICK-003"],3)
    assert round(r["precision"],2)==0.67
    assert r["recall"]==1.0
    assert round(r["f1"],2)==0.80

def test_ap():
    assert round(average_precision(["A","B","C"],["A","C"]),2)==0.83
    assert round(average_precision(["B","A","C"],["A","C"]),2)==0.58
