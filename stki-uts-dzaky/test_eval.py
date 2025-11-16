from src.eval import precision, recall, f1

# contoh hasil dan dok relevan
retrieved = ["dok1.txt", "dok2.txt"]
relevant = {"dok1.txt", "dok3.txt"}

p = precision(retrieved, relevant)
r = recall(retrieved, relevant)
f = f1(p, r)

print("Precision:", p)
print("Recall:", r)
print("F1-score:", f)
