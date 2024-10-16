from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
sms_data = [
 "Free entry in 2 a weekly competition to win FA Cup final tickets",
 "Hey, I will call you later. Don't forget to bring the document.",
 "Congratulations! You've won a free cruise to the Bahamas",
 "Hi there, can we meet tomorrow for lunch?",
 "URGENT! Your mobile number has won a $2000 prize!",
 "Reminder: Your appointment with the dentist is at 3 PM today.",
 "You have won a lottery! Claim your prize now by calling us.",
 "Are we still meeting at the coffee shop today?",
 "Exclusive deal just for you! Buy now and get 50% off!",
 "Can you send me the report by end of the day?"
]
sms_labels = [
 "spam", "ham", "spam", "ham", "spam",
 "ham", "spam", "ham", "spam", "ham"
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms_data)
X_train, X_test, y_train, y_test = train_test_split(X, sms_labels, test_size=0.3,
random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test, y_pred)
print("Test Data:")
for doc, actual, predicted in zip(sms_data[len(sms_data) - len(y_test):], y_test,
y_pred):
 print(f"Message: {doc}, Actual: {actual}, Predicted: {predicted}")

print(f"\nAccuracy: {accuracy:.2f}")
print(report)
print("Deep Marathe -53004230016")