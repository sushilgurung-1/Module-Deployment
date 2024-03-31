from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

@app.route("/")
def home():
    # text=""
    # if request.method == "POST":
    #     text = request.form.get("email-content") # we fetch the text form the textarea
    return render_template("index.html",)


@app.route("/predict", methods=["Post"])
def predict():

    email_text = request.form.get("email-content")
    tokenized_email = tokenizer.transform([email_text]) # x
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return render_template("index.html", predictions=predictions,text=email_text)


if  __name__ == "__main__":
    app.run(debug=True)