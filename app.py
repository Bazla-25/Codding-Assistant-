from flask import Flask, render_template, request
from gihub import fetch_news, fetch_url

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        title = request.form['title']
        return render_template('index.html', topic=topic, title=title)
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    topic = request.form['topic']
    title = request.form['title']
    return render_template('index.html', topic=topic, title=title)


if __name__ == '__main__':
    app.run(debug=True)
