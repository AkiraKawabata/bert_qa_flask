from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_model import QA_data
from transformers import BertJapaneseTokenizer, AutoModelForQuestionAnswering
import torch


app = Flask(__name__)



engine = create_engine('sqlite:///qa.db')
session = sessionmaker(bind=engine)()


# getのときの処理
@app.route('/', methods=['GET'])
def get():
	return render_template('index.html', \
		title = 'BertQA API', \
		message = 'contextと質問を入力してください')


# postされたデータをもとにDBに値を登録する。
@app.route('/register', methods=['POST'])
def register():
	context = request.form['context']
	query =  request.form['query']

	# モデルとトークナイザーの準備
	model = AutoModelForQuestionAnswering.from_pretrained('output/')  
	tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking') 

	# 推論の実行
	inputs = tokenizer.encode_plus(query, context, add_special_tokens=True, return_tensors="pt")
	input_ids = inputs["input_ids"].tolist()[0]
	output = model(**inputs)
	answer_start = torch.argmax(output.start_logits)  
	answer_end = torch.argmax(output.end_logits) + 1 
	answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

	new_data = QA_data(query = query, context = context, answer = answer)
	session.add(new_data)
	session.commit()
	return redirect(url_for('database'))

'''
	return render_template('index.html', \
		title = '{} への答え'.format(query), \
		message = '答え {}'.format(answer))
'''


# /database での値取得
@app.route("/database", methods = ['GET'])
def database():
    data = session.query(QA_data).all()
    return render_template('data.html', qa_data=data)


if __name__ == "__main__":
	app.run(debug=True)