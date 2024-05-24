from flask import Flask, request, abort, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import tempfile, os
import datetime
import openai
import time
import traceback
import json

app = Flask(__name__)
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
# Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
# OPENAI API Key初始化設定
openai.api_key = os.getenv('OPENAI_API_KEY')

# 用來存儲使用者 user_id 的文件
USER_ID_FILE = 'user_ids.json'

#從文件中讀取已記錄的 user_id。
def load_user_ids():
    try:
        with open(USER_ID_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# 將所有記錄的使用者 ID 保存到文件中，以便下次可以讀取和使用這些 ID。
def save_user_ids(user_ids):
    with open(USER_ID_FILE, 'w') as f:
        json.dump(user_ids, f) #將使用者 ID 列表寫入文件。

def GPT_response(text):
    # 接收回應
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=text, temperature=0.5, max_tokens=500)
    print(response)
    # 重組回應
    answer = response['choices'][0]['text'].replace('。','')
    return answer

# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# 監聽來自 /notify_fall 的 Post Request
@app.route("/notify_fall", methods=['POST']) #使用 Flask 的路由裝飾器，定義了一個名為 /notify_fall 的路由，該路由只接受 POST 請求。
def notify_fall(): 
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join(static_tmp_path, 'fall_detected.jpg')
    image_file.save(image_path)

    message = request.form.get("message", "Fall detected!") # 从请求的表单数据中获取 message
    user_ids = load_user_ids()  # 加載所有使用者的 user_id

    errors = []
    for user_id in user_ids:
        try:
            image_message = ImageSendMessage(
                original_content_url=f"{request.url_root}static/tmp/fall_detected.jpg",
                preview_image_url=f"{request.url_root}static/tmp/fall_detected.jpg"
            )
            line_bot_api.push_message(user_id, [TextSendMessage(text=message), image_message])
        except Exception as e:
            print(f"Error sending message to {user_id}: {e}")
            errors.append(user_id)

    if errors:
        return jsonify({"status": "partial_error", "errors": errors}), 500
    else:
        return jsonify({"status": "success"}), 200

# 處理使用者加入事件
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id #從事件中獲取使用者的 user_id。
    user_ids = load_user_ids() #用 load_user_ids() 函數讀取已記錄的使用者 ID。
    if user_id not in user_ids:
        user_ids.append(user_id)
        save_user_ids(user_ids)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="歡迎加入跌倒警報系統")) #向新加入的使用者發送歡迎訊息。

# 處理訊息
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    msg = event.message.text
    try:
        GPT_answer = GPT_response(msg)
        print(GPT_answer)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(GPT_answer))
    except:
        print(traceback.format_exc())
        line_bot_api.reply_message(event.reply_token, TextSendMessage('你所使用的OPENAI API key額度可能已經超過，請於後台Log內確認錯誤訊息'))
        
@handler.add(PostbackEvent)
def handle_postback(event):
    print(event.postback.data)

@handler.add(MemberJoinedEvent)
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name}歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)
        
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    os.makedirs(static_tmp_path, exist_ok=True)
    app.run(host='0.0.0.0', port=port)
