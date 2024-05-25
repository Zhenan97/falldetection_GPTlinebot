from flask import Flask, request, abort, jsonify, send_file
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import os
import json
import openai
import traceback
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mediapipe as mp

app = Flask(__name__)
static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')
os.makedirs(static_tmp_path, exist_ok=True)

# Channel Access Token and Channel Secret
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# OPENAI API Key初始化設定
openai.api_key = os.getenv('OPENAI_API_KEY')

# 加载训练好的模型
model = load_model('fall_detection_model.h5')

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

def preprocess_frame_sequence(frame_sequence, max_frames=50):
    # 将帧序列填充/截断到max_frames长度
    padded_sequence = pad_sequences(frame_sequence, maxlen=max_frames, padding='post', truncating='post', dtype='float32')
    # 添加额外的维度
    padded_sequence = np.repeat(np.expand_dims(padded_sequence, axis=-1), 3, axis=-1)
    return padded_sequence

# 监听所有来自 /callback 的 Post Request
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

# 监听来自 /notify_fall 的 Post Request
@app.route("/notify_fall", methods=['POST'])
def notify_fall():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join(static_tmp_path, 'fall_detected.jpg')
    image_file.save(image_path)

    message = request.form.get("message", "Fall detected!") # 从请求的表单数据中获取 message
    user_ids = load_user_ids()  # 加载所有使用者的 user_id

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

# 处理从本地发送的帧
@app.route("/process_frame", methods=['POST'])
def process_frame():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 初始化MediaPipe姿态模型
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    frame_sequence = []

    # 将图像转换为RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #设置 image.flags.writeable = False 可以防止对图像数组的修改
    image.flags.writeable = False

    # 处理图像并检测姿态
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制检测到的骨架
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) #mp_pose.POSE_CONNECTIONS：这是一个包含姿态关键点连接关系的常量，定义了关键点之间的连接。

        # 提取骨架点
        skeleton_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
        frame_sequence.append(skeleton_points)

        # 如果序列长度超过max_frames，删除最旧的帧
        if len(frame_sequence) > 50:
            frame_sequence.pop(0)

        # 如果序列长度足够，进行预测
        if len(frame_sequence) == 50:
            # 预处理帧序列
            #原始输入数据是一个形状为 (50, 33, 3)
            input_sequence = preprocess_frame_sequence([frame_sequence])
            # 直接使用 input_sequence 进行预测,通过 np.expand_dims 将其变为 (50, 33, 3, 3)
            prediction = model.predict(input_sequence)
            fall_detected = prediction[0] > 0.5

            # 在帧上显示预测结果
            if fall_detected:
                cv2.putText(image, "Fall Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 保存跌倒图像
                cv2.imwrite(os.path.join(static_tmp_path, 'fall_detected.jpg'), frame)
                
                # 发送通知到 Line Bot
                message = "Fall detected!"
                user_ids = load_user_ids()
                for user_id in user_ids:
                    try:
                        image_message = ImageSendMessage(
                            original_content_url=f"{request.url_root}static/tmp/fall_detected.jpg",
                            preview_image_url=f"{request.url_root}static/tmp/fall_detected.jpg"
                        )
                        line_bot_api.push_message(user_id, [TextSendMessage(text=message), image_message])
                    except Exception as e:
                        print(f"Error sending message to {user_id}: {e}")
            else:
                cv2.putText(image, "No Fall", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return jsonify({"status": "success"}), 200

#返回最新捕获的图像。
@app.route("/latest_image", methods=['GET'])
def latest_image():
    try:
        return send_file('latest_image.jpg', mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500
    
# 处理使用者加入事件
@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    user_ids = load_user_ids()
    if user_id not in user_ids:
        user_ids.append(user_id)
        save_user_ids(user_ids)
    buttons_template = ButtonsTemplate(
        title='Welcome',
        text='歡迎加入跌倒警報系統，請選擇操作',
        actions=[
            PostbackAction(label='查看即時影像', data='show_image')
        ]
    )
    template_message = TemplateSendMessage(
        alt_text='Buttons alt text', template=buttons_template)
    line_bot_api.reply_message(event.reply_token, template_message)
    
# 处理消息
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
    data = event.postback.data
    if data == 'show_image':
        image_url = f"{request.url_root}latest_image"
        line_bot_api.reply_message(
            event.reply_token,
            ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
        )
        
def welcome(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f'{name}歡迎加入')
    line_bot_api.reply_message(event.reply_token, message)
        
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
