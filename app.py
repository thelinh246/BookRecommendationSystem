import streamlit as st
import google.generativeai as genai  
import config  # Import file config.py

# Cấu hình API Key từ config.py
genai.configure(api_key=config.GEMINI_API_KEY)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Chatbot Gemini AI", page_icon="🤖", layout="centered")


# Tiêu đề ứng dụng
st.title("Chatbot AI -BOOKRECOMMENDATION")

# Dùng session state để lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị tin nhắn trước đó
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhập câu hỏi từ người dùng
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    # Hiển thị tin nhắn của người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gửi yêu cầu đến Google Gemini API
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(user_input)

    # Lấy phản hồi từ API
    bot_response = response.text

    # Hiển thị phản hồi của chatbot
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
