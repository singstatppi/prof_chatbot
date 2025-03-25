import streamlit as st
st.sidebar.page_link("./chat_app.py", label="Home")
st.sidebar.page_link("pages/about_us.py", label="About Us")
st.sidebar.page_link("pages/contact_us.py", label="Contact Us")

st.title("Contact Us")
st.markdown("For any inquiries or technical assistance, please contact us at: [technical@singstat.gov.sg]")