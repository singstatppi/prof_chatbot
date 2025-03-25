import streamlit as st
st.sidebar.page_link("./chat_app.py", label="Home")
st.sidebar.page_link("pages/about_us.py", label="About Us")
st.sidebar.page_link("pages/contact_us.py", label="Contact Us")

st.title("About Us")
st.markdown("This is a chat assistant designed to help with queries related to economic indicators, data analysis, and more. "
            "Feel free to ask questions and explore the capabilities of this assistant.")