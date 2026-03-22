import time
import os
import streamlit as st
import requests
from urllib.parse import urlparse
import streamlit as st
import requests
from app.utils.exception_handler import handle_exception


class Helper:
    def typewriter_effect(text: str, speed: int):
        try:
            lines = text.split("\n") 
            container = st.empty()
            full_text = ""

            for line in lines:
                words = line.split() 
                for index in range(len(words) + 1):
                    curr_line = " ".join(words[:index])
                    container.markdown(full_text + curr_line)  
                    time.sleep(1 / speed)
                # Add line break for Markdown
                full_text += curr_line + "  \n" 
        except Exception as e:
            return handle_exception(e)


    def validate_url(url):
        try:
            if not url:
                return False, "Please enter a URL."

            # Parse the URL
            result = urlparse(url)

            # Check if URL has a scheme and netloc
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format. Please include http:// or https://"

            # Check if it's http or https
            if result.scheme not in ["http", "https"]:
                return False, "URL must start with http:// or https://"

            # Try to fetch the URL to verify it's accessible
            try:
                headers = {'User-Agent': os.getenv('USER_AGENT', 'WebChat/1.0')}
                response = requests.head(url, headers=headers, timeout=5)
                if response.status_code >= 400:
                    return (False, f"Unable to access the website. Status code: {response.status_code}",)
            except requests.RequestException as e:
                return False, f"Error accessing the website: {str(e)}"

            return True, ""

        except Exception as e:
            return handle_exception(e)
