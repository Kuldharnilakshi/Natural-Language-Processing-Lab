import re
import pandas as pd


class TextCleaner:
    """Reusable text-cleaning utility using regex patterns."""

    EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w\.-]+')
    URL_PATTERN = re.compile(r'https?://[\S]+|www\.[\S]+')
    PHONE_PATTERN = re.compile(r'\+?\d[\d\s()\-]{7,}\d')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    SPECIAL_CHAR = re.compile(r'[^a-zA-Z0-9\s]')
    MULTI_SPACE = re.compile(r'\s+')

    def extract_emails(self, text):
        return self.EMAIL_PATTERN.findall(text)

    def extract_urls(self, text):
        return self.URL_PATTERN.findall(text)

    def extract_phones(self, text):
        phones = self.PHONE_PATTERN.findall(text)
        return [re.sub(r'[^\d+]', '', p) for p in phones]

    def remove_html(self, text):
        return self.HTML_PATTERN.sub('', text)

    def clean_all(self, text):
        text = self.remove_html(text)
        text = self.URL_PATTERN.sub('[URL]', text)
        text = self.EMAIL_PATTERN.sub('[EMAIL]', text)
        text = self.MENTION_PATTERN.sub('[USER]', text)
        text = self.HASHTAG_PATTERN.sub('', text)
        text = self.SPECIAL_CHAR.sub(' ', text)
        text = self.MULTI_SPACE.sub(' ', text)

        return text.strip().lower()


# Noisy text
noisy_texts = [
    "Check https://openai.com for AI! Contact: info@example.com",
    "@john Great talk at #NLPConf2025! Call me at +91-9876-543-210",
    "Email admin@college.edu.in or visit www.sanjivani.org.in"
]

# Create object
cleaner = TextCleaner()

# Process texts
for text in noisy_texts:
    print("Original:", text)
    print("Cleaned :", cleaner.clean_all(text))
    print("-" * 60)