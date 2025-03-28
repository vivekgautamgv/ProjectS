# utils/helpers.py (new file)
def get_market_news(category='general'):
    """Fetch market news from NewsAPI"""
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    url = f"https://newsapi.org/v2/everything?q={category}&apiKey={pub_7672884e36707058eb0609bb39a9423f8331f}"
    
    try:
        response = requests.get(url)
        articles = response.json().get('articles', [])
        return [{
            'title': a['title'],
            'source': a['source']['name'],
            'published': a['publishedAt'][:10],
            'summary': a['description'],
            'url': a['url']
        } for a in articles[:10]]
    except:
        return []