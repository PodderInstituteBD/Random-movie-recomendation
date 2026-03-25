import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

df = pd.read_csv("movie-remondation_system/imdb_top_1000.csv")

df['Overview'] = df['Overview'].fillna('')
df['Genre'] = df['Genre'].fillna('')
df['Series_Title'] = df['Series_Title'].fillna('')

df['combined'] = (
    df['Series_Title'] + " " +
    df['Genre'] + " " +
    df['Overview']
)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def show_movie(index):
    movie = df.iloc[index]
    sentiment = get_sentiment(movie['Overview'])

    print("\n🎬 Movie Recommendation")
    print("-" * 50)
    print(f"Title       : {movie['Series_Title']}")
    print(f"Genre       : {movie['Genre']}")
    print(f"IMDB Rating : {movie['IMDB_Rating']}")
    print(f"Overview    : {movie['Overview'][:200]}...")
    print(f"Sentiment   : {round(sentiment, 2)}")
    print("-" * 50)

def recommend(movie_title):
    movie_title = movie_title.lower()

    matches = df[df['Series_Title'].str.lower().str.contains(movie_title)]

    if matches.empty:
        print("❌ Movie not found.")
        return

    idx = matches.index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print("\n🔥 Top 5 Recommendations:")
    for i in scores[1:6]:
        show_movie(i[0])

def random_movie():
    index = random.randint(0, len(df) - 1)
    show_movie(index)

def recommend_by_genre():
    genre = input("Enter genre (e.g. Action, Drama): ").lower()
    results = df[df['Genre'].str.lower().str.contains(genre)]

    if results.empty:
        print("❌ No movies found for this genre.")
        return

    print("\n🎯 Genre-based Recommendations:")
    for i in results.sample(min(5, len(results))).index:
        show_movie(i)

def main():
    while True:
        print("\n===== 🎬 Movie Recommendation System =====")
        print("1. AI Recommendation")
        print("2. Random Recommendation")
        print("3. Genre-based Recommendation")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter movie name: ")
            recommend(name)

        elif choice == "2":
            random_movie()

        elif choice == "3":
            recommend_by_genre()

        elif choice == "4":
            print("Goodbye 👋")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
