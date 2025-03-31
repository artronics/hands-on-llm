import pandas as pd
import os

DATASET_DIR = f"{os.path.dirname(os.path.realpath(__file__))}/../dataset"


def load_playlists(model_path: str) -> [[int]]:
    with open(os.path.join(DATASET_DIR, f"{model_path}/train.txt")) as trained_model:
        lines = trained_model.read().split('\n')[2:]
        playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]
        return playlists


def load_songs(model_path: str) -> pd.DataFrame:
    with open(os.path.join(DATASET_DIR, f"{model_path}/song_hash.txt")) as songs_file:
        content = songs_file.read().split('\n')
        songs = [s.rstrip().split('\t') for s in content]
        df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
        df.set_index('id')
        return df


def train(model_path: str):
    from gensim.models import Word2Vec

    playlists = load_playlists(model_path)
    model = Word2Vec(
        playlists,
        vector_size=32,
        window=20,
        negative=50,
        min_count=1,
        workers=4
    )
    return model


def get_recommendations(song_id: int, model_path: str):
    import numpy as np

    model = train(model_path)
    similar_songs = np.array(
        model.wv.most_similar(positive=str(song_id), topn=5)
    )[:, 0]

    songs = load_songs(model_path)
    return songs.iloc[similar_songs]


def main():
    rec = get_recommendations(1, "music_playlist/yes_complete")
    print(rec)


if __name__ == "__main__":
    main()
