import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout

def build_recommender_model(
    num_moods,
    num_occasions,
    num_categories,
    num_products,
    embedding_dim=16,
):
    # Inputs
    mood_input = Input(shape=(1,), name="mood_input")
    occasion_input = Input(shape=(1,), name="occasion_input")
    category_input = Input(shape=(1,), name="category_input")

    # Embeddings
    mood_emb = Embedding(input_dim=num_moods, output_dim=embedding_dim)(mood_input)
    occasion_emb = Embedding(input_dim=num_occasions, output_dim=embedding_dim)(occasion_input)
    category_emb = Embedding(input_dim=num_categories, output_dim=embedding_dim)(category_input)

    # Flatten
    mood_vec = Flatten()(mood_emb)
    occasion_vec = Flatten()(occasion_emb)
    category_vec = Flatten()(category_emb)

    # Concatenate
    x = Concatenate()([mood_vec, occasion_vec, category_vec])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    # Output layer (softmax over all products)
    output = Dense(num_products, activation="softmax", name="product_output")(x)

    model = Model(inputs=[mood_input, occasion_input, category_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

