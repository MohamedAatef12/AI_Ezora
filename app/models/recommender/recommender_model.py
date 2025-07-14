import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout


def build_recommender_model(
    num_moods: int,
    num_occasions: int,
    num_categories: int,
    num_products: int,
    embedding_dim: int = 16,
) -> Model:
    """
    Builds a recommendation model using categorical inputs and embeddings.

    Args:
        num_moods (int): Number of unique moods.
        num_occasions (int): Number of unique occasions.
        num_categories (int): Number of unique category combinations.
        num_products (int): Total number of products to predict.
        embedding_dim (int): Size of the embedding vectors.

    Returns:
        tf.keras.Model: Compiled recommendation model.
    """

    # Inputs
    mood_input = Input(shape=(1,), name="mood_input")
    occasion_input = Input(shape=(1,), name="occasion_input")
    category_input = Input(shape=(1,), name="category_input")

    # Embeddings
    mood_emb = Embedding(input_dim=num_moods, output_dim=embedding_dim)(mood_input)
    occasion_emb = Embedding(input_dim=num_occasions, output_dim=embedding_dim)(occasion_input)
    category_emb = Embedding(input_dim=num_categories, output_dim=embedding_dim)(category_input)

    # Flatten embeddings
    mood_vec = Flatten()(mood_emb)
    occasion_vec = Flatten()(occasion_emb)
    category_vec = Flatten()(category_emb)

    # Merge all embeddings
    x = Concatenate()([mood_vec, occasion_vec, category_vec])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)

    # Output layer: softmax over product classes
    output = Dense(num_products, activation="softmax", name="product_output")(x)

    # Build model
    model = Model(
        inputs=[mood_input, occasion_input, category_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
